"""
Base model classes and utilities for ICU LLaMA models.

Shared transformer and sampling code for the conditional ICU generators.
"""

import math
from abc import ABC, abstractmethod
from functools import partial
from types import SimpleNamespace
import os
import pandas as pd
import numpy as np

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer
    from sdv.sampling import Condition
except ImportError:
    SingleTableMetadata = None
    TVAESynthesizer = None
    Condition = None

from syntheticicu.demo_utils import (
    compute_sex_prevalence,
    demographics_to_prompt_tensor,
    sample_sex_prior,
)
import hydra
import joblib
import lightning as L
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache, DynamicLayer
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
import time


def _to_cpu(obj):
    """Recursively move tensors in nested lists/tuples to CPU."""
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(o) for o in obj)
    return obj


def _atomic_dump(obj, path: str):
    """joblib.dump via a temp file so a kill mid-write never leaves a corrupt file."""
    tmp_path = path + ".tmp"
    joblib.dump(obj, tmp_path)
    os.replace(tmp_path, path)


def batch_file_path(checkpoint_dir: str, batch_index: int) -> str:
    """Where generated batch `batch_index` of a corpus is saved."""
    return os.path.join(checkpoint_dir, f"batch_{batch_index:05d}.joblib")


# The per-row tensors that fix a corpus: saved in meta.joblib on the first run.
PROMPT_NAMES = ("input_ids_all", "label_ids_all", "center_ids_all")


def load_generation_state(checkpoint_dir, run_meta):
    """The saved meta.joblib of an interrupted run, or None for a new corpus.

    Raises when it was written with other settings: its prompts and batches
    would not belong to the corpus now requested.
    """
    if checkpoint_dir is None:
        return None
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_path = os.path.join(checkpoint_dir, "meta.joblib")
    if not os.path.exists(meta_path):
        return None
    resume_state = joblib.load(meta_path)
    mismatched = {
        k: (resume_state["run_meta"].get(k), v)
        for k, v in run_meta.items()
        if resume_state["run_meta"].get(k) != v
    }
    if mismatched:
        raise ValueError(
            f"Checkpoint dir {checkpoint_dir} was created with different "
            f"settings (saved, current): {mismatched}. Remove it to start over."
        )
    print(f"[Resume] Loaded generation state from {meta_path}", flush=True)
    return resume_state


def assert_prompts_consistent(prompts, total_samples):
    """Every prompt tensor has one row per sample, and labels are 0 or 1."""
    assert set(prompts) == set(PROMPT_NAMES)
    assert prompts["input_ids_all"].ndim == 2
    assert prompts["input_ids_all"].shape[1] >= 1
    for name in PROMPT_NAMES:
        assert prompts[name].shape[0] == total_samples, name
    assert torch.isin(
        prompts["label_ids_all"], prompts["label_ids_all"].new_tensor([0, 1])
    ).all()


def tabular_demographic_prompt_ids(gen_demo, num_control, num_case, vocabulary):
    """[female age height weight GAP] token rows sampled from a tabular model.

    Rows follow the prompt order, controls (death_icu=0) then cases.
    """
    conditions = []
    if num_control > 0:
        conditions.append(
            Condition(num_rows=num_control, column_values={"death_icu": 0})
        )
    if num_case > 0:
        conditions.append(Condition(num_rows=num_case, column_values={"death_icu": 1}))
    df_demo = (
        gen_demo.sample_from_conditions(conditions, max_tries_per_batch=1000)
        .sort_values("death_icu")
        .reset_index(drop=True)
    )
    # Drop outcome column before converting to tokens
    df_demo_tokens = df_demo.drop(columns=["death_icu"])
    prompt_ids = demographics_to_prompt_tensor(df_demo_tokens, vocabulary).to(
        torch.long
    )
    total_samples = num_control + num_case
    assert (
        prompt_ids.shape[0] == total_samples
    ), f"Demographic model returned {prompt_ids.shape[0]} rows, expected {total_samples}"
    return prompt_ids


def center_ids_for_strategy(
    num_control,
    num_case,
    center_id,
    center_strategy,
    num_centers,
    null_center_id,
    device,
):
    """Centre id per prompt row (controls, then cases).

    An explicit center_id gives every row that centre; "balanced" cycles the
    centres within each outcome class, so both classes see every centre in
    equal share; anything else gives the null centre (no conditioning).
    """
    total_samples = num_control + num_case
    if center_id is not None:
        return torch.full(
            (total_samples,), int(center_id), dtype=torch.long, device=device
        )
    if center_strategy == "balanced":
        center_control = torch.arange(num_control, device=device) % num_centers
        center_case = torch.arange(num_case, device=device) % num_centers
        center_ids = torch.cat([center_control, center_case])
        print(center_ids)
        return center_ids
    return torch.full((total_samples,), null_center_id, dtype=torch.long, device=device)


def generate_or_load_batches(
    model, batch_count, batch_size, prompts, base_seed, batch_options, checkpoint_dir
):
    """All batches of a corpus in order: loaded when saved, generated otherwise.

    With checkpoint_dir set, a generated batch is saved before the next one
    starts, so an interrupted run loses at most the batch in progress.
    """
    assert batch_count >= 1
    all_samples, all_attentions, all_scores = [], [], []
    for i in tqdm(range(batch_count), desc="Generating batches"):
        batch_path = (
            batch_file_path(checkpoint_dir, i) if checkpoint_dir is not None else None
        )
        if batch_path is not None and os.path.exists(batch_path):
            samples, attentions, scores = joblib.load(batch_path)
        else:
            samples, attentions, scores = generate_batch(
                model, i, batch_size, prompts, base_seed, batch_options
            )
            if batch_path is not None:
                samples, attentions, scores = _to_cpu((samples, attentions, scores))
                _atomic_dump((samples, attentions, scores), batch_path)
        all_samples.extend(samples)
        all_attentions.extend(attentions)
        if scores:
            all_scores.extend(scores)
    return all_samples, all_attentions, all_scores


def generate_batch(model, batch_index, batch_size, prompts, base_seed, batch_options):
    """Generate one corpus batch: rows [batch_index * batch_size, +batch_size).

    Every path that generates a batch (the single-GPU loop in generate_batched
    and the multi-GPU workers) goes through here, so the seed and the rows of
    batch i cannot differ between them, and a split run equals a single run.

    prompts holds input_ids_all, label_ids_all and center_ids_all on the
    model's device; batch_options are keyword arguments of
    _generate_single_batch.
    """
    sample_count = prompts["input_ids_all"].shape[0]
    start = batch_index * batch_size
    assert 0 <= start < sample_count
    end = min(start + batch_size, sample_count)
    # Per-batch seed: batch i draws the same tokens whether the run was resumed
    # or split across GPUs, and no two batches share a random stream.
    torch.manual_seed(base_seed + batch_index)
    return model._generate_single_batch(
        input_ids=prompts["input_ids_all"][start:end],
        label_ids=prompts["label_ids_all"][start:end],
        center_ids=prompts["center_ids_all"][start:end],
        **batch_options,
    )


def generate_batches_on_device(
    model,
    device_index,
    batch_indices,
    batch_size,
    prompts,
    base_seed,
    batch_options,
    checkpoint_dir,
):
    """Worker process body: generate and save the given batches on one GPU.

    Batches whose file already exists are skipped, so a rerun after a time
    limit continues where the last run stopped.
    """
    assert 0 <= device_index < torch.cuda.device_count()
    assert list(batch_indices) == sorted(set(batch_indices))
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    prompts = {name: tensor.to(device) for name, tensor in prompts.items()}
    with torch.no_grad():
        for batch_index in batch_indices:
            path = batch_file_path(checkpoint_dir, batch_index)
            if os.path.exists(path):
                continue
            batch = generate_batch(
                model, batch_index, batch_size, prompts, base_seed, batch_options
            )
            _atomic_dump(_to_cpu(batch), path)
            print(f"[Generate] GPU {device_index}: saved {path}", flush=True)


def generate_batches_on_devices(
    model,
    device_indices,
    batch_count,
    batch_size,
    prompts,
    base_seed,
    batch_options,
    checkpoint_dir,
):
    """Write every missing batch file of a corpus, one worker process per GPU.

    Batch b goes to device_indices[b % len(device_indices)]. Batches take about
    the same time (each runs until its longest row ends), so this balances the
    GPUs. Workers are spawned processes: CUDA cannot be used in forked ones, and
    each worker gets its own Python interpreter for its decoding loop. The model
    and prompts are pickled to every worker, so pass them on the CPU and without
    the dataset (see ICULlamaBase._generate_batches_in_workers).
    """
    assert len(device_indices) >= 1
    assert len(set(device_indices)) == len(device_indices)
    assert batch_count >= 1
    batch_indices_per_worker = [
        list(range(worker_index, batch_count, len(device_indices)))
        for worker_index in range(len(device_indices))
    ]
    # Every batch goes to exactly one worker.
    assert sorted(sum(batch_indices_per_worker, [])) == list(range(batch_count))
    os.makedirs(checkpoint_dir, exist_ok=True)
    context = torch.multiprocessing.get_context("spawn")
    workers = []
    for device_index, batch_indices in zip(device_indices, batch_indices_per_worker):
        worker = context.Process(
            target=generate_batches_on_device,
            args=(
                model,
                device_index,
                batch_indices,
                batch_size,
                prompts,
                base_seed,
                batch_options,
                checkpoint_dir,
            ),
            name=f"generate-gpu{device_index}",
        )
        worker.start()
        workers.append(worker)

    # No timeout on join: a corpus takes as long as it takes, and the bound is
    # the job's Slurm time limit, before which run_generation.sh requeues the
    # job. Finished batches are on disk, so the rerun only does the rest.
    failed = []
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            failed.append(f"{worker.name} (exit code {worker.exitcode})")
    if failed:
        raise RuntimeError(
            f"Generation worker failed: {', '.join(failed)}. Finished batches are "
            f"saved in {checkpoint_dir}; rerun to continue from them."
        )
    for batch_index in range(batch_count):
        assert os.path.exists(batch_file_path(checkpoint_dir, batch_index))


def compute_sinusoidal_embeddings(
    positions: torch.Tensor, d_model: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute sinusoidal embeddings for arbitrary position indices.

    Args:
        positions: Tensor of position indices.
        d_model: Model hidden dimension.
        dtype: Output embedding dtype.

    Returns:
        Tensor with the position shape followed by the model dimension.
    """
    positions = positions.to(dtype=torch.float32)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    angles = positions.unsqueeze(-1) * div_term
    embeddings = torch.empty(
        *positions.shape, d_model, device=positions.device, dtype=torch.float32
    )
    embeddings[..., 0::2] = torch.sin(angles)
    embeddings[..., 1::2] = torch.cos(angles)
    return embeddings.to(dtype=dtype)


def create_sinusoidal_embeddings(max_len: int, d_model: int) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings like in the original Transformer paper.

    Args:
        max_len: Maximum sequence length
        d_model: Model hidden dimension

    Returns:
        Tensor of shape (max_len, d_model) with sinusoidal embeddings
    """
    return compute_sinusoidal_embeddings(torch.arange(max_len), d_model)


def get_llama_model(model_name: str, vocabulary: dict, **kwargs):
    """
    Create a LlamaForCausalLM model with the specified configuration.

    Args:
        model_name: One of 'llama-xs' or 'llama-s'
        vocabulary: Token vocabulary dict
        **kwargs: Additional config arguments

    Returns:
        Configured LlamaForCausalLM instance
    """
    model_sizes = {
        "llama-xs": {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 4 * 512,
            "n_embd": 512,
            "n_head": 8,
            "n_layer": 6,
        },
        "llama-s": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 4 * 768,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
        },
    }
    model_size_config = model_sizes[model_name]
    model_config = {
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 4096,
        # "attn_implementation": "eager", #spda doesnt returns attentions, to debug
    }

    vocab_config = {
        "vocab_size": len(vocabulary),
        "add_bos_token": False,
        "bos_token_id": vocabulary["<SOS>"],
        "eos_token_id": vocabulary["<EOS>"],
        "pad_token_id": vocabulary["<PAD>"],
    }

    config = transformers.LlamaConfig(
        **vocab_config, **model_config, **model_size_config, **kwargs
    )
    return transformers.LlamaForCausalLM(config)


class PreallocatedKVLayer(DynamicLayer):
    """A DynamicLayer whose keys and values live in buffers allocated once.

    DynamicLayer grows its cache with torch.cat, which copies the whole cache
    on every decoding step: at 5000 steps that copying dominates the step time.
    Here the first update allocates [batch, heads, capacity, head_dim] buffers,
    each update writes the new positions into them, and keys/values are views
    of the filled part. Attention therefore sees exactly the keys and values a
    DynamicLayer would hold, and the mask sizes (get_seq_length) are the same.
    """

    def __init__(self, capacity: int):
        super().__init__()
        assert capacity > 0
        self.capacity = capacity
        self.length = 0

    def lazy_initialization(self, key_states, value_states) -> None:
        assert key_states.ndim == 4  # [batch, heads, positions, head_dim]
        assert value_states.shape[:3] == key_states.shape[:3]
        self.dtype, self.device = key_states.dtype, key_states.device
        batch_size, head_count, _, head_dim = key_states.shape
        buffer_shape = (batch_size, head_count, self.capacity, head_dim)
        self.key_buffer = torch.empty(
            buffer_shape, dtype=self.dtype, device=self.device
        )
        self.value_buffer = torch.empty(
            (batch_size, head_count, self.capacity, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=self.device,
        )
        self.keys = self.key_buffer[:, :, :0, :]
        self.values = self.value_buffer[:, :, :0, :]
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        assert key_states.shape[:2] == self.key_buffer.shape[:2]
        assert key_states.shape[-2] == value_states.shape[-2]
        length_new = self.length + key_states.shape[-2]
        if length_new > self.capacity:
            raise RuntimeError(
                f"KV cache capacity {self.capacity} exceeded ({length_new} positions)"
            )
        self.key_buffer[:, :, self.length : length_new, :] = key_states
        self.value_buffer[:, :, self.length : length_new, :] = value_states
        self.length = length_new
        self.keys = self.key_buffer[:, :, :length_new, :]
        self.values = self.value_buffer[:, :, :length_new, :]
        return self.keys, self.values


def preallocated_kv_cache(layer_count: int, capacity: int) -> Cache:
    """A KV cache for `capacity` positions per layer that never copies on growth."""
    assert layer_count > 0
    return Cache(layers=[PreallocatedKVLayer(capacity) for _ in range(layer_count)])


# hard
class MoELMHead(nn.Module):
    """
    Mixture of Experts Language Model Head.

    Training: Top-2 soft routing — stable gradients, experts learn specialization.
    Inference: Top-1 hard routing — full commitment to one expert per token,
               allows capturing abrupt clinical transitions (trigger events).
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.gating_network = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_experts)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        training: bool = True,
    ):
        # hidden_states: (B, T, H)

        gating_scores = self.gating_network(hidden_states)  # (B, T, num_experts)

        if training:
            # Top-2 soft: each token is routed to the 2 most relevant experts
            # Softmax only over those 2 → clean gradients, gradual specialization
            top2_values, top2_indices = torch.topk(gating_scores, k=2, dim=-1)
            top2_weights = torch.softmax(top2_values, dim=-1)  # (B, T, 2)

            gating_weights = torch.zeros_like(gating_scores)  # (B, T, num_experts)
            gating_weights.scatter_(-1, top2_indices, top2_weights)
        else:
            # Top-1 hard: full commitment to ONE expert per token
            # When the hidden state changes (trigger event), the expert changes
            # top1_indices = gating_scores.argmax(dim=-1, keepdim=True)  # (B, T, 1)
            # gating_weights = torch.zeros_like(gating_scores)
            # gating_weights.scatter_(-1, top1_indices, 1.0)
            # Inference: Top-2 with Low Temperature (Sharp Soft Routing)
            temperature = 0.5  # Hyperparameter to tune (lower = closer to Top-1)
            scaled_logits = gating_scores / temperature

            top2_values, top2_indices = torch.topk(scaled_logits, k=2, dim=-1)
            top2_weights = torch.softmax(
                top2_values, dim=-1
            )  # This will produce very sharp weights, e.g.: [0.95, 0.05]

            gating_weights = torch.zeros_like(gating_scores)
            gating_weights.scatter_(-1, top2_indices, top2_weights)

        if attention_mask is not None:
            # set gating weights to zero for padding tokens to avoid skewing expert usage stats
            masked_weights = gating_weights * attention_mask.unsqueeze(-1)
            # mean expert usage across batch and sequence, weighted by number of valid tokens
            total_tokens = attention_mask.sum()
            expert_usage = masked_weights.sum(dim=(0, 1)) / (total_tokens + 1e-6)
        else:
            expert_usage = gating_weights.mean(dim=(0, 1))

        # Combine experts
        gating_weights_expanded = gating_weights.unsqueeze(-1)  # (B, T, num_experts, 1)
        expert_logits = torch.stack(
            [expert(hidden_states) for expert in self.experts], dim=2
        )  # (B, T, num_experts, V)

        combined_logits = torch.sum(gating_weights_expanded * expert_logits, dim=2)
        # (B, T, V)

        return combined_logits, gating_weights, expert_usage


class LlamaWithTimePos(LlamaForCausalLM):
    """
    Llama model extended with time positional embeddings.

    This class handles:
    - Adding sinusoidal time embeddings to token embeddings
    - Tracking time positions during autoregressive generation by counting GAP tokens
    - Optional gap prediction head for multi-task learning
    """

    def __init__(
        self,
        config,
        use_time_embeddings=False,
        gap_token_id=None,
        gap_head=None,
        num_experts=4,
    ):
        super().__init__(config)
        self.use_time_embeddings = use_time_embeddings
        self.gap_token_id = gap_token_id
        self.gap_head = gap_head

        # Replace standard linear lm_head with our MoE head
        self.num_experts = num_experts
        self.lm_head = MoELMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_experts=self.num_experts,
        )

        self.config.tie_word_embeddings = False  # We can't tie embeddings with a custom MoE head, so we disable it in the config to avoid warnings.

        self.loss = partial(
            ForCausalLMLoss,
            vocab_size=self.config.vocab_size,
            ignore_index=self.config.pad_token_id,
        )

    def compute_loss(self, logits, labels, **kwargs):
        return self.loss(
            logits=logits,
            labels=labels,
            **kwargs,
        )

    def _compute_time_positions_from_tokens(self, input_ids):
        """
        Compute time positions by counting GAP tokens in the sequence.
        Time starts at 1 and increments by 1 after each GAP token.
        """
        batch_size, seq_len = input_ids.shape
        time_positions = torch.zeros_like(input_ids, dtype=torch.long)

        for b in range(batch_size):
            current_time = 1
            for i in range(seq_len):
                time_positions[b, i] = current_time
                if (
                    self.gap_token_id is not None
                    and input_ids[b, i] == self.gap_token_id
                ):
                    current_time += 1

        return time_positions

    def prepare_inputs_for_generation(
        self, input_ids=None, inputs_embeds=None, past_key_values=None, **kwargs
    ):
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
        elif "time_positions" in kwargs and kwargs["time_positions"] is not None:
            tp = kwargs["time_positions"]
            batch_size, seq_len = tp.shape
            device = tp.device
        elif past_key_values is not None:
            k0 = past_key_values[0][0]
            batch_size = k0.shape[0]
            seq_len = 1
            device = k0.device

        if self.use_time_embeddings:
            if "time_positions" not in kwargs or kwargs["time_positions"] is None:
                if past_key_values is None:
                    if input_ids is not None and self.gap_token_id is not None:
                        time_positions = self._compute_time_positions_from_tokens(
                            input_ids
                        )
                        kwargs["time_positions"] = time_positions
                    else:
                        time_positions = (
                            torch.arange(seq_len, device=device, dtype=torch.long)
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                        kwargs["time_positions"] = time_positions
                else:
                    if not hasattr(self, "_current_time_position"):
                        self._current_time_position = torch.ones(
                            (batch_size,), device=device, dtype=torch.long
                        )

                    if (
                        input_ids is not None
                        and self.gap_token_id is not None
                        and input_ids.shape[1] > 0
                    ):
                        last_token = input_ids[:, -1]
                        is_gap = last_token == self.gap_token_id
                        self._current_time_position = torch.where(
                            is_gap,
                            self._current_time_position + 1,
                            self._current_time_position,
                        )

                    time_positions = self._current_time_position.unsqueeze(1)
                    kwargs["time_positions"] = time_positions

        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            **kwargs,
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        time_positions=None,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        past_key_values=None,
        labels=None,
        **kwargs,
    ):
        if inputs_embeds is None and input_ids is not None:
            embeddings = self.model.embed_tokens(input_ids)
        else:
            embeddings = inputs_embeds

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = (input_ids != self.config.pad_token_id).long()
            else:
                batch_size, seq_len, _ = embeddings.shape
                attention_mask = torch.ones(
                    (batch_size, seq_len), dtype=torch.long, device=embeddings.device
                )

        if self.use_time_embeddings and time_positions is not None:
            if (
                past_key_values is not None
                and input_ids is not None
                and input_ids.shape[1] == 1
            ):
                time_embeds = compute_sinusoidal_embeddings(
                    time_positions[:, -1],
                    self.config.hidden_size,
                    dtype=embeddings.dtype,
                )
                time_embeds = time_embeds.unsqueeze(1)
            else:
                time_embeds = compute_sinusoidal_embeddings(
                    time_positions,
                    self.config.hidden_size,
                    dtype=embeddings.dtype,
                )
            embeddings = embeddings + time_embeds

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=embeddings,
            past_key_values=past_key_values,
            use_cache=kwargs.get("use_cache", False),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,  # it will return the last hidden state in outputs.hidden_states[-1] even if output_hidden_states is False
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # (Batch, Seq, Hidden)
        # logits = self.lm_head(hidden_states)
        # logits = logits.float()

        logits, gating_weights, real_expert_usage = self.lm_head(
            hidden_states, attention_mask=attention_mask, training=self.training
        )
        logits = logits.float()

        # Save expert usage dynamically so the LightningModule can log it later
        self.current_expert_usage = real_expert_usage
        self.current_routing_probs = real_expert_usage

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, **kwargs)

        # The gap head is not run here: ICULlamaGap._compute_gap_loss applies it
        # to the returned hidden states, and its output here was never used.

        if output_hidden_states:
            returned_hidden_states = outputs.hidden_states
        else:
            returned_hidden_states = (hidden_states,)

        # Return clean CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=returned_hidden_states,
            attentions=outputs.attentions,
        )


class ICULlamaBase(L.LightningModule, ABC):
    """
    Base class for ICU LLaMA models.

    Provides shared functionality for:
    - DataModule instantiation
    - Time embedding setup
    - Training/validation/test steps
    - Optimizer configuration
    - Token sequence processing
    - Sample prevalence computation

    Subclasses must implement:
    - instantiate_model(): Model-specific initialization
    - forward(): Forward pass with model-specific conditioning
    - shared_step(): Loss computation with model-specific logic
    - generate(): Generation with model-specific conditioning
    """

    def __init__(self, cfg, new_cfg=None):
        super().__init__()

        self.cfg = cfg
        if new_cfg is not None:
            self.cfg = new_cfg

        if "tokenizer" not in self.cfg:
            self.instantiate_datamodule()
        else:
            self._datamodule = None
            self.vocabulary = joblib.load(self.cfg.tokenizer.vocab)
            self.id_to_token = joblib.load(self.cfg.tokenizer.id_to_token)

        self.instantiate_model(cfg)
        self.save_hyperparameters()

    def instantiate_datamodule(self):
        """Instantiate the DataModule and extract tokenizer info."""
        self._datamodule = hydra.utils.instantiate(self.cfg.datamodule)
        self.vocabulary = self._datamodule.tokenizer.vocab
        self.id_to_token = self._datamodule.tokenizer.id_to_token

    def _setup_time_embeddings(self, config):
        """
        Set up time positional embeddings if enabled in datamodule.

        Args:
            config: Model configuration with hidden_size
        """
        if (
            hasattr(self._datamodule, "add_time_point_positional_embeddings")
            and self._datamodule.add_time_point_positional_embeddings
        ):
            self.use_time_embeddings = True
            self.time_embedding_dim = config.hidden_size
        else:
            self.use_time_embeddings = False

    def on_load_checkpoint(self, checkpoint):
        """Discard the obsolete deterministic time-embedding lookup table."""
        checkpoint["state_dict"].pop("time_positional_embeddings", None)

    @abstractmethod
    def instantiate_model(self, cfg):
        """
        Instantiate the model with config-specific settings.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def forward(self, input_ids, **kwargs):
        """
        Forward pass through the model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def shared_step(self, batch, batch_idx, phase="train"):
        """
        Shared step for training/validation/test.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate(self, **kwargs):
        """
        Generate sequences.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _generate_single_batch(
        self,
        input_ids: torch.Tensor,
        label_ids: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Generate sequences for a single batch.

        Args:
            input_ids: Input token IDs for the batch
            label_ids: Class labels for the batch
            **kwargs: Additional model-specific arguments (e.g., cfg_scale)

        Returns:
            Tuple of (samples_list, attentions_list, scores_list) for the batch
        """
        pass

    @torch.no_grad()
    def generate_batched(
        self,
        num_samples_per_class: int = None,
        use_real_prevalence: bool = False,
        batch_size: int = 64,
        include_demographics: bool = True,
        gen_demo=None,
        model_demo_path: str = None,
        checkpoint_dir: str = None,
        gpu_count: int = 1,
        **kwargs,
    ) -> tuple:
        """
        Generate synthetic ICU patient trajectories in batches.

        Args:
            num_samples_per_class: samples per class (control / case).
            use_real_prevalence: if True, weight samples by real class ratio.
            batch_size: generation batch size.
            include_demographics: whether to prepend demographics to each sequence.
            gen_demo: a trained tabular model (or compatible SDV model).
                      If None and include_demographics=True, the LLaMA model
                      generates demographics autoregressively via the logits processor.
            model_demo_path: unused here (model already trained before calling).
            checkpoint_dir: if set, each finished batch is saved there and
                      batches already on disk are loaded instead of regenerated,
                      so a job killed at its time limit can be resumed.
            gpu_count: GPUs to generate on (the first gpu_count visible ones).
                      Above 1, batches are split across one worker process per
                      GPU; the corpus is identical to a single-GPU run. Needs
                      checkpoint_dir, through which workers return batches.
            **kwargs: forwarded to _generate_single_batch.

        Returns:
            (samples_list, label_ids, center_ids, attentions_list, scores_list)
        """
        device = self.device
        if num_samples_per_class is None:
            num_samples_per_class = getattr(self.cfg.sampling, "num_samples", 64)
        if use_real_prevalence:
            num_case, num_control, _ = self.compute_samples_by_prevalence(
                label_name=self._datamodule.return_label,
                total_samples=num_samples_per_class,
            )
        else:
            num_case = num_samples_per_class
            num_control = num_samples_per_class
        total_samples = num_case + num_control

        center_id = getattr(self.cfg.sampling, "center_id", None)
        center_strategy = getattr(self.cfg.sampling, "center_strategy", "null")
        print(center_id)
        print(center_strategy)
        # Demographics: prompted from the tabular model (TVAE) when one is given,
        # otherwise generated by the LLaMA model through the logits processor.
        pre_generated = include_demographics and gen_demo is not None

        # Resume state: prompts, labels and centres are fixed on the first run
        # so batch i always covers the same rows
        run_meta = {
            "total_samples": total_samples,
            "batch_size": batch_size,
            "pretrained_path": getattr(self.cfg.model, "pretrained_path", None),
            "center_strategy": center_strategy,
            "center_id": center_id,
            "cfg_scale": kwargs.get("cfg_scale"),
        }
        resume_state = load_generation_state(checkpoint_dir, run_meta)
        if resume_state is not None:
            prompts = {name: resume_state[name].to(device) for name in PROMPT_NAMES}
        else:
            prompts = self._build_prompts(
                num_control=num_control,
                num_case=num_case,
                center_id=center_id,
                center_strategy=center_strategy,
                gen_demo=gen_demo if pre_generated else None,
            )
            if checkpoint_dir is not None:
                state = {"run_meta": run_meta}
                state.update({name: prompts[name].cpu() for name in PROMPT_NAMES})
                _atomic_dump(state, os.path.join(checkpoint_dir, "meta.joblib"))
        assert_prompts_consistent(prompts, total_samples)
        print("input_ids_all shape:", prompts["input_ids_all"].shape)
        print("input_ids_all[0]:", prompts["input_ids_all"][0])

        batch_count = (total_samples + batch_size - 1) // batch_size  # div_ceil
        base_seed = int(getattr(self.cfg, "seed", 0))
        print(f"[Seed] Reseeding each batch with {base_seed} + batch index", flush=True)
        batch_options = {
            "include_demographics": include_demographics,
            "pre_generated": pre_generated,
            **kwargs,
        }
        if gpu_count > 1:
            if checkpoint_dir is None:
                raise ValueError(
                    "Multi-GPU generation needs checkpoint_dir: workers hand "
                    "their batches back through its files."
                )
            self._generate_batches_in_workers(
                gpu_count=gpu_count,
                batch_count=batch_count,
                batch_size=batch_size,
                prompts=prompts,
                base_seed=base_seed,
                batch_options=batch_options,
                checkpoint_dir=checkpoint_dir,
            )
            # Every batch file now exists, so the loop below only loads them.

        all_samples, all_attentions, all_scores = generate_or_load_batches(
            self,
            batch_count,
            batch_size,
            prompts,
            base_seed,
            batch_options,
            checkpoint_dir,
        )
        assert len(all_samples) == total_samples
        return (
            all_samples,
            prompts["label_ids_all"],
            prompts["center_ids_all"],
            all_attentions,
            all_scores,
        )

    def _build_prompts(
        self, num_control, num_case, center_id, center_strategy, gen_demo
    ):
        """Prompt rows of a new corpus: controls first, then cases.

        input_ids_all is [SOS] per row, or [SOS | female age height weight GAP]
        when gen_demo (a trained tabular model) supplies the demographics.
        """
        device = self.device
        total_samples = num_control + num_case
        sos_token = self._datamodule.tokenizer.get_sos_token_id()
        sos_column = torch.full(
            (total_samples, 1), sos_token, dtype=torch.long, device=device
        )
        if gen_demo is None:
            input_ids_all = sos_column
        else:
            demographic_ids = tabular_demographic_prompt_ids(
                gen_demo, num_control, num_case, self.vocabulary
            ).to(device)
            input_ids_all = torch.cat([sos_column, demographic_ids], dim=1)
        return {
            "input_ids_all": input_ids_all,
            "label_ids_all": torch.tensor(
                [0] * num_control + [1] * num_case, dtype=torch.long, device=device
            ),
            "center_ids_all": center_ids_for_strategy(
                num_control=num_control,
                num_case=num_case,
                center_id=center_id,
                center_strategy=center_strategy,
                num_centers=getattr(self.label_embedder, "num_centers", 6),
                null_center_id=self.label_embedder.null_center_id,
                device=device,
            ),
        }

    def _generate_batches_in_workers(
        self,
        gpu_count,
        batch_count,
        batch_size,
        prompts,
        base_seed,
        batch_options,
        checkpoint_dir,
    ):
        """Run generate_batches_on_devices with a copy of this model fit to pickle.

        Each worker gets a pickled copy of the model. Generation needs only the
        tokenizer of the datamodule, not the dataset it holds, so the datamodule
        is swapped for a tokenizer-only stand-in for the hand-over, and the
        model goes to the CPU so workers do not share GPU 0's memory. Both are
        restored afterwards, also when a worker fails.
        """
        assert gpu_count > 1  # One GPU generates in this process instead.
        if gpu_count > torch.cuda.device_count():
            raise ValueError(
                f"gpu_count={gpu_count}, but only {torch.cuda.device_count()} "
                "GPUs are visible."
            )
        device = self.device
        datamodule = self._datamodule
        self._datamodule = SimpleNamespace(tokenizer=datamodule.tokenizer)
        self.to("cpu")
        try:
            generate_batches_on_devices(
                self,
                device_indices=list(range(gpu_count)),
                batch_count=batch_count,
                batch_size=batch_size,
                prompts={name: tensor.cpu() for name, tensor in prompts.items()},
                base_seed=base_seed,
                batch_options=batch_options,
                checkpoint_dir=checkpoint_dir,
            )
        finally:
            self._datamodule = datamodule
            self.to(device)

    def training_step(self, batch, batch_idx):
        """Training step - delegates to shared_step."""
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx):
        """Validation step - delegates to shared_step."""
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        """Test step - delegates to shared_step."""
        return self.shared_step(batch, batch_idx, phase="test")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = hydra.utils.instantiate(
            self.cfg.train.optimizer, self.model.parameters()
        )
        lr_scheduler = hydra.utils.call(
            self.cfg.train.lr_scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def _process_token_sequence(self, tok_seq):
        """
        Process a token sequence by removing special tokens.

        Args:
            tok_seq: Tensor of token IDs

        Returns:
            List of token strings with special tokens removed
        """
        sos_ids = [v for k, v in self.vocabulary.items() if k.startswith("<SOS")]

        tok_seq = tok_seq[
            (~torch.isin(tok_seq, torch.tensor(sos_ids, device=tok_seq.device)))
            & (tok_seq != self.vocabulary["<PAD>"])
        ]
        # One tolist() instead of an .item() per token, which on a GPU tensor
        # is a host-GPU round trip per token.
        seq = [self.id_to_token[token_id] for token_id in tok_seq.tolist()]
        return seq

    def compute_samples_by_prevalence(self, label_name: str, total_samples: int):
        """
        Compute number of samples per class based on real prevalence.

        Args:
            label_name: Name of the label column (e.g., 'death')
            total_samples: Total number of samples to generate

        Returns:
            Tuple of (num_case, num_control)
        """
        indices = self._datamodule.train_icu_dataset.indices
        y_raw = self._datamodule.train_icu_dataset.dataset.y_full[indices, 1]

        y_collapsed = torch.where(y_raw > 0, torch.tensor(1), torch.tensor(0))
        prevalence = (y_collapsed == 1).float().mean().item()
        requested = self.cfg.sampling.get("positive_fraction")
        if requested is not None:
            if not 0 < requested < 1:
                raise ValueError(
                    "Generation prevalence must lie strictly between zero and one"
                )
            prevalence = float(requested)

        num_case = max(1, int(round(total_samples * prevalence)))
        num_control = max(1, total_samples - num_case)

        return num_case, num_control, prevalence
