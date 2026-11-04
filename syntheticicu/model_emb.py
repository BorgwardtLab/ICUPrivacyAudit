"""
ICU LLaMA model with embedding-based conditioning and Classifier-Free Guidance (CFG).

This model uses learned label embeddings prepended to sequences for class-conditional
generation, with support for CFG during inference.
"""

import torch
import torch.nn as nn
import hydra

from syntheticicu.base_model import (
    ICULlamaBase,
    LlamaWithTimePos,
    get_llama_model,
    create_sinusoidal_embeddings,
    compute_sinusoidal_embeddings,
    preallocated_kv_cache,
)
from syntheticicu.data_classes import (
    DemographicsLogitsProcessor,
    FeatureConstraintLogitsProcessor,
    build_feature_indices,
)

# Re-export for backward compatibility with configs
__all__ = [
    "ICULlamaCFG",
    "ICULlama",  # Alias for backward compatibility
    "LabelEmbedder",
    "get_llama_model",
    "LlamaWithTimePos",
    "create_sinusoidal_embeddings",
    "compute_sinusoidal_embeddings",
]


def _sample_next_token(
    history_ids,
    cond_scores,
    cfg_scale,
    temperature,
    logits_processors,
    uncond_scores=None,
):
    """Apply optional CFG, constrain logits, and sample one token per patient."""
    if uncond_scores is None:
        sampling_scores = cond_scores
    else:
        sampling_scores = uncond_scores + cfg_scale * (cond_scores - uncond_scores)

    for processor in logits_processors:
        sampling_scores = processor(history_ids, sampling_scores)

    # The three checks come back from the GPU in one transfer instead of three,
    # so the host waits for the GPU once per decoding step for them. They still
    # run before sampling and raise in the same order.
    has_nan, has_posinf, has_row_all_neginf = torch.stack(
        [
            torch.isnan(sampling_scores).any(),
            torch.isposinf(sampling_scores).any(),
            torch.isneginf(sampling_scores).all(dim=-1).any(),
        ]
    ).tolist()
    if has_nan:
        raise RuntimeError("Generation produced NaN logits before sampling.")
    if has_posinf:
        raise RuntimeError("Generation produced +inf logits before sampling.")
    if has_row_all_neginf:
        raise RuntimeError(
            "Generation constraints masked every token for at least one sample."
        )

    probabilities = torch.softmax(sampling_scores.float() / temperature, dim=-1)
    next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    return next_tokens, sampling_scores


def _build_initial_time_embedding_mask(input_ids, prefix_len, vocabulary):
    """Mask prompt SOS/EOS tokens while leaving conditioning prefixes unmasked."""
    token_mask = input_ids == vocabulary["<EOS>"]
    for token, token_id in vocabulary.items():
        if token.startswith("<SOS"):
            token_mask |= input_ids == token_id

    prefix_mask = torch.zeros(
        (input_ids.shape[0], prefix_len),
        dtype=torch.bool,
        device=input_ids.device,
    )
    return torch.cat([prefix_mask, token_mask], dim=1)


def prompt_embeddings(
    label_embedder, embed_tokens, input_ids, label_ids, center_ids, cfg_scale
):
    """Conditioning prefix + prompt token embeddings for each CFG branch.

    Returns (cond_embeds, uncond_embeds, prefix_len); uncond_embeds is None
    without CFG. The unconditional branch keeps the centre and drops only the
    outcome label.
    """
    class_embeds = label_embedder(label_ids, train=False, center_ids=center_ids)
    token_embeds = embed_tokens(input_ids)
    cond_embeds = torch.cat([class_embeds, token_embeds], dim=1)
    uncond_embeds = None
    if cfg_scale > 1.0:
        uncond_class_embeds = label_embedder(
            label_ids,
            train=False,
            force_uncond_labels=True,
            force_uncond_centers=False,
            center_ids=center_ids,
        )
        uncond_embeds = torch.cat([uncond_class_embeds, token_embeds], dim=1)
    assert cond_embeds.shape[0] == input_ids.shape[0]
    return cond_embeds, uncond_embeds, class_embeds.shape[1]


def logits_processors_for_batch(
    model, prompt_length, include_demographics, pre_generated
):
    """The constraints applied to every sampling step of one batch, in order."""
    logits_processors = []
    tokenizer = model._datamodule.tokenizer
    if include_demographics and hasattr(tokenizer, "demographic_token_ids"):
        demo_token_ids = tokenizer.get_demographic_token_ids()
        if demo_token_ids:
            logits_processors.append(
                DemographicsLogitsProcessor(
                    demographic_token_ids=demo_token_ids,
                    gap_token_id=model.vocabulary["<GAP>"],
                    vocab_size=len(model.vocabulary),
                    start_pos=prompt_length,
                    pre_generated=pre_generated,
                )
            )
    if getattr(model.cfg.sampling, "enforce_single_feature_per_timepoint", True):
        # Built once per model: it only depends on the vocabulary.
        if not hasattr(model, "_feature_indices"):
            model._feature_indices, model._num_features = build_feature_indices(
                model.vocabulary
            )
        logits_processors.append(
            FeatureConstraintLogitsProcessor(
                feature_indices=model._feature_indices,
                gap_token_id=model.vocabulary["<GAP>"],
                num_features=model._num_features,
            )
        )
    return logits_processors


def initial_inputs_embeds(
    cond_inputs_embeds,
    uncond_inputs_embeds,
    time_positions,
    time_embedding_mask,
    use_cfg,
    time_embedding_dim,
):
    """Prompt embeddings fed at the first decoding step.

    With CFG the rows are [unconditional; conditional]. time_embedding_dim is
    None for a model without time embeddings; otherwise sinusoidal time
    embeddings are added, zeroed where time_embedding_mask is set (prompt
    SOS/EOS), as in training.
    """
    assert cond_inputs_embeds.shape[:2] == time_positions.shape
    assert time_positions.shape == time_embedding_mask.shape
    if use_cfg:
        inputs_embeds = torch.cat([uncond_inputs_embeds, cond_inputs_embeds], dim=0)
    else:
        inputs_embeds = cond_inputs_embeds
    if time_embedding_dim is None:
        return inputs_embeds
    time_embeddings = compute_sinusoidal_embeddings(
        duplicate_for_cfg(time_positions, use_cfg),
        time_embedding_dim,
        dtype=inputs_embeds.dtype,
    )
    time_embeddings = time_embeddings.masked_fill(
        duplicate_for_cfg(time_embedding_mask, use_cfg).unsqueeze(-1), 0.0
    )
    return inputs_embeds + time_embeddings


def duplicate_for_cfg(tensor, use_cfg):
    """[tensor; tensor] along the batch for the two CFG branches, else tensor."""
    if use_cfg:
        return torch.cat([tensor, tensor], dim=0)
    return tensor


def split_cfg_scores(last_logits, use_cfg):
    """(unconditional, conditional) next-token logits; (None, logits) without CFG."""
    if use_cfg:
        uncond_scores, cond_scores = last_logits.chunk(2)
        return uncond_scores, cond_scores
    return None, last_logits


def conditional_attentions(attentions, batch_size, use_cfg):
    """Per-layer attentions of the conditional rows (the second half with CFG)."""
    return tuple(layer[batch_size:] if use_cfg else layer for layer in attentions)


def attentions_per_sample(generated_attentions, batch_size):
    """Regroup per-step, per-layer attentions into one list per sample."""
    if not generated_attentions:
        return []
    return [
        [
            tuple(layer[sample_idx] for layer in step_attentions)
            for step_attentions in generated_attentions
        ]
        for sample_idx in range(batch_size)
    ]


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vectors for conditioning.

    Last index (num_classes) is reserved for the 'null' (unconditional) embedding,
    which is used for Classifier-Free Guidance.

    Args:
        num_classes: Number of classes (excluding null)
        hidden_size: Embedding dimension
        dropout_prob: Probability to replace label with null during training (CFG dropout)
        token_count: Number of prefix tokens to produce
        num_centers: Number of centers (excluding null)
        center_dropout_prob: Probability to replace center with null during training
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float = 0.1,
        token_count: int = 1,
        num_centers: int = 2,
        center_dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.uncond_id = int(num_classes)
        self.token_count = int(token_count)
        self.dropout_prob = float(dropout_prob)
        self.embedding = nn.Embedding(self.num_classes + 1, hidden_size)

        self.num_centers = int(num_centers)
        self.center_embedding = nn.Embedding(self.num_centers + 1, hidden_size)
        self.null_center_id = int(num_centers)
        self.center_dropout_prob = float(center_dropout_prob)

    def forward(
        self,
        labels: torch.LongTensor,
        train: bool = True,
        force_uncond: bool = False,
        force_uncond_labels: bool = False,
        force_uncond_centers: bool = False,
        center_ids: torch.LongTensor = None,
    ):
        device = labels.device
        labels = labels.clone().long()

        # label. Dropout uses torch.where instead of `if mask.any(): labels[mask]`:
        # same result and RNG draws, without making the host wait for the GPU.
        if force_uncond or force_uncond_labels:
            labels[:] = self.uncond_id
        elif train and self.dropout_prob > 0.0:
            mask = torch.rand(labels.shape[0], device=device) < self.dropout_prob
            labels = torch.where(mask, self.uncond_id, labels)

        label_emb = self.embedding(labels).unsqueeze(1)  # (B, 1, H)

        # center
        if center_ids is None:
            center_ids = torch.full_like(labels, self.null_center_id)
        else:
            center_ids = center_ids.clone().long()

        if force_uncond_centers:
            center_ids[:] = self.null_center_id
        elif train and self.center_dropout_prob > 0.0:
            center_mask = (
                torch.rand(center_ids.shape[0], device=device)
                < self.center_dropout_prob
            )
            center_ids = torch.where(center_mask, self.null_center_id, center_ids)

        center_emb = self.center_embedding(center_ids).unsqueeze(1)  # (B, 1, H)

        # concat label and center embeddings, and expand if token_count > 1
        if self.token_count > 1:
            label_emb = label_emb.expand(-1, self.token_count, -1)
            center_emb = center_emb.expand(-1, self.token_count, -1)

        return torch.cat([label_emb, center_emb], dim=1)  # (B, 2*token_count, H)


class ICULlamaCFG(ICULlamaBase):
    """
    ICU LLaMA with embedding-based conditioning and Classifier-Free Guidance.

    Uses learned label embeddings prepended to the sequence for conditioning,
    with support for CFG during generation (cfg_scale parameter).
    """

    def instantiate_model(self, cfg):
        """Instantiate the model with label embedder for CFG."""
        config = hydra.utils.call(cfg.model, vocabulary=self.vocabulary).config
        self._setup_time_embeddings(config)

        self.prefix_token_count = config.prefix_token_count
        hidden_size = config.hidden_size

        # Instantiate the label embedder
        self.label_embedder = LabelEmbedder(
            num_classes=config.num_classes,
            hidden_size=hidden_size,
            dropout_prob=config.label_dropout,
            token_count=self.prefix_token_count,
            num_centers=getattr(config, "num_centers", 2),
            center_dropout_prob=getattr(config, "center_dropout_prob", 0.3),
        )

        # Instantiate the LlamaWithTimePos model
        self.model = LlamaWithTimePos(
            config=config,
            use_time_embeddings=self.use_time_embeddings,
            gap_token_id=self.vocabulary["<GAP>"],
        )

    def forward(
        self,
        input_ids,
        time_positions=None,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        class_labels=None,
        center_ids=None,
    ):
        """
        Forward pass with class label conditioning.

        Args:
            input_ids: Input token IDs
            time_positions: Time position indices
            attention_mask: Attention mask
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            class_labels: Class labels for conditioning
            center_ids: Optional center IDs for additional conditioning

        Returns:
            Logits tensor or full model output
        """
        embeddings = self.model.model.embed_tokens(input_ids)

        self.actual_prefix_len = 0  # Default to 0 if no class labels
        if class_labels is not None:
            # Prepend class embeddings
            class_embed = self.label_embedder(
                class_labels, train=self.training, center_ids=center_ids
            )
            embeddings = torch.cat([class_embed, embeddings], dim=1)

            # calculate prefix length
            self.actual_prefix_len = class_embed.shape[1]

            # Update attention mask for prefix
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    (attention_mask.size(0), self.actual_prefix_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Update time_positions for prefix
            if time_positions is not None:
                prefix_tp = torch.zeros(
                    (time_positions.size(0), self.actual_prefix_len),
                    dtype=time_positions.dtype,
                    device=time_positions.device,
                )
                time_positions = torch.cat([prefix_tp, time_positions], dim=1)

        if self.use_time_embeddings and time_positions is not None:
            time_embeds = compute_sinusoidal_embeddings(
                time_positions,
                self.time_embedding_dim,
                dtype=embeddings.dtype,
            )

            # Mask SOS and EOS tokens
            sos_tokens = [
                self.vocabulary[tok]
                for tok in self.vocabulary.keys()
                if tok.startswith("<SOS")
            ]

            # Add pad to the prefix for masking
            input_ids_with_prefix = torch.cat(
                [
                    torch.full(
                        (input_ids.size(0), self.actual_prefix_len),
                        self.vocabulary["<PAD>"],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    ),
                    input_ids,
                ],
                dim=1,
            )

            sos_masks = [(input_ids_with_prefix == sos_tok) for sos_tok in sos_tokens]
            mask = torch.any(torch.stack(sos_masks), dim=0) | (
                input_ids_with_prefix == self.vocabulary["<EOS>"]
            )
            time_embeds = time_embeds.masked_fill(mask.unsqueeze(-1), 0.0)

            embeddings += time_embeds

        return self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

    def shared_step(self, batch, batch_idx, phase="train"):
        """
        Shared step with prefix-aware loss computation.

        Handles both formats:
        - With time positions: (X, time_positions, attention_mask, S, IDs)
        - Without: (X, attention_mask, S, IDs)
        """
        if len(batch) == 5:
            X, time_positions, attention_mask, S, IDs = batch
        else:
            X, attention_mask, S, IDs = batch
            time_positions = None

        center_ids = IDs[:, -1] if IDs.size(1) > 3 else None

        if self.use_time_embeddings and time_positions is not None:
            logits = self.forward(
                input_ids=X,
                class_labels=IDs[:, 1],
                center_ids=center_ids,
                time_positions=time_positions,
                attention_mask=attention_mask,
            )
        else:
            logits = self.forward(
                input_ids=X,
                class_labels=IDs[:, 1],
                center_ids=center_ids,
                attention_mask=attention_mask,
            )

        # Dimensions for loss calculation
        T = X.size(1)
        V = logits.size(-1)
        prefix = self.actual_prefix_len

        # Trim logits to match input length after prefix
        logits = logits[:, : T + prefix, :]

        # Shift logits and labels for causal LM (skip prefix in logits)
        shift_logits = logits[:, prefix:-1, :]  # Align with X[1:]
        shift_labels = X[:, 1:]  # Real targets

        # Compute loss ignoring padding
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.vocabulary["<PAD>"])
        loss = loss_fct(shift_logits.reshape(-1, V), shift_labels.reshape(-1))

        on_step = phase == "train"

        self.log(
            f"{phase}_loss",
            loss,
            on_step=on_step,
            on_epoch=not on_step,
            sync_dist=True,
        )

        return loss

    def _generate_single_batch(
        self,
        input_ids: torch.Tensor,
        label_ids: torch.Tensor,
        cfg_scale: float = 1.0,
        **kwargs,
    ) -> tuple:
        """
        Generate sequences for a single batch with optional CFG.

        Args:
            input_ids: (B, seq_len) — SOS only, or SOS+demo tokens if pre_generated.
            label_ids: (B,) class labels.
            cfg_scale: CFG guidance scale (1.0 = disabled).
            **kwargs:
                include_demographics (bool): whether demographics are part of the sequence.
                pre_generated (bool): True when demo tokens are already in input_ids.

        Returns:
            (samples_list, attentions_list, scores_list)
            samples_list contains full sequences including demo tokens (if any),
            so they are visible for downstream analysis / visualisation.
        """
        return self._generate_single_batch_manual(
            input_ids=input_ids,
            label_ids=label_ids,
            cfg_scale=cfg_scale,
            include_demographics=kwargs.get("include_demographics", True),
            pre_generated=kwargs.get("pre_generated", False),
            center_ids=kwargs.get("center_ids"),
        )

    def _generate_single_batch_manual(
        self,
        input_ids,
        label_ids,
        cfg_scale,
        include_demographics,
        pre_generated,
        center_ids,
    ):
        """Generate with explicit time tracking and optional CFG."""
        if cfg_scale < 1.0:
            raise ValueError(f"cfg_scale must be at least 1, got {cfg_scale}")
        if self.cfg.sampling.num_beams != 1:
            raise ValueError("Manual generation only supports num_beams=1.")

        device = input_ids.device
        batch_size = input_ids.shape[0]
        output_attentions = self.cfg.sampling.output_attentions
        output_scores = self.cfg.sampling.output_scores
        temperature = getattr(self.cfg.sampling, "temperature", 1.0)
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if pre_generated:
            expected_prompt_len = 1 + 4 + 1
            if input_ids.shape[1] != expected_prompt_len:
                raise ValueError(
                    "TVAE prompts must contain "
                    "[SOS | female | age | height | weight | GAP]."
                )

        cond_embeds, uncond_embeds, prefix_len = prompt_embeddings(
            label_embedder=self.label_embedder,
            embed_tokens=self.model.model.embed_tokens,
            input_ids=input_ids,
            label_ids=label_ids,
            center_ids=center_ids,
            cfg_scale=cfg_scale,
        )
        time_positions = torch.zeros(
            (batch_size, cond_embeds.shape[1]),
            dtype=torch.long,
            device=device,
        )
        time_embedding_mask = _build_initial_time_embedding_mask(
            input_ids,
            prefix_len=prefix_len,
            vocabulary=self.vocabulary,
        )
        logits_processors = logits_processors_for_batch(
            self,
            prompt_length=input_ids.shape[1],
            include_demographics=include_demographics,
            pre_generated=pre_generated,
        )

        (
            history_ids,
            generated_attentions,
            generated_scores,
        ) = self._generate_tokens(
            history_ids=input_ids,
            uncond_inputs_embeds=uncond_embeds,
            cond_inputs_embeds=cond_embeds,
            initial_time_positions=time_positions,
            initial_time_embedding_mask=time_embedding_mask,
            next_time_positions=torch.full(
                (batch_size,),
                0 if include_demographics and not pre_generated else 1,
                dtype=torch.long,
                device=device,
            ),
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_new_tokens=self.cfg.sampling.max_new_tokens,
            logits_processors=logits_processors,
            output_attentions=output_attentions,
            output_scores=output_scores,
        )

        # One device-to-host copy for the whole batch: decoding rows that are
        # still on the GPU costs a host-GPU round trip per token.
        history_ids_cpu = history_ids.cpu()
        samples_list = [
            self._process_token_sequence(history_ids_cpu[i]) for i in range(batch_size)
        ]
        assert len(samples_list) == batch_size
        attentions_list = attentions_per_sample(generated_attentions, batch_size)
        return samples_list, attentions_list, generated_scores

    def _generate_tokens(
        self,
        history_ids,
        cond_inputs_embeds,
        uncond_inputs_embeds,
        initial_time_positions,
        initial_time_embedding_mask,
        next_time_positions,
        cfg_scale,
        temperature,
        max_new_tokens,
        logits_processors,
        output_attentions=False,
        output_scores=False,
    ):
        """Autoregress with explicit time positions and optional CFG.

        With CFG the model batch is [unconditional rows; conditional rows], and
        both halves are fed the same tokens and time positions.
        """
        if cfg_scale < 1.0:
            raise ValueError(f"cfg_scale must be at least 1, got {cfg_scale}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if self.cfg.sampling.num_beams != 1:
            raise ValueError("Manual generation only supports num_beams=1.")

        use_cfg = cfg_scale > 1.0
        if use_cfg and uncond_inputs_embeds is None:
            raise ValueError("CFG generation requires unconditional embeddings.")
        if not use_cfg and uncond_inputs_embeds is not None:
            raise ValueError("Unconditional embeddings require cfg_scale > 1.")

        batch_size = history_ids.shape[0]
        device = history_ids.device
        model_inputs_embeds = initial_inputs_embeds(
            cond_inputs_embeds=cond_inputs_embeds,
            uncond_inputs_embeds=uncond_inputs_embeds,
            time_positions=initial_time_positions,
            time_embedding_mask=initial_time_embedding_mask,
            use_cfg=use_cfg,
            time_embedding_dim=(
                self.time_embedding_dim if self.use_time_embeddings else None
            ),
        )
        assert model_inputs_embeds.shape[0] == (2 if use_cfg else 1) * batch_size
        attention_mask = torch.ones(
            model_inputs_embeds.shape[:2], dtype=torch.long, device=device
        )

        current_time = next_time_positions
        history_ids = history_ids.clone()
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        pad_token_id = self.vocabulary["<PAD>"]
        eos_token_id = self.vocabulary["<EOS>"]
        gap_token_id = self.vocabulary["<GAP>"]
        generated_scores = []
        generated_attentions = []
        # Room for the prompt (with its conditioning prefix) and every new token:
        # the cache never grows by copying
        past_key_values = preallocated_kv_cache(
            layer_count=self.model.config.num_hidden_layers,
            capacity=model_inputs_embeds.shape[1] + max_new_tokens,
        )
        next_input_ids = None
        token_time_positions = None

        for step_index in range(max_new_tokens):
            # The first step feeds the whole prompt (with its time embeddings
            # already added); later steps feed the last token and its time.
            if step_index == 0:
                step_inputs = {
                    "inputs_embeds": model_inputs_embeds,
                    "time_positions": None,
                }
            else:
                step_inputs = {
                    "input_ids": next_input_ids,
                    "time_positions": token_time_positions,
                }
            outputs = self.model(
                **step_inputs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=output_attentions,
            )
            assert outputs.past_key_values is past_key_values

            uncond_scores, cond_scores = split_cfg_scores(
                outputs.logits[:, -1, :], use_cfg
            )
            next_tokens, sampling_scores = _sample_next_token(
                history_ids=history_ids,
                cond_scores=cond_scores,
                uncond_scores=uncond_scores,
                cfg_scale=cfg_scale,
                temperature=temperature,
                logits_processors=logits_processors,
            )
            next_tokens = torch.where(
                unfinished,
                next_tokens,
                torch.full_like(next_tokens, pad_token_id),
            )
            history_ids = torch.cat([history_ids, next_tokens.unsqueeze(1)], dim=1)

            if output_scores:
                generated_scores.append(sampling_scores)
            if output_attentions and outputs.attentions is not None:
                generated_attentions.append(
                    conditional_attentions(outputs.attentions, batch_size, use_cfg)
                )

            token_time_positions = duplicate_for_cfg(current_time, use_cfg).unsqueeze(1)
            current_time = torch.where(
                next_tokens == gap_token_id,
                current_time + 1,
                current_time,
            )
            unfinished &= next_tokens != eos_token_id
            if not unfinished.any():
                break

            next_input_ids = duplicate_for_cfg(next_tokens, use_cfg).unsqueeze(1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=1,
            )

        return history_ids, generated_attentions, generated_scores

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        num_samples_per_class=None,
        use_real_prevalence=False,
        cfg_scale=1.0,
        label_ids=None,
    ):
        """
        Generate ICU stay sequences with CFG support.

        Args:
            input_ids: Optional pre-defined input IDs
            num_samples_per_class: Number of samples per class
            use_real_prevalence: If True, use real class prevalence
            cfg_scale: Classifier-Free Guidance scale (1.0 = no guidance)
            label_ids: Labels for pre-defined input IDs

        Returns:
            Tuple of (samples_list, label_ids, attentions_list, scores_list)
        """
        device = self.device

        if input_ids is None:
            if num_samples_per_class is None:
                num_samples_per_class = self.cfg.sampling.num_samples
            sos_token = self._datamodule.tokenizer.get_sos_token_id()

            if use_real_prevalence:
                num_case, num_control = self.compute_samples_by_prevalence(
                    label_name=self._datamodule.return_label,
                    total_samples=num_samples_per_class,
                )
            else:
                num_case = num_samples_per_class
                num_control = num_samples_per_class

            # Base input ids: SOS for all samples
            input_ids = (
                torch.tensor([sos_token] * (num_control + num_case), dtype=torch.long)
                .unsqueeze(1)
                .to(device)
            )
            label_ids = torch.tensor(
                [0] * num_control + [1] * num_case, dtype=torch.long
            ).to(device)
        elif label_ids is None:
            raise ValueError("label_ids must be provided when input_ids is provided.")

        samples, attentions, scores = self._generate_single_batch(
            input_ids=input_ids,
            label_ids=label_ids,
            cfg_scale=cfg_scale,
            include_demographics=getattr(
                self.cfg.sampling, "include_demographics", True
            ),
            pre_generated=False,
            center_ids=None,
        )
        return samples, label_ids, attentions, scores


# Alias for backward compatibility with existing configs
ICULlama = ICULlamaCFG
