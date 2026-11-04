"""Opt-in bin-plus-residual model core, independent of the binned workflows.

Numerical events retain their feature/bin token and carry a residual in [0, 1].
The next bin is categorical; its residual follows a conditional Beta density.
This module does not load data or register itself with existing entry points.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaModel


class NumericalBins(nn.Module):
    """Convert aligned values to residuals using explicit, finite bin bounds.

    Args:
        lower: Lower bound for each vocabulary token.
        upper: Upper bound for each vocabulary token.
        numerical: Whether each token carries a continuous measurement.

    Fit bounds using training patients only. Non-numerical tokens, including
    demographics and structural tokens, have no numerical value.
    """

    def __init__(self, lower, upper, numerical):
        super().__init__()
        lower = torch.as_tensor(lower, dtype=torch.float32)
        upper = torch.as_tensor(upper, dtype=torch.float32)
        numerical = torch.as_tensor(numerical, dtype=torch.bool)
        if (
            lower.ndim != 1
            or lower.shape != upper.shape
            or lower.shape != numerical.shape
        ):
            raise ValueError(
                "Bin bounds and numerical mask must be equal-length vectors"
            )
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("Bin bounds must be finite")
        if (upper[numerical] <= lower[numerical]).any():
            raise ValueError("Numerical bins must have positive width")
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.register_buffer("numerical", numerical)

    @classmethod
    def from_vocabulary(cls, vocabulary, feature_edges):
        """Map explicit feature edges onto existing feature/bin token IDs.

        Args:
            vocabulary: Contiguous token name to integer ID mapping.
            feature_edges: Numerical feature name to its ordered bin edges.
                Include both finite endpoints. All bins must have positive width.
        """
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("Vocabulary IDs must be contiguous from zero")
        lower, upper = torch.zeros(len(vocabulary)), torch.zeros(len(vocabulary))
        numerical = torch.zeros(len(vocabulary), dtype=torch.bool)
        for feature, edges in feature_edges.items():
            edges = torch.as_tensor(edges, dtype=torch.float32)
            if edges.ndim != 1 or len(edges) < 2:
                raise ValueError(f"{feature} needs at least two bin edges")
            for index in range(len(edges) - 1):
                token_id = vocabulary[f"{feature}_bin_{index}"]
                lower[token_id], upper[token_id] = edges[index], edges[index + 1]
                numerical[token_id] = True
            expected = {f"{feature}_bin_{index}" for index in range(len(edges) - 1)}
            actual = {name for name in vocabulary if name.startswith(f"{feature}_bin_")}
            if actual != expected:
                raise ValueError(f"Bin edges do not match vocabulary for {feature}")
        return cls(lower, upper, numerical)

    def validate_residuals(self, input_ids, residuals):
        """Reject missing, out-of-range, or misaligned numerical inputs."""
        self._validate_ids(input_ids)
        if input_ids.shape != residuals.shape:
            raise ValueError("Token IDs and residuals must have identical shapes")
        if not torch.isfinite(residuals).all():
            raise ValueError(
                "Residuals must be finite; use zero for non-numerical tokens"
            )
        numeric = self.numerical[input_ids]
        if ((residuals[numeric] < 0) | (residuals[numeric] > 1)).any():
            raise ValueError("Numerical residuals must lie in [0, 1]")
        if (residuals[~numeric] != 0).any():
            raise ValueError("Non-numerical residuals must be zero")

    def _validate_ids(self, input_ids):
        if input_ids.dtype != torch.long:
            raise ValueError("Token IDs must have dtype torch.long")
        if ((input_ids < 0) | (input_ids >= len(self.lower))).any():
            raise ValueError("Token IDs outside the vocabulary")

    def encode(self, input_ids, values):
        """Encode raw values aligned with token IDs, rejecting bin mismatches."""
        self._validate_ids(input_ids)
        if input_ids.shape != values.shape:
            raise ValueError(
                "Token IDs and numerical values must have identical shapes"
            )
        numeric = self.numerical[input_ids]
        observed = values[numeric].float()
        low, high = self.lower[input_ids][numeric], self.upper[input_ids][numeric]
        if not torch.isfinite(observed).all():
            raise ValueError("Numerical measurements must be finite")
        if ((observed < low) | (observed > high)).any():
            raise ValueError("Numerical measurements lie outside their token's bin")
        residuals = torch.zeros_like(values, dtype=torch.float32)
        residuals[numeric] = (observed - low) / (high - low)
        return residuals

    def decode(self, input_ids, residuals):
        """Decode values in original units, returning NaN for discrete tokens."""
        self.validate_residuals(input_ids, residuals)
        numeric = self.numerical[input_ids]
        values = self.lower[input_ids] + residuals * (
            self.upper[input_ids] - self.lower[input_ids]
        )
        return values.masked_fill(~numeric, float("nan"))


@dataclass
class NumericalOutput:
    """Predictions and optional teacher-forced loss terms."""

    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: torch.Tensor | None = None
    token_loss: torch.Tensor | None = None
    numerical_loss: torch.Tensor | None = None


class BinResidualLlama(nn.Module):
    """Generate bin tokens and continuous residuals with a causal LLaMA.

    Args:
        config: Transformers LlamaConfig with explicit PAD, BOS, and EOS IDs.
        bins: NumericalBins covering the complete vocabulary.
        num_centers: Number of center labels.
        num_classes: Number of outcome labels.
        numerical_loss_weight: Weight of mean numerical NLL relative to token CE.
        outcome_dropout: Probability of using the null outcome prefix in training.
        center_dropout: Probability of using the null center prefix in training.
        gap_token_id: Optional hourly separator for time-aware generation.

    Beta likelihoods are evaluated in float32. Endpoint residuals are clipped
    to [1e-6, 1-1e-6] for density evaluation only. Numerical NLL is a density
    in residual coordinates, not a probability mass or a calibrated privacy score.
    """

    def __init__(
        self,
        config,
        bins,
        num_centers,
        num_classes=2,
        numerical_loss_weight=1.0,
        outcome_dropout=0.1,
        center_dropout=0.3,
        gap_token_id=None,
    ):
        super().__init__()
        if config.vocab_size != len(bins.lower):
            raise ValueError("Bin metadata must cover the entire vocabulary")
        special_ids = [config.pad_token_id, config.bos_token_id, config.eos_token_id]
        if any(index is None for index in special_ids) or len(set(special_ids)) != 3:
            raise ValueError("Distinct PAD, BOS, and EOS IDs are required")
        if bins.numerical[special_ids].any():
            raise ValueError("Structural tokens cannot carry numerical values")
        if gap_token_id is not None and bins.numerical[gap_token_id]:
            raise ValueError("GAP cannot carry a numerical value")
        if not 0 <= outcome_dropout <= 1 or not 0 <= center_dropout <= 1:
            raise ValueError("Conditioning dropout must lie in [0, 1]")
        if numerical_loss_weight < 0:
            raise ValueError("numerical_loss_weight must be nonnegative")
        if num_classes < 1 or num_centers < 1:
            raise ValueError("Positive outcome and center counts are required")
        self.config, self.bins = config, bins
        self.num_classes, self.num_centers = num_classes, num_centers
        self.outcome_dropout, self.center_dropout = outcome_dropout, center_dropout
        self.numerical_loss_weight = numerical_loss_weight
        self.gap_token_id = gap_token_id
        hidden = config.hidden_size
        self.backbone = LlamaModel(config)
        self.lm_head = nn.Linear(hidden, config.vocab_size, bias=False)
        self.value_embedding = nn.Linear(1, hidden, bias=False)
        self.outcome_embedding = nn.Embedding(num_classes + 1, hidden)
        self.center_embedding = nn.Embedding(num_centers + 1, hidden)
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )

    def _condition(self, labels, size, dropout, embedding):
        if labels.dtype != torch.long:
            raise ValueError("Conditioning labels must have dtype torch.long")
        if ((labels < 0) | (labels > size)).any():
            raise ValueError("Conditioning label outside the supported range")
        labels = labels.long()
        if self.training and dropout:
            labels = labels.masked_fill(torch.rand_like(labels.float()) < dropout, size)
        return embedding(labels)

    def value_distribution(self, hidden_states, next_ids):
        """Predict a Beta residual distribution given history and the next bin."""
        target_embedding = self.backbone.embed_tokens(next_ids)
        parameters = self.value_head(
            torch.cat([hidden_states, target_embedding], dim=-1)
        )
        alpha, beta = (F.softplus(parameters.float()) + 0.01).unbind(-1)
        return torch.distributions.Beta(alpha, beta)

    def forward(
        self,
        input_ids,
        residuals,
        class_labels,
        center_ids,
        time_positions=None,
        compute_loss=False,
    ):
        """Predict next events; losses use targets strictly after each context."""
        self.bins.validate_residuals(input_ids, residuals)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Expected nonempty [batch, sequence] token IDs")
        if (
            class_labels.shape != input_ids.shape[:1]
            or center_ids.shape != class_labels.shape
        ):
            raise ValueError("Provide one outcome and center label per sequence")
        mask = input_ids != self.config.pad_token_id
        embeddings = self.backbone.embed_tokens(input_ids)
        numeric = self.bins.numerical[input_ids]
        value_input = (residuals * numeric).to(embeddings.dtype).unsqueeze(-1)
        embeddings = embeddings + self.value_embedding(value_input)
        if time_positions is not None:
            if time_positions.shape != input_ids.shape:
                raise ValueError("Time positions must align with token IDs")
            frequency = torch.exp(
                torch.arange(
                    0, self.config.hidden_size, 2, device=input_ids.device
                ).float()
                * (-math.log(10000.0) / self.config.hidden_size)
            )
            angles = time_positions.float().unsqueeze(-1) * frequency
            time_embedding = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(
                -2
            )
            embeddings = embeddings + time_embedding[..., : self.config.hidden_size].to(
                embeddings.dtype
            )
        prefix = torch.stack(
            [
                self._condition(
                    class_labels,
                    self.num_classes,
                    self.outcome_dropout,
                    self.outcome_embedding,
                ),
                self._condition(
                    center_ids,
                    self.num_centers,
                    self.center_dropout,
                    self.center_embedding,
                ),
            ],
            dim=1,
        )
        prefix_mask = torch.ones(
            input_ids.shape[0], 2, device=input_ids.device, dtype=torch.bool
        )
        hidden = self.backbone(
            inputs_embeds=torch.cat([prefix, embeddings], dim=1),
            attention_mask=torch.cat([prefix_mask, mask], dim=1),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, 2:]
        output = NumericalOutput(self.lm_head(hidden), hidden)
        if compute_loss:
            targets = input_ids[:, 1:]
            valid = mask[:, 1:]
            if not valid.any():
                raise ValueError(
                    "Training requires at least one non-padding next token"
                )
            output.token_loss = F.cross_entropy(
                output.logits[:, :-1][valid].float(), targets[valid]
            )
            numeric_targets = self.bins.numerical[targets] & valid
            if numeric_targets.any():
                distribution = self.value_distribution(
                    hidden[:, :-1][numeric_targets], targets[numeric_targets]
                )
                output.numerical_loss = -distribution.log_prob(
                    residuals[:, 1:][numeric_targets].float().clamp(1e-6, 1 - 1e-6)
                ).mean()
            else:
                output.numerical_loss = hidden.sum() * 0
            output.loss = (
                output.token_loss + self.numerical_loss_weight * output.numerical_loss
            )
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        residuals,
        class_labels,
        center_ids,
        max_new_tokens,
        time_positions=None,
        temperature=1.0,
        logits_processors=(),
    ):
        """Sample events, feeding each sampled residual back into the context.

        Use eval mode. Prompts must be unpadded. This simple reference sampler
        recomputes the full context and supports conditioning without CFG scaling.
        Existing feature constraints can be supplied as logits processors.
        """
        if self.training:
            raise ValueError("Call eval() before generation")
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError(
                "Nonnegative token budget and positive temperature required"
            )
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Generation requires nonempty [batch, sequence] prompts")
        if (input_ids == self.config.pad_token_id).any():
            raise ValueError("Generation prompts must be unpadded")
        self.bins.validate_residuals(input_ids, residuals)
        if time_positions is not None and self.gap_token_id is None:
            raise ValueError("Time-aware generation requires gap_token_id")
        finished = input_ids[:, -1] == self.config.eos_token_id
        for _ in range(max_new_tokens):
            if finished.all():
                break
            output = self(
                input_ids, residuals, class_labels, center_ids, time_positions
            )
            scores = output.logits[:, -1].float() / temperature
            for processor in logits_processors:
                scores = processor(input_ids, scores)
            scores[:, self.config.pad_token_id] = -torch.inf
            scores[:, self.config.bos_token_id] = -torch.inf
            if not torch.isfinite(scores[~finished]).any(dim=-1).all():
                raise ValueError("No valid next token after applying constraints")
            scores[finished] = 0
            next_ids = torch.multinomial(scores.softmax(-1), 1).squeeze(-1)
            next_ids = next_ids.masked_fill(finished, self.config.pad_token_id)
            next_residuals = self.value_distribution(
                output.hidden_states[:, -1], next_ids
            ).sample()
            next_residuals = next_residuals.masked_fill(
                ~self.bins.numerical[next_ids], 0
            )
            if time_positions is not None:
                next_time = time_positions[:, -1] + (
                    input_ids[:, -1] == self.gap_token_id
                )
                time_positions = torch.cat(
                    [time_positions, next_time.unsqueeze(-1)], dim=-1
                )
            input_ids = torch.cat([input_ids, next_ids.unsqueeze(-1)], dim=-1)
            residuals = torch.cat([residuals, next_residuals.unsqueeze(-1)], dim=-1)
            finished = finished | (next_ids == self.config.eos_token_id)
        result = {
            "input_ids": input_ids,
            "residuals": residuals,
            "values": self.bins.decode(input_ids, residuals),
        }
        if time_positions is not None:
            result["time_positions"] = time_positions
        return result
