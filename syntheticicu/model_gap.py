"""
ICU LLaMA model with gap prediction multi-task learning.

This model extends ICULlamaCFG with an auxiliary task that predicts which features
will appear in the next time gap. Uses multi-hot binary classification at each GAP token.
"""

import torch
import torch.nn as nn
import hydra
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from syntheticicu.base_model import (
    ICULlamaBase,
    LlamaWithTimePos,
    get_llama_model,
    create_sinusoidal_embeddings,
)
from syntheticicu.model_emb import ICULlamaCFG, LabelEmbedder
from syntheticicu.data_classes import CFGLogitsProcessor

# Re-export for backward compatibility
__all__ = [
    "ICULlamaGap",
    "LabelEmbedder",
    "get_llama_model",
    "LlamaWithTimePos",
    "create_sinusoidal_embeddings",
    "build_next_gap_targets",
]


def build_next_gap_targets(
    input_ids: torch.Tensor,
    gap_token_id: int,
    feature_indices: torch.Tensor,
    num_features: int,
    bins_per_feature: int,
    skip_demo_gap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build multi-hot targets for gap prediction task.

    For each GAP token at position t, compute a binary target indicating which
    features appear in the tokens between this GAP and the next GAP (or EOS).
    Terminal GAPs (preceding EOS or PAD) and optionally the first GAP
    (demographics separator) are excluded from the loss mask.

    Args:
        input_ids: (B, T) tensor of token IDs
        gap_token_id: Token ID for <GAP>
        eos_token_id: Token ID for <EOS>
        pad_token_id: Token ID for <PAD>
        feature_indices: (V,) tensor mapping token ID -> feature index (-1 for non-features)
        num_features: Number of distinct features
        bins_per_feature: Number of bins per feature
        skip_demo_gap: If True, exclude the first GAP (demographics separator) from loss

    Returns:
        gap_mask: (B, T) bool — GAPs that contribute to loss
        targets: (B, T, num_features) binary targets
    """
    B, T = input_ids.shape
    device = input_ids.device

    feat_ids = feature_indices[input_ids]  # (B, T)
    full_gap_mask = input_ids == gap_token_id  # (B, T)
    gap_mask = full_gap_mask.clone()

    # Exclude first GAP (demographics separator) if requested
    # if skip_demo_gap:
    #     first_gap_pos = full_gap_mask.long().argmax(dim=1)  # (B,)
    #     gap_mask[torch.arange(B, device=device), first_gap_pos] = False

    # Build targets
    gap_cumsum = full_gap_mask.long().cumsum(dim=1)
    is_feature = feat_ids >= 0
    valid_feature_mask = is_feature & (gap_cumsum > 0)

    targets = torch.zeros(B, T, num_features, device=device, dtype=torch.float)

    if valid_feature_mask.any():
        b_idx, t_idx = valid_feature_mask.nonzero(as_tuple=True)
        f_idx = feat_ids[b_idx, t_idx]
        seg = gap_cumsum[b_idx, t_idx]

        max_gaps = int(gap_cumsum[:, -1].max().item()) + 1
        gap_pos_table = torch.zeros(B, max_gaps, dtype=torch.long, device=device)
        gb, gt = full_gap_mask.nonzero(as_tuple=True)
        gk = gap_cumsum[gb, gt]
        gap_pos_table[gb, gk] = gt

        gap_t_idx = gap_pos_table[b_idx, seg]

        valid = gap_mask[b_idx, gap_t_idx]
        b_idx = b_idx[valid]
        gap_t_idx = gap_t_idx[valid]
        f_idx = f_idx[valid]

        targets[b_idx, gap_t_idx, f_idx] = 1.0

    return gap_mask, targets


class ICULlamaGap(ICULlamaCFG):
    """
    ICU LLaMA with gap prediction multi-task learning.

    Extends ICULlamaCFG with an auxiliary head that predicts which features
    will appear in the next timepoint at each GAP token.

    The total loss is:
        loss = loss_ce + lambda_gap * (loss_existence + loss_classification)

    Args (from config):
        num_features: Number of distinct features (default from datamodule)
        bins_per_feature: Number of bins per feature (default from datamodule)
        lambda_gap: Weight for gap prediction loss (default: 0.1)
    """

    def instantiate_model(self, cfg):
        """Instantiate the model with gap prediction head."""
        config = hydra.utils.call(cfg.model, vocabulary=self.vocabulary).config
        self._setup_time_embeddings(config)

        self.prefix_token_count = config.prefix_token_count
        hidden_size = config.hidden_size

        # Get gap prediction config (with defaults)
        gap_cfg = getattr(cfg, "gap_prediction", None)
        if gap_cfg is not None:
            self.num_features = getattr(gap_cfg, "num_features", None)
            self.bins_per_feature = getattr(gap_cfg, "bins_per_feature", None)
            self.lambda_gap = getattr(gap_cfg, "lambda_gap", 0.1)
        else:
            self.num_features = None
            self.bins_per_feature = None
            self.lambda_gap = 0.1

        # Infer from datamodule tokenizer if not set in config
        if self.num_features is None and hasattr(self._datamodule, "tokenizer"):
            self.num_features = self._datamodule.tokenizer.num_features
        if self.bins_per_feature is None and hasattr(self._datamodule, "tokenizer"):
            self.bins_per_feature = self._datamodule.tokenizer.num_bins

        # Validate that we have the necessary parameters
        if self.num_features is None or self.bins_per_feature is None:
            raise ValueError(
                "num_features and bins_per_feature must be specified in config "
                "or available from datamodule tokenizer"
            )

        # Build feature_indices tensor: maps token_id → feature_index
        # -1 for special tokens (PAD, SOS, EOS, GAP)
        self._build_feature_indices()

        # Instantiate the label embedder
        self.label_embedder = LabelEmbedder(
            num_classes=config.num_classes,
            hidden_size=hidden_size,
            dropout_prob=config.label_dropout,
            token_count=self.prefix_token_count,
            num_centers=getattr(config, "num_centers", 2),
            center_dropout_prob=getattr(config, "center_dropout_prob", 0.3),
        )

        # Build gap prediction head (MLP: hidden_size → hidden_size → num_features)
        self.gap_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(hidden_size, self.num_features),
        )

        self.model = LlamaWithTimePos(
            config=config,
            use_time_embeddings=self.use_time_embeddings,
            gap_token_id=self.vocabulary["<GAP>"],
            gap_head=self.gap_head,
        )

    def _build_feature_indices(self):
        """
        Build a tensor mapping token IDs to feature indices.

        For feature tokens (e.g., "heart_rate_bin_0"), maps to feature index.
        For special tokens, maps to -1.
        """
        vocab_size = len(self.vocabulary)
        feature_indices = torch.full((vocab_size,), -1, dtype=torch.long)

        # Get sorted feature list from vocabulary
        # Exclude demographic tokens (those ending with _static)
        feature_tokens = [
            tok
            for tok in self.vocabulary.keys()
            if "_bin_" in tok and not tok.startswith("<") and "_static_bin_" not in tok
        ]

        # Extract unique feature names and build mapping
        feature_names = sorted(set(tok.rsplit("_bin_", 1)[0] for tok in feature_tokens))
        feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

        # Validate against num_features
        if len(feature_name_to_idx) != self.num_features:
            raise ValueError(
                f"Number of features in vocabulary ({len(feature_name_to_idx)}) does not match "
                f"num_features ({self.num_features}). Check vocabulary or config."
            )

        # Map each token to its feature index
        for tok, tok_id in self.vocabulary.items():
            if "_bin_" in tok and not tok.startswith("<"):
                feature_name = tok.rsplit("_bin_", 1)[0]
                if feature_name in feature_name_to_idx:
                    feature_indices[tok_id] = feature_name_to_idx[feature_name]

        self.register_buffer("feature_indices", feature_indices)

    def _compute_gap_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        gap_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gap prediction loss.

        Loss consists of:
        1. Existence loss: BCE on whether any feature appears (1 if any target > 0)
        2. Classification loss: BCE on per-feature multi-hot targets

        Args:
            hidden_states: (B, T+prefix, H) hidden states from model
            input_ids: (B, T) original input token IDs (without prefix)
            gap_mask: (B, T) boolean mask for GAP positions
            targets: (B, T, num_features) multi-hot targets

        Returns:
            Combined gap prediction loss
        """
        # Skip prefix positions when extracting hidden states for gap prediction
        prefix = self.actual_prefix_len
        hidden_for_gap = hidden_states[:, prefix:, :]  # (B, T, H)

        # Get gap head predictions
        gap_logits = self.gap_head(hidden_for_gap)  # (B, T, num_features)

        # Flatten to positions with GAP tokens only
        gap_positions = gap_mask.nonzero(as_tuple=True)
        if len(gap_positions[0]) == 0:
            # No GAP tokens in this batch
            return torch.tensor(0.0, device=hidden_states.device)

        pred = gap_logits[gap_positions]  # (N, num_features)
        tgt = targets[gap_positions]  # (N, num_features)

        # Existence loss: does any feature appear?
        existence_pred = pred.max(dim=-1).values  # (N,)
        existence_tgt = (tgt.sum(dim=-1) > 0).float()  # (N,)
        loss_existence = nn.functional.binary_cross_entropy_with_logits(
            existence_pred, existence_tgt
        )

        # Classification loss: which features appear
        loss_classification = nn.functional.binary_cross_entropy_with_logits(pred, tgt)

        return loss_existence + loss_classification

    def shared_step(self, batch, batch_idx, phase="train"):
        """
        Shared step with gap prediction multi-task learning.

        Computes:
        - loss_ce: Cross-entropy loss for next token prediction
        - loss_gap: BCE loss for gap prediction
        - total: loss_ce + lambda_gap * loss_gap
        """
        if len(batch) == 5:
            X, time_positions, attention_mask, S, IDs = batch
        else:
            X, attention_mask, S, IDs = batch
            time_positions = None

        center_ids = IDs[:, -1] if IDs.size(1) > 3 else None

        # Forward pass with hidden states
        outputs = self.forward(
            input_ids=X,
            class_labels=IDs[:, 1],
            center_ids=center_ids,
            time_positions=time_positions,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        # Extract logits and hidden states from output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]

        else:
            # Fallback if forward returns just logits
            logits = outputs
            hidden_states = None

        loss_ce = self._next_token_loss(logits, X)
        loss_gap = torch.tensor(0.0, device=X.device)
        if hidden_states is not None:
            loss_gap = self._next_gap_loss(hidden_states, X)
        loss_balance = self._moe_balance_loss(device=X.device)
        lambda_balance = getattr(self.model.config, "lambda_balance", 0.05)

        # Combined loss
        loss = loss_ce + (self.lambda_gap * loss_gap) + (lambda_balance * loss_balance)

        self._log_step_losses(
            phase=phase,
            losses={
                "loss": loss,
                "loss_ce": loss_ce,
                "loss_gap": loss_gap,
                "loss_balance": loss_balance,
            },
            batch_size=X.size(0),
        )
        return loss

    def _next_token_loss(self, logits, X):
        """Cross-entropy of each token given the ones before (same as parent).

        Logits carry the conditioning prefix first; PAD targets are ignored.
        """
        T = X.size(1)
        V = logits.size(-1)
        prefix = self.actual_prefix_len
        assert logits.size(1) >= T + prefix

        logits = logits[:, : T + prefix, :]
        shift_logits = logits[:, prefix:-1, :]
        shift_labels = X[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.vocabulary["<PAD>"])
        return loss_fct(shift_logits.reshape(-1, V), shift_labels.reshape(-1))

    def _next_gap_loss(self, hidden_states, X):
        """Loss of predicting, at each GAP, which features the next time block holds."""
        gap_mask, targets = build_next_gap_targets(
            input_ids=X,
            gap_token_id=self.vocabulary["<GAP>"],
            feature_indices=self.feature_indices,
            num_features=self.num_features,
            bins_per_feature=self.bins_per_feature,
            skip_demo_gap=False,  # Include all GAPs in loss (including demographics separator)
        )
        assert targets.shape[:2] == X.shape
        return self._compute_gap_loss(
            hidden_states=hidden_states,
            input_ids=X,
            gap_mask=gap_mask,
            targets=targets,
        )

    def _moe_balance_loss(self, device):
        """Penalty on experts routed less than half of uniform usage (0 if no MoE)."""
        if (
            hasattr(self.model, "current_routing_probs")
            and self.model.current_routing_probs is not None
        ):
            p_i = self.model.current_routing_probs
            min_usage = 1.0 / (p_i.size(0) * 2)
            return F.relu(min_usage - p_i).sum()
        return torch.tensor(0.0, device=device)

    def _log_step_losses(self, phase, losses, batch_size):
        """Log each loss term, per step in training and per epoch otherwise."""
        on_step = phase == "train"
        for name, value in losses.items():
            self.log(
                f"{phase}_{name}",
                value,
                on_step=on_step,
                on_epoch=not on_step,
                sync_dist=True,
            )
        if hasattr(self.model, "current_expert_usage"):
            for i, usage in enumerate(self.model.current_expert_usage):
                self.log(
                    f"moe_diagnostics/{phase}_expert_{i}_usage",
                    # float32 like the former usage.item(), which Lightning turned
                    # into a float32 tensor, but without a host-GPU round trip.
                    usage.float(),
                    on_step=on_step,
                    on_epoch=not on_step,
                    prog_bar=False,
                    batch_size=batch_size,
                    sync_dist=True,
                )
