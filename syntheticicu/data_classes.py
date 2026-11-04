#!/usr/bin/env python

"""data_classes.py: Data loading classes for the ICUTranslator project."""

__author__ = "Lorena Gallego Viñarás, Emma Redor and Lucas Miranda"

from collections import OrderedDict, defaultdict, Counter
import copy
import glob
import os
from sklearn.model_selection import train_test_split
from functools import partial
from typing import Union
import math
import sys
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

try:
    import sparse
except ImportError:
    sparse = None
from tqdm import tqdm
import warnings
import joblib
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import re
import importlib
import types
from typing import List, Optional, Tuple, Any, Dict, Union


from syntheticicu.utils import (
    get_feature_set,
    parse_icu_data,
    pad_collate,
    subset_icu_dataset,
    format_set,
    fiddle_collate,
)

try:
    from syntheticicu.preprocessing.FIDDLE.run import run_fiddle
except ImportError:
    run_fiddle = None

from functools import partial

try:
    from transformers import LogitsProcessor
except ImportError:

    class LogitsProcessor:
        pass


from typing import List, Optional, Tuple, Any
from joblib import Parallel, delayed

try:
    pd.set_option("future.no_silent_downcasting", True)
except AttributeError:
    pass

# Centre → integer id mapping, shared across the dataset and the model
# (used both for conditioning and for resolving per-centre truncation caps).
CENTRE_MAP = {
    "miiv": 0,
    "eicu": 1,
    "mimic": 2,
    "sic": 3,
    "aumc": 4,
    "hirid": 5,
}


class FIDDLETokenizer:
    def __init__(
        self,
        features,
        num_bins=5,
        shuffle_tokens_within_timepoint=True,
        add_time_point_positional_embeddings=True,
        label=None,
        include_demographics=True,
        demographic_bins=None,
    ):
        """Initialize a FIDDLE tokenizer with a single vocabulary.

        Args:
            features (list): The list of features to tokenize.
            num_bins (int): The number of bins to discretize the features. Defaults to 5.
            shuffle_tokens_within_timepoint (bool): Whether to shuffle tokens within timepoint (for future use)
            add_time_point_positional_embeddings (bool): Whether to add time position embeddings (for future use)
            label: Label type for conditioning tokens (optional)
            include_demographics (bool): Whether to include demographic tokens in sequences. Defaults to True.
            demographic_bins (dict): Number of bins per demographic feature. Defaults to
                female 2, age/height/weight 5. ICUDataset passes the dataset's own bins.
        """
        self.features = sorted(features)
        self.num_features = len(features)
        self.num_bins = num_bins
        self.shuffle_tokens_within_timepoint = shuffle_tokens_within_timepoint
        self.add_time_point_positional_embeddings = add_time_point_positional_embeddings
        self.label_type = label
        self.include_demographics = include_demographics

        # Demographic features and their bin counts (fixed order)
        self.demographic_features = [
            "female_static",
            "age_static",
            "height_static",
            "weight_static",
        ]
        if demographic_bins is None:
            demographic_bins = {
                "female_static": 2,  # Binary: 0 or 1
                "age_static": 5,
                "height_static": 5,
                "weight_static": 5,
            }
        self.demographic_bins = {
            feature: int(demographic_bins[feature])
            for feature in self.demographic_features
        }

        # Create single vocabulary with one token per feature per value
        self.vocab = OrderedDict()
        self.id_to_token = OrderedDict()

        # Add time-series feature-value combinations
        for feature in self.features:
            for bin_idx in range(num_bins):
                token = f"{feature}_bin_{bin_idx}"
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.id_to_token[token_id] = token

        # Add demographic tokens if enabled (one per bin, 17 with the default 2 + 5 + 5 + 5)
        self.demographic_token_ids = {}  # Maps feature_name -> list of token_ids
        if self.include_demographics:
            for demo_feature in self.demographic_features:
                num_demo_bins = self.demographic_bins[demo_feature]
                self.demographic_token_ids[demo_feature] = []
                for bin_idx in range(num_demo_bins):
                    token = f"{demo_feature}_bin_{bin_idx}"
                    token_id = len(self.vocab)
                    self.vocab[token] = token_id
                    self.id_to_token[token_id] = token
                    self.demographic_token_ids[demo_feature].append(token_id)

        # Add special tokens
        special_tokens = ["<SOS>", "<GAP>", "<EOS>", "<PAD>"]
        # special_tokens = ["<SOS>", "<GAP>", "<EOS>", "<PAD>", "<CONTROL>"]

        # # Add label-specific SOS token if enabled
        # if self.label_type is not None:
        #     labels = (
        #         [self.label_type]
        #         if isinstance(self.label_type, str)
        #         else list(self.label_type)
        #     )
        #     special_tokens += [f"<SOS_{lbl.upper()}>" for lbl in labels]
        #     # special_tokens += [f"<{lbl.upper()}>" for lbl in labels]

        for token in special_tokens:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.id_to_token[token_id] = token

        self.vocab_size = len(self.vocab)

    def encode(self, features):
        """
        Encode features for a single timepoint.

        Args:
            features: (num_features * num_bins,) one-hot encoded array

        Returns:
            list: List of token IDs for active features
        """
        # Reshape to (num_features, num_bins) for processing
        features_reshaped = features.reshape(self.num_features, self.num_bins)

        active_tokens = []

        for feat_idx, feature_name in enumerate(self.features):
            # Find which bin is active for this feature
            active_bins = np.where(features_reshaped[feat_idx])[0]

            # Add token for each active bin
            for bin_idx in active_bins:
                token = f"{feature_name}_bin_{bin_idx}"
                if token in self.vocab:
                    active_tokens.append(self.vocab[token])

        return active_tokens

    def decode(self, token_ids):
        """
        Decode token IDs back to feature vector.

        Args:
            token_ids: List of token IDs for a sequence

        Returns:
            List of (num_features * num_bins,) arrays for each timepoint
        """
        timepoints = []
        current_timepoint = np.zeros(self.num_features * self.num_bins)

        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "")

            if "<SOS" in token:
                # if self.label_type.upper() in token:
                continue
            elif token == "<GAP>":
                # End of current timepoint
                timepoints.append(current_timepoint.copy())
                current_timepoint = np.zeros(self.num_features * self.num_bins)
            elif token == "<EOS>":
                # End of sequence
                if np.any(current_timepoint):  # Add final timepoint if it has data
                    timepoints.append(current_timepoint.copy())
                break
            elif token == "<PAD>":
                continue
            elif "_bin_" in token:
                # Parse feature token
                parts = token.split("_bin_")
                if len(parts) == 2:
                    feature_name = parts[0]
                    try:
                        bin_idx = int(parts[1])
                        if (
                            feature_name in self.features
                            and 0 <= bin_idx < self.num_bins
                        ):
                            feat_idx = self.features.index(feature_name)
                            feature_bin_idx = feat_idx * self.num_bins + bin_idx
                            current_timepoint[feature_bin_idx] = 1
                    except ValueError:
                        continue

        return timepoints

    def encode_sequence(
        self, sequence, label_value=None, demographics=None, static_names=None
    ):
        """
        Encode a sequence of timesteps into linear token sequence.

        Args:
            sequence: List of (num_features * num_bins,) arrays
            label_value: Optional label value for conditioning (unused currently)
            demographics: Optional tensor of FIDDLE-processed demographics (17 bins).
                If provided and include_demographics=True, demographic tokens are
                prepended after SOS, followed by a GAP, before the time series.

        Returns:
            list or tuple:
                - If add_time_point_positional_embeddings is False: Linear sequence of token IDs
                - If add_time_point_positional_embeddings is True: (token_sequence, time_positions)
        """
        token_sequence = []
        time_positions = [] if self.add_time_point_positional_embeddings else None

        # Add SOS token
        sos_token = "<SOS>"
        # if self.label_type is not None and label_value[1]:
        #     val = float(label_value[1])
        #     if self.label_type == "death":
        #         sos_token = (
        #             f"<SOS_{self.label_type.upper()}>" if val == 2.0 else "<SOS>"
        #             # f"<{self.label_type.upper()}>" if val == 2.0 else "<CONTROL>"
        #         )
        #     else:
        #         sos_token = f"<SOS_{self.label_type.upper()}>"
        #         # sos_token = f"<{self.label_type.upper()}>"
        # else:
        #     sos_token = "<CONTROL>"

        token_sequence.append(self.vocab[sos_token])
        if self.add_time_point_positional_embeddings:
            time_positions.append(0)  # SOS is at time position 0

        # Add demographic tokens if enabled and provided
        if self.include_demographics and demographics is not None:
            demo_tokens = self.encode_demographics(
                demographics, static_names=static_names
            )
            token_sequence.extend(demo_tokens)
            if self.add_time_point_positional_embeddings:
                # All demographic tokens have time position 0
                time_positions.extend([0] * len(demo_tokens))

            # Add GAP after demographics to separate from time series
            token_sequence.append(self.vocab["<GAP>"])
            if self.add_time_point_positional_embeddings:
                time_positions.append(0)  # Demographics GAP is also at time position 0

        current_time_position = 1  # Start counting from 1 for actual timepoints

        # Encode each timestep
        for timestep in sequence:
            # Get active tokens for this timestep
            active_tokens = self.encode(timestep)

            # Add active tokens (shuffling will be done later if enabled)
            token_sequence.extend(active_tokens)

            # Add time positions for active tokens
            if self.add_time_point_positional_embeddings:
                time_positions.extend([current_time_position] * len(active_tokens))

            # Add GAP token after each timestep
            token_sequence.append(self.vocab["<GAP>"])
            if self.add_time_point_positional_embeddings:
                time_positions.append(current_time_position)

            current_time_position += 1

        # Remove last GAP tokens with no measurements
        while token_sequence and token_sequence[-1] == self.vocab["<GAP>"]:
            token_sequence.pop()
            if self.add_time_point_positional_embeddings:
                time_positions.pop()

        # Add EOS token
        token_sequence.append(self.vocab["<EOS>"])
        if self.add_time_point_positional_embeddings:
            last_tp = time_positions[-1] if time_positions else 0
            time_positions.append(last_tp)

        if self.add_time_point_positional_embeddings:
            return token_sequence, time_positions
        else:
            return token_sequence

    # def _shuffle_tokens_within_timeblocks(self, token_sequence, time_positions):
    #     """
    #     Shuffle tokens within each time block while preserving SOS, GAP, and EOS positions.

    #     Args:
    #         token_sequence: List of token IDs
    #         time_positions: List of time positions (or None)

    #     Returns:
    #         tuple: (shuffled_token_sequence, shuffled_time_positions)
    #     """
    #     if token_sequence is None:
    #         return token_sequence, time_positions

    #     shuffled_tokens = token_sequence.clone()
    #     shuffled_positions = (
    #         time_positions.clone() if time_positions is not None else None
    #     )

    #     # Find positions of GAP tokens and special tokens
    #     gap_token_id = self.vocab["<GAP>"]
    #     sos_token_id = self.vocab["<SOS>"]
    #     eos_token_id = self.vocab["<EOS>"]
    #     # sos_condition_token_id = (
    #     #     self.vocab[f"<SOS_{self.label_type.upper()}>"]
    #     #     if self.label_type is not None
    #     #     else None
    #     # )

    #     # sos_condition_token_id = (
    #     #     self.vocab[f"<{self.label_type.upper()}>"]
    #     #     if self.label_type is not None
    #     #     else None
    #     # )
    #     # sos_control_token_id = self.vocab["<CONTROL>"]

    #     # Get indices of GAP tokens (and SOS/EOS which should not be moved)
    #     gap_indices = []
    #     for i, token in enumerate(token_sequence):
    #         if token in [
    #             gap_token_id,
    #             sos_token_id,
    #             eos_token_id,
    #             # sos_condition_token_id,
    #             # sos_control_token_id,
    #         ]:
    #             gap_indices.append(i)

    #     # Add start and end boundaries
    #     boundaries = [0] + gap_indices + [len(token_sequence)]

    #     # Shuffle tokens within each time block
    #     for i in range(len(boundaries) - 1):
    #         start = boundaries[i]
    #         end = boundaries[i + 1]

    #         # Skip if this is a boundary token (SOS, GAP, EOS)
    #         if start in gap_indices:
    #             start += 1
    #         if end - 1 in gap_indices:
    #             end -= 1

    #         # Only shuffle if there are tokens to shuffle in this block
    #         if start < end:
    #             # Get tokens and positions for this time block
    #             block_tokens = shuffled_tokens[start:end]
    #             block_positions = (
    #                 shuffled_positions[start:end]
    #                 if shuffled_positions is not None
    #                 else None
    #             )

    #             # Create shuffled indices
    #             indices = list(range(len(block_tokens)))
    #             np.random.shuffle(indices)

    #             # Apply shuffling
    #             shuffled_block_tokens = torch.tensor(
    #                 [block_tokens[idx] for idx in indices]
    #             )
    #             shuffled_tokens[start:end] = shuffled_block_tokens
    #             if block_positions is not None:
    #                 shuffled_block_positions = torch.tensor(
    #                     [block_positions[idx] for idx in indices]
    #                 )
    #                 shuffled_positions[start:end] = shuffled_block_positions

    #     return shuffled_tokens, shuffled_positions

    def decode_sequence(self, token_ids):
        """
        Decode a linear token sequence back to feature vectors.

        Args:
            token_ids: List of token IDs

        Returns:
            List of (num_features * num_bins,) arrays
        """
        return self.decode(token_ids)

    # Utility methods for special tokens
    def get_sos_token_id(self):
        """Get the SOS token ID."""
        return self.vocab["<SOS>"]

    # def get_sos_control_token_id(self):
    #     """Get the CONTROL token ID."""
    #     return self.vocab["<CONTROL>"]

    def get_sos_label_token_id(self):
        """Get the SOS_<LABEL> token ID."""
        return self.vocab.get(f"<SOS_{self.label_type.upper()}>", None)
        # return self.vocab.get(f"<{self.label_type.upper()}>", None)

    def get_eos_token_id(self):
        """Get the EOS token ID."""
        return self.vocab["<EOS>"]

    def get_gap_token_id(self):
        """Get the GAP token ID."""
        return self.vocab["<GAP>"]

    def get_pad_token_id(self):
        """Get the PAD token ID."""
        return self.vocab["<PAD>"]

    def get_token_id(self, token):
        """Get the token ID for a given token."""
        return self.vocab.get(token, None)

    def get_token(self, token_id):
        """Get the token for a given token ID."""
        return self.id_to_token.get(token_id, None)

    def __len__(self):
        """Return total vocabulary size."""
        return self.vocab_size

    def __repr__(self):
        return (
            f"FIDDLE_tokenizer("
            f"num_features={self.num_features}, "
            f"num_bins={self.num_bins}, "
            f"vocab_size={self.vocab_size}, "
            f"include_demographics={self.include_demographics})"
        )

    def get_demographic_token_ids(self):
        """Get the dictionary mapping demographic features to their token IDs."""
        return self.demographic_token_ids

    def get_num_demographic_tokens(self):
        """Get total number of demographic tokens (excluding the trailing GAP)."""
        if not self.include_demographics:
            return 0
        return sum(self.demographic_bins.values())  # 2 + 5 + 5 + 5 = 17

    def encode_demographics(self, processed_statics, static_names=None):
        """
        Convert FIDDLE-processed statics to demographic token IDs.

        Args:
            processed_statics: Tensor of shape (num_static_bins,) with one-hot encoded
                demographics from FIDDLE processing.
            static_names: Array of feature names from FIDDLE (e.g. self.static_names).
                If provided, the vector is read using the actual FIDDLE ordering inferred
                from these names, then emitted in demographic_features order.
                If None, falls back to assuming the vector is already in demographic_features order.

        Returns:
            List of 4 token IDs in demographic_features order:
            [female_token_id, age_token_id, height_token_id, weight_token_id]
        """
        if not self.include_demographics:
            return []

        demo_tokens = []

        if static_names is not None:
            # Build map: feature_name -> start index in the FIDDLE vector
            feature_start = {}
            for idx, name in enumerate(static_names):
                matched = next(
                    (d for d in self.demographic_features if name.startswith(d)),
                    None,
                )
                if matched and matched not in feature_start:
                    feature_start[matched] = idx

            # Read in desired order (female -> age -> height -> weight)
            for demo_feature in self.demographic_features:
                num_bins = self.demographic_bins[demo_feature]
                start = feature_start[demo_feature]
                feature_bins = processed_statics[start : start + num_bins]

                active_bins = torch.where(feature_bins == 1)[0]
                bin_idx = (
                    active_bins[0].item() if len(active_bins) > 0 else num_bins // 2
                )
                token_id = self.demographic_token_ids[demo_feature][bin_idx]
                demo_tokens.append(token_id)

        else:
            # Fallback used by tests and current generated fixtures:
            # female_static(2) -> age_static(5) -> height_static(5) -> weight_static(5).
            feature_offsets = {}
            offset = 0
            for feat in self.demographic_features:
                feature_offsets[feat] = offset
                offset += self.demographic_bins[feat]

            for demo_feature in self.demographic_features:
                num_bins = self.demographic_bins[demo_feature]
                start = feature_offsets[demo_feature]
                feature_bins = processed_statics[start : start + num_bins]

                active_bins = torch.where(feature_bins == 1)[0]
                bin_idx = (
                    active_bins[0].item() if len(active_bins) > 0 else num_bins // 2
                )
                token_id = self.demographic_token_ids[demo_feature][bin_idx]
                demo_tokens.append(token_id)

        return demo_tokens

    def encode_direct_demographics(self, processed_statics, statics_set=None):
        """Convert direct shared static value IDs to demographic token IDs."""
        if not self.include_demographics:
            return []

        if processed_statics is None:
            return []

        if torch.is_tensor(processed_statics):
            values = processed_statics.detach().cpu().tolist()
        else:
            values = np.asarray(processed_statics).tolist()

        if statics_set is None:
            statics_set = self.demographic_features

        values_by_feature = {
            feature: int(value) for feature, value in zip(statics_set, values)
        }
        demo_tokens = []

        for demo_feature in self.demographic_features:
            value = self.demographic_bin(
                demo_feature, values_by_feature.get(demo_feature, -1)
            )
            demo_tokens.append(self.demographic_token_ids[demo_feature][value])

        return demo_tokens

    def demographic_bin(self, feature, value):
        """
        Bin of a direct static value. Values outside [0, n_bins) are the missing
        or pad ids and get the median bin. ICUDataset._direct_statics_to_onehot
        applies the same rule, so tokens and the statics one-hot always agree.
        """
        num_demo_bins = self.demographic_bins[feature]
        value = int(value)
        if 0 <= value < num_demo_bins:
            return value
        return num_demo_bins // 2  # median imputation

    def encode_direct_sequence(self, value_ids, demographics=None, statics_set=None):
        """Linearize direct ragged FIDDLE value IDs for the autoregressive LM."""
        values = (
            value_ids.detach().cpu().numpy()
            if torch.is_tensor(value_ids)
            else np.asarray(value_ids)
        )
        token_sequence = [self.vocab["<SOS>"]]
        time_positions = [0] if self.add_time_point_positional_embeddings else None

        if self.include_demographics and demographics is not None:
            demo_tokens = self.encode_direct_demographics(
                demographics, statics_set=statics_set
            )
            token_sequence.extend(demo_tokens)
            token_sequence.append(self.vocab["<GAP>"])
            if self.add_time_point_positional_embeddings:
                time_positions.extend([0] * (len(demo_tokens) + 1))

        current_time_position = 1
        for row in values:
            for feature_idx, value in enumerate(row):
                value = int(value)
                if 0 <= value < self.num_bins and feature_idx < len(self.features):
                    token = f"{self.features[feature_idx]}_bin_{value}"
                    token_id = self.vocab.get(token)
                    if token_id is not None:
                        token_sequence.append(token_id)
                        if self.add_time_point_positional_embeddings:
                            time_positions.append(current_time_position)

            token_sequence.append(self.vocab["<GAP>"])
            if self.add_time_point_positional_embeddings:
                time_positions.append(current_time_position)
            current_time_position += 1

        while token_sequence and token_sequence[-1] == self.vocab["<GAP>"]:
            token_sequence.pop()
            if self.add_time_point_positional_embeddings:
                time_positions.pop()

        token_sequence.append(self.vocab["<EOS>"])
        if self.add_time_point_positional_embeddings:
            last_tp = time_positions[-1] if time_positions else 0
            time_positions.append(last_tp)
            return token_sequence, time_positions
        return token_sequence


class CFGLogitsProcessor(LogitsProcessor):
    """
    Expects the generation batch to be duplicated as [uncond_batch; cond_batch].
    For each step receives 'scores' of shape (2B, V). It computes:
        guided = uncond + scale * (cond - uncond)
    and returns torch.cat([guided, guided], dim=0) (shape preserved).

    Memory-optimized: explicitly deletes intermediate tensors to reduce memory footprint.
    """

    def __init__(self, cfg_scale: float):
        self.cfg_scale = float(cfg_scale)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # scores: (2B, V)
        if self.cfg_scale <= 1.0:
            return scores

        B2, V = scores.shape
        if B2 % 2 != 0:
            # If batch not duplicated, just return scores unchanged
            return scores

        B = B2 // 2
        uncond = scores[:B]
        cond = scores[B:]

        # Apply CFG formula: guided = uncond + scale * (cond - uncond)
        guided = uncond + self.cfg_scale * (cond - uncond)
        # guided = torch.clamp(guided, min=-1e9, max=1e9)
        # guided = torch.nan_to_num(guided, nan=0.0, posinf=1e9, neginf=-1e9)

        # Explicitly delete intermediates to help garbage collector
        del uncond, cond

        # Replicate guided so returned shape is (2B, V) to match input batch
        guided_both = torch.cat([guided, guided], dim=0)

        del guided  # Free the intermediate

        return guided_both


def build_feature_indices(
    vocabulary: dict, exclude_demographics: bool = True
) -> tuple[torch.Tensor, int]:
    """
    Build a tensor mapping token IDs to feature indices for time-series features.

    For time-series feature tokens (e.g., "heart_rate_bin_0"), maps to feature index.
    For special tokens and demographic tokens, maps to -1.

    Args:
        vocabulary: Token vocabulary dict mapping token strings to IDs
        exclude_demographics: If True, demographic tokens (*_static_*) are excluded
            and mapped to -1. This prevents FeatureConstraintLogitsProcessor from
            affecting demographic tokens. Defaults to True.

    Returns:
        Tuple of (feature_indices tensor of shape (vocab_size,), num_features)
    """
    vocab_size = max(vocabulary.values(), default=-1) + 1
    feature_indices = torch.full((vocab_size,), -1, dtype=torch.long)

    # Get feature tokens (those with _bin_ pattern, excluding special tokens)
    feature_tokens = [
        tok for tok in vocabulary.keys() if "_bin_" in tok and not tok.startswith("<")
    ]

    # Optionally exclude demographic/static features
    if exclude_demographics:
        feature_tokens = [tok for tok in feature_tokens if "_static" not in tok]

    # Extract unique feature names and build mapping
    feature_names = sorted(set(tok.rsplit("_bin_", 1)[0] for tok in feature_tokens))
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # Map each token to its feature index
    for tok, tok_id in vocabulary.items():
        if "_bin_" in tok and not tok.startswith("<"):
            feature_name = tok.rsplit("_bin_", 1)[0]
            # Skip demographic features if excluding them
            if exclude_demographics and "_static" in tok:
                continue
            if feature_name in feature_name_to_idx:
                feature_indices[tok_id] = feature_name_to_idx[feature_name]

    return feature_indices, len(feature_name_to_idx)


class FeatureConstraintLogitsProcessor(LogitsProcessor):
    """
    Enforces that each feature appears at most once per time block.

    A time block is the set of tokens between consecutive GAP tokens. Within a block,
    once a token for feature X is generated, all tokens for feature X are masked
    (logits set to -inf) until the next GAP token is generated.

    This ensures generated sequences follow the real data structure where each
    timepoint has at most one measurement per feature.
    """

    def __init__(
        self,
        feature_indices: torch.Tensor,
        gap_token_id: int,
        num_features: int,
    ):
        """
        Args:
            feature_indices: (vocab_size,) tensor mapping token_id → feature_idx (-1 for special tokens)
            gap_token_id: Token ID for <GAP>
            num_features: Total number of distinct features
        """
        self.feature_indices = feature_indices
        self.gap_token_id = gap_token_id
        self.num_features = num_features
        # Copied to the scores' device on first use. Copying it on every decoding
        # step would make the host wait for the GPU each time.
        self._feature_indices_on_device = feature_indices

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size, seq_len = input_ids.shape
        device = scores.device
        assert scores.shape[0] == batch_size
        assert scores.shape[1] <= self.feature_indices.numel()

        if self._feature_indices_on_device.device != device:
            self._feature_indices_on_device = self.feature_indices.to(device)
        feature_indices = self._feature_indices_on_device

        # Step 1: Find tokens after last GAP using cumsum trick
        is_gap = input_ids == self.gap_token_id  # (B, T)
        gap_cumsum = is_gap.long().cumsum(dim=1)  # (B, T)
        total_gaps = gap_cumsum[:, -1:]  # (B, 1)

        # Mask for positions after last GAP (excluding GAP itself)
        after_last_gap = (gap_cumsum == total_gaps) & ~is_gap  # (B, T)

        # Step 2: Get feature index for each token in sequence
        token_features = feature_indices[input_ids]  # (B, T)
        valid_feature = token_features >= 0  # (B, T)
        relevant = after_last_gap & valid_feature  # (B, T)

        # Step 3: Build used_features mask (B, num_features). Irrelevant positions
        # scatter into a spare last column that is dropped, so there is no
        # data-dependent indexing (`.any()`, boolean masks) that waits for the GPU.
        feature_slots = torch.where(relevant, token_features, self.num_features)
        used_features = torch.zeros(
            batch_size, self.num_features + 1, dtype=torch.bool, device=device
        )
        used_features.scatter_(
            1, feature_slots, torch.ones_like(feature_slots, dtype=torch.bool)
        )
        used_features = used_features[:, : self.num_features]

        # Step 4: Build vocab mask (B, V)
        vocab_size = scores.shape[1]
        vocab_features = (
            feature_indices[:vocab_size].unsqueeze(0).expand(batch_size, -1)
        )
        valid_vocab = vocab_features >= 0

        # Clamp to valid range for gather (special tokens have -1)
        vocab_features_safe = vocab_features.clamp(min=0)

        # Gather: tokens_to_mask[b,v] = used_features[b, feature_of_v]
        tokens_to_mask = torch.gather(used_features, dim=1, index=vocab_features_safe)
        tokens_to_mask = tokens_to_mask & valid_vocab  # Don't mask special tokens

        # Step 5: Apply mask
        scores = scores.masked_fill(tokens_to_mask, float("-inf"))

        return scores


class DemographicsLogitsProcessor(LogitsProcessor):
    """
    Demographic constraint for autoregressive generation.

    Two modes, selected via `pre_generated`:

    pre_generated=False  (no external demo model)
        Positions 0-3 relative to start_pos: force each demographic token
        in fixed order (female → age → height → weight).
        Position 4: force <GAP> separator.
        Position 5+: block ALL demographic tokens from ever appearing again.

    pre_generated=True  (demographics already prepended from e.g. TVAE)
        Block ALL demographic tokens at every position, unconditionally.
        The demos are already in the prompt; the model must never re-emit them.
    -------
    Args:
        demographic_token_ids: dict mapping feature name → list of valid token IDs
            e.g. {"female_static": [10, 11], "age_static": [12, 13, 14, 15, 16], ...}
        gap_token_id: token ID for <GAP>
        vocab_size: total vocabulary size
        start_pos: number of real prompt token IDs before demographic generation
            starts (e.g. 1 for [SOS], 3 for [SOS, CENTRE, COND]).
        pre_generated: if True, demographics are already in the prompt → block always
    """

    def __init__(
        self,
        demographic_token_ids: dict,
        gap_token_id: int,
        vocab_size: int,
        start_pos: int = 1,
        pre_generated: bool = False,  # True when demo came from an external model
    ):
        self.demographic_order = [
            "female_static",
            "age_static",
            "height_static",
            "weight_static",
        ]
        self.demographic_token_ids = demographic_token_ids
        self.gap_token_id = gap_token_id
        self.vocab_size = vocab_size
        self.start_pos = start_pos
        self.pre_generated = pre_generated

        # Precompute valid token masks for each demographic position
        self.demo_masks = []
        for demo_feature in self.demographic_order:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            for tok_id in demographic_token_ids[demo_feature]:
                mask[tok_id] = True
            self.demo_masks.append(mask)

        # GAP-only mask
        self.gap_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self.gap_mask[gap_token_id] = True

        # Mask of ALL demographic tokens (to block them after generation)
        self.all_demo_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for demo_feature in self.demographic_order:
            for tok_id in demographic_token_ids[demo_feature]:
                self.all_demo_mask[tok_id] = True

        # Device copies of the masks, made on first use. Copying them on every
        # decoding step would make the host wait for the GPU each time.
        self._masks_device = None
        self._demo_masks_on_device = None
        self._gap_mask_on_device = None
        self._all_demo_mask_on_device = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        device = scores.device
        assert scores.shape[1] == self.vocab_size
        if self._masks_device != device:
            self._demo_masks_on_device = [mask.to(device) for mask in self.demo_masks]
            self._gap_mask_on_device = self.gap_mask.to(device)
            self._all_demo_mask_on_device = self.all_demo_mask.to(device)
            self._masks_device = device

        seq_len = input_ids.shape[1]
        relative_pos = (
            seq_len - self.start_pos
        )  # number of demographics already generated

        if self.pre_generated:
            # Demographics already in prompt — block them everywhere, always
            block_mask = self._all_demo_mask_on_device
            scores = scores.masked_fill(block_mask.unsqueeze(0), float("-inf"))

        else:
            if 0 <= relative_pos <= 3:
                # Force the correct demographic token for this position
                valid_mask = self._demo_masks_on_device[relative_pos]
                scores = scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))

            elif relative_pos == 4:
                # Force GAP separator
                gap_mask = self._gap_mask_on_device
                scores = scores.masked_fill(~gap_mask.unsqueeze(0), float("-inf"))

            else:
                # Demographics already generated — block them from reappearing
                block_mask = self._all_demo_mask_on_device
                scores = scores.masked_fill(block_mask.unsqueeze(0), float("-inf"))

        return scores


class ICUDataset:
    """
    Base ICU dataset class that handles all data loading, preprocessing,
    tokenization, FIDDLE processing, and ICU-specific operations.

    This class manages the complex data pipeline but doesn't implement
    PyTorch Dataset interface - that's handled by child classes.
    """

    def __init__(
        self,
        core_path: str = ".",
        meta_path: str = ".",
        pkl_path: Optional[str] = None,
        manage_missing_statics: str = "impute",
        sources: List[str] = ["eicu", "miiv", "aumc", "mimic", "hirid", "sic"],
        ids_to_keep: Optional[List] = None,
        max_gap: int = 6,
        label: Optional[str] = None,
        min_length: Optional[int] = 6,
        min_onset: int = 6,
        samples: Optional[int] = None,
        X_FIDDLE_bins: Optional[dict] = None,
        S_FIDDLE_bins: Optional[dict] = None,
        return_patient_ids: bool = False,
        include_ed: bool = False,
        feature_set: Union[str, List] = "all",
        statics_set: List[str] = [
            "female_static",
            "age_static",
            "height_static",
            "weight_static",
        ],
        time_shift: Union[int, Tuple[int, int]] = 1,
        masking_fraction: float = 0,
        num_workers: int = 1,
        run_fiddle: bool = True,
        get_item_mode: str = "FIDDLE",
        verbose=False,
        shuffle_tokens_within_timepoint: bool = False,
        add_time_point_positional_embeddings: bool = True,
        return_label="death_icu",
        demo=False,
        demo_fraction: float = 0.1,
        include_demographics: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize ICU dataset with all preprocessing parameters.

        Args:
            core_path: Path to core ICU data
            meta_path: Path to metadata
            pkl_path: Path to pre-saved dataset (optional)
            manage_missing_statics: How to handle missing static features ('impute', 'ignore', 'drop')
            sources: List of data sources to include
            ids_to_keep: Specific patient/stay IDs to keep
            max_gap: Maximum allowed gap in measurements
            label: Outcome label to predict
            min_length: Minimum length of stay to include
            min_onset: Minimum onset time for positive cases
            samples: Number of random samples to take
            X_FIDDLE_bins: Pre-computed FIDDLE bins for time series
            S_FIDDLE_bins: Pre-computed FIDDLE bins for static features
            return_patient_ids: Whether to return patient vs stay IDs
            include_ed: Whether to include emergency department data
            feature_set: Features to use ('vitals', 'labs', 'all', or custom list)
            statics_set: Static features to include
            time_shift: Time shift for self-supervision
            masking_fraction: Fraction of tokens to mask for self-supervision
            num_workers: Number of parallel workers
            run_fiddle: Whether to run FIDDLE processing
            get_item_mode: Data format mode ("tuples", "FIDDLE")
        """
        # Store all configuration
        self.core_path = core_path
        self.meta_path = meta_path
        self.pkl_path = pkl_path
        self.manage_missing_statics = manage_missing_statics
        self.sources = sources
        self.ids_to_keep = ids_to_keep
        self.max_gap = max_gap
        self.label = [label] if isinstance(label, str) else label
        self.min_onset = min_onset
        self.min_length = min_length
        self.samples = samples
        self.X_FIDDLE_bins = X_FIDDLE_bins
        self.S_FIDDLE_bins = S_FIDDLE_bins
        self.return_patient_ids = return_patient_ids
        self.include_ed = include_ed
        self.statics_set = statics_set
        self.time_shift = time_shift
        self.masking_fraction = masking_fraction
        self.num_workers = num_workers
        self.run_fiddle = run_fiddle
        self.get_item_mode = get_item_mode
        self.verbose = verbose
        self.shuffle_tokens_within_timepoint = shuffle_tokens_within_timepoint
        self.add_time_point_positional_embeddings = add_time_point_positional_embeddings
        self.max_length = kwargs.get("max_length", 196)
        self.truncation_time = kwargs.get("truncation_time", None)
        self.return_label = return_label
        self.feature_set = feature_set
        self.demo = demo
        self.demo_fraction = demo_fraction
        self.include_demographics = include_demographics
        self.num_bins = kwargs.get("num_bins", 5)

        # Process feature set
        if isinstance(self.feature_set, str):
            self.features = sorted(get_feature_set(self.feature_set))
        else:
            self.features = sorted(self.feature_set)

        self.tokenizer = None
        self.processed_statics = None

        # Load and process data
        self._load_and_process_data()

        # Initialize processing state
        if self.is_tokenized:
            self.tokenize_data = False

        else:
            self.tokenize_data = kwargs.get("tokenize_data", True)

        # Run FIDDLE processing if requested
        if self.run_fiddle and self.pkl_path is None:
            self._run_fiddle_processing()

        if self.tokenize_data:
            if not hasattr(self, "tokenizer"):
                self.tokenizer = FIDDLETokenizer(
                    features=self.features,
                    num_bins=self.num_bins,
                    shuffle_tokens_within_timepoint=self.shuffle_tokens_within_timepoint,
                    add_time_point_positional_embeddings=self.add_time_point_positional_embeddings,
                    label=self.return_label,
                    include_demographics=self.include_demographics,
                    demographic_bins=self._demographic_bins(),
                )
            self.tokenize(fine_tune=kwargs.get("fine_tune", False))

        self._chunk_long_sequences(overlap_ratio=0.15)
        self._build_label_table()

    def _build_label_table(self):
        """Build y_full, one label row per sample: [stay_id, outcome, onset].

        __getitem__ reads labels from this table, so the per-stay metadata lookup
        runs once here instead of once per sample in every DataLoader worker.
        """
        if self.return_label is not None:
            # Reset index if needed before reindexing (e.g. after _make_demo_sample)
            if "stay_id" not in self.data_table.columns:
                self.data_table = self.data_table.reset_index()

            # Index data_table for O(log N) lookups instead of O(N) scans
            if "centre" in self.data_table.columns:
                self.data_table = (
                    self.data_table.drop_duplicates(subset=["centre", "stay_id"])
                    .set_index(["centre", "stay_id"])
                    .sort_index()
                )
            else:
                self.data_table = (
                    self.data_table.drop_duplicates(subset=["stay_id"])
                    .set_index("stay_id")
                    .sort_index()
                )

            all_labels = []
            for idx, y_id in enumerate(self.y):
                label_full = self.retrieve_labels_from_id(
                    y_id,
                    self.return_label,
                    source=self.stay_sources[idx],
                )
                all_labels.append(label_full)
            self.y_full = torch.tensor(np.array(all_labels, dtype=np.float32))
            assert len(self.y_full) == len(self.stay_sources)
        else:
            self.y_full = torch.tensor(self.y, dtype=torch.float32)
        assert len(self.y_full) == len(self.y)

    def _make_demo_sample(self):
        # Ensure stay_id and centre are columns, not index
        if "stay_id" not in self.data_table.columns:
            self.data_table = self.data_table.reset_index()

        # Sample unique stays by (centre, stay_id) to avoid cross-centre duplicates
        if "centre" in self.data_table.columns:
            unique_keys = list(
                set(
                    zip(
                        self.data_table["centre"].str.lower(),
                        self.data_table["stay_id"].astype(int),
                    )
                )
            )
        else:
            unique_keys = [(None, int(s)) for s in self.data_table["stay_id"].unique()]

        num_demo = max(1, int(len(unique_keys) * self.demo_fraction))
        selected_idx = np.random.choice(len(unique_keys), num_demo, replace=False)
        selected = set(unique_keys[i] for i in selected_idx)

        # Filter data_table
        if "centre" in self.data_table.columns:
            keep_mask_dt = [
                (str(c).lower(), int(s)) in selected
                for c, s in zip(self.data_table["centre"], self.data_table["stay_id"])
            ]
        else:
            keep_mask_dt = [
                (None, int(s)) in selected for s in self.data_table["stay_id"]
            ]
        self.data_table = self.data_table[keep_mask_dt].reset_index(drop=True)

        # Filter tensors by (source, stay_id)
        if hasattr(self, "y") and self.y is not None:
            y_ids = self.y[:, 0].numpy() if torch.is_tensor(self.y) else self.y[:, 0]
            sources = np.asarray(self.stay_sources)
            keep_mask = np.array(
                [
                    (str(sources[i]).lower(), int(y_ids[i])) in selected
                    for i in range(len(y_ids))
                ]
            )
            keep_idx = np.where(keep_mask)[0]

            for attr in [
                "X",
                "y",
                "lengths",
                "X_tokens",
                "X_time_positions",
                "statics",
                "times",
                "missing_mask",
            ]:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    arr = getattr(self, attr)
                    if isinstance(arr, list):
                        setattr(self, attr, [arr[i] for i in keep_idx])
                    elif torch.is_tensor(arr):
                        setattr(self, attr, arr[keep_idx])
                    elif isinstance(arr, np.ndarray):
                        setattr(self, attr, arr[keep_idx])

            self.stay_sources = sources[keep_idx]

    def _load_and_process_data(self):
        if self.pkl_path is not None:
            self._load_saved_dataset()
            if self.demo:
                self._make_demo_sample()
            return

    def _load_saved_dataset(self):
        """Load and filter one or more pre-saved pickled datasets."""
        loaded_dset = self._load_saved_dataset_object()

        self.is_tokenized = (
            hasattr(loaded_dset, "X_tokens") and loaded_dset.X_tokens is not None
        )
        self.is_direct_fiddle = hasattr(loaded_dset, "X_value_ids")

        identity_columns = self._metadata_identity_columns(loaded_dset.data_table)
        self.data_table = loaded_dset.data_table.drop_duplicates(
            subset=identity_columns, keep="first"
        ).reset_index(drop=True)

        self._apply_filters()
        if self.samples is not None:
            self._sample_data()
        keep_indices = self._indices_for_loaded_stays(
            loaded_dset, self._stay_keys_from_metadata(self.data_table)
        )
        filtered_dset = self._subset_loaded_dataset(loaded_dset, keep_indices)
        filtered_dset.data_table = self.data_table.copy()
        self.is_tokenized = (
            hasattr(filtered_dset, "X_tokens") and filtered_dset.X_tokens is not None
        )
        self.is_direct_fiddle = hasattr(filtered_dset, "X_value_ids")

        if self.is_tokenized:
            for attr, value in filtered_dset.__dict__.items():
                if attr in ["is_tokenized", "feature_set", "min_length"]:
                    continue
                setattr(self, attr, value)

            if hasattr(filtered_dset, "tokenizer"):
                self.tokenizer = filtered_dset.tokenizer
            self.demo = False
        else:
            self._transfer_from_loaded_dataset(filtered_dset)

    def _register_pickle_aliases(self):
        """Register old NAUTILUS module names used by saved pickle files."""
        import syntheticicu
        import syntheticicu.data_classes
        import syntheticicu.utils

        syntheticicu_module = importlib.import_module("syntheticicu")
        data_classes = importlib.import_module("syntheticicu.data_classes")
        utils_module = importlib.import_module("syntheticicu.utils")

        sys.modules["nautilus"] = syntheticicu_module
        sys.modules["nautilus.data_classes"] = data_classes
        sys.modules["nautilus.utils"] = utils_module

        datasets_module = types.ModuleType("nautilus.datasets")
        datasets_module.ICUDataset = data_classes.ICUDataset
        datasets_module.NAUTILUSDataset = data_classes.ICUDataset
        sys.modules["nautilus.datasets"] = datasets_module

        tokenizer_module = types.ModuleType("nautilus.tokenizer")
        tokenizer_module.FIDDLETokenizer = data_classes.FIDDLETokenizer
        sys.modules["nautilus.tokenizer"] = tokenizer_module

    def _resolve_pkl_paths(self):
        """Resolve a file, directory, glob, or iterable of pickle paths."""
        if isinstance(self.pkl_path, (list, tuple)):
            paths = [str(path) for path in self.pkl_path]
        else:
            pkl_path = str(self.pkl_path)
            if os.path.isdir(pkl_path):
                paths = sorted(glob.glob(os.path.join(pkl_path, "*.pkl")))
            elif any(char in pkl_path for char in "*?["):
                paths = sorted(glob.glob(pkl_path))
            else:
                paths = [pkl_path]

        paths = [path for path in paths if path.endswith(".pkl")]
        if not paths:
            raise FileNotFoundError(f"No .pkl files found for pkl_path={self.pkl_path}")
        return paths

    def _load_saved_dataset_object(self):
        """Load one or more saved dataset objects and merge shards if needed."""
        self._register_pickle_aliases()
        paths = self._resolve_pkl_paths()
        datasets = [joblib.load(path, mmap_mode=None) for path in paths]
        if len(datasets) == 1:
            return datasets[0]
        return self._merge_loaded_datasets(datasets)

    def _merge_loaded_datasets(self, datasets):
        """Merge source/fixed-size shards saved on the per-stay axis."""
        merged = copy.copy(datasets[0])
        merged.data_table = pd.concat(
            [dataset.data_table for dataset in datasets], ignore_index=True
        )

        sequence_attrs = [
            "X",
            "X_value_ids",
            "X_feature_ids",
            "statics",
            "processed_statics",
            "times",
            "missing_mask",
            "lengths",
            "y",
            "stay_sources",
            "X_tokens",
            "X_time_positions",
        ]
        for attr in sequence_attrs:
            values = [
                getattr(dataset, attr) for dataset in datasets if hasattr(dataset, attr)
            ]
            if values:
                setattr(merged, attr, self._concat_loaded_values(values))
        return merged

    @staticmethod
    def _concat_loaded_values(values):
        if all(isinstance(value, list) for value in values):
            merged = []
            for value in values:
                merged.extend(value)
            return merged
        if all(torch.is_tensor(value) for value in values):
            return torch.cat(values, dim=0)
        if all(isinstance(value, np.ndarray) for value in values):
            return np.concatenate(values, axis=0)
        if all(isinstance(value, pd.Series) for value in values):
            return pd.concat(values, ignore_index=True)
        arrays = [np.asarray(value) for value in values]
        return np.concatenate(arrays, axis=0)

    @staticmethod
    def _take_loaded_values(values, indices):
        if values is None:
            return None
        if isinstance(values, list):
            return [values[int(i)] for i in indices]
        if torch.is_tensor(values):
            return values[torch.as_tensor(indices, dtype=torch.long)]
        if isinstance(values, np.ndarray):
            return values[indices]
        if isinstance(values, pd.Series):
            return values.iloc[indices].reset_index(drop=True)
        return np.asarray(values)[indices]

    @staticmethod
    def _stay_ids_from_dataset(dataset):
        y = dataset.y
        if torch.is_tensor(y):
            return y[:, 0].detach().cpu().numpy()
        return np.asarray(y)[:, 0]

    @staticmethod
    def _normalize_source(source):
        return str(source).lower()

    @staticmethod
    def _metadata_identity_columns(data_table):
        return ["centre", "stay_id"] if "centre" in data_table else ["stay_id"]

    def _stay_keys_from_metadata(self, data_table):
        if "centre" not in data_table:
            return [(None, int(stay_id)) for stay_id in data_table.stay_id]
        return [
            (self._normalize_source(source), int(stay_id))
            for source, stay_id in zip(data_table.centre, data_table.stay_id)
        ]

    def _stay_keys_from_dataset(self, dataset):
        stay_ids = self._stay_ids_from_dataset(dataset)
        if not hasattr(dataset, "stay_sources") or "centre" not in dataset.data_table:
            return [(None, int(stay_id)) for stay_id in stay_ids]
        sources = np.asarray(dataset.stay_sources)
        if len(sources) != len(stay_ids):
            raise ValueError("stay_sources and y must share the stay axis.")
        return [
            (self._normalize_source(source), int(stay_id))
            for source, stay_id in zip(sources, stay_ids)
        ]

    def _indices_for_loaded_stays(self, dataset, stay_keys):
        loaded_stay_keys = self._stay_keys_from_dataset(dataset)
        if len(set(loaded_stay_keys)) != len(loaded_stay_keys):
            raise ValueError(
                "Loaded shards contain duplicate (centre, stay_id) identities."
            )
        stay_to_index = {key: idx for idx, key in enumerate(loaded_stay_keys)}
        missing = [key for key in stay_keys if key not in stay_to_index]
        if missing:
            raise ValueError(
                f"Metadata stays are missing from loaded tensors: {missing[:5]}"
            )
        indices = [stay_to_index[key] for key in stay_keys]
        return np.asarray(indices, dtype=np.int64)

    def _subset_loaded_dataset(self, dataset, indices):
        subset = copy.copy(dataset)
        sequence_attrs = [
            "X",
            "X_value_ids",
            "X_feature_ids",
            "statics",
            "processed_statics",
            "times",
            "missing_mask",
            "lengths",
            "y",
            "stay_sources",
            "X_tokens",
            "X_time_positions",
        ]
        for attr in sequence_attrs:
            if hasattr(dataset, attr):
                setattr(
                    subset,
                    attr,
                    self._take_loaded_values(getattr(dataset, attr), indices),
                )

        keep_stay_keys = set(self._stay_keys_from_dataset(subset))
        metadata_keys = self._stay_keys_from_metadata(dataset.data_table)
        subset.data_table = dataset.data_table.loc[
            [key in keep_stay_keys for key in metadata_keys]
        ].copy()
        return subset

    def _load_raw_data(self):
        """Load raw ICU data from files."""
        extra_columns = ["patient_uid", "los", "centre"]
        if self.statics_set:
            extra_columns.extend(self.statics_set)
        if self.label:
            extra_columns.extend(self.label)

        # Load ICU data
        self.data_table = parse_icu_data(
            core=self.core_path,
            meta=self.meta_path,
            feature_columns=self.features,
            extra_columns=extra_columns,
            sources=self.sources,
            include_ed=self.include_ed,
            min_length=self.min_length,
        )

    def _preprocess_data(self):
        """Apply all preprocessing steps to the raw data."""
        # Handle urine output rate calculation
        if (
            "urine_raw" in self.data_table.columns
            and "weight_static" in self.data_table.columns
        ):
            self.data_table["urine_raw"] = (
                self.data_table["urine_raw"] / self.data_table["weight_static"]
            )

        # Apply filters
        self._apply_filters()

        # Handle static features
        self._process_static_features()

        # Handle labels
        if self.label is not None:
            self.data_table[self.label] = (
                self.data_table.groupby("stay_id")[self.label]
                .apply(lambda x: x.ffill().bfill())
                .values
            )

        # Recompute lengths after filtering
        self.data_table.set_index("stay_id", inplace=True)
        self.data_table["los"] = self.data_table.groupby("stay_id").size()
        self.data_table.reset_index(inplace=True)

        # Sample data if requested
        if self.samples is not None:
            self._sample_data()

    def _apply_filters(self):
        """Apply basic filtering criteria."""
        # Filter by sources
        self.data_table = self.data_table[self.data_table.centre.isin(self.sources)]

        # Filter by length
        self.data_table = self.data_table.loc[(self.data_table.los >= self.min_length)]

        if self.ids_to_keep is not None:
            id_column = "patient_uid" if self.return_patient_ids else "stay_id"
            self.data_table = self.data_table.loc[
                self.data_table[id_column]
                .astype(int)
                .isin([int(id) for id in self.ids_to_keep])
            ]

        # Handle label-specific filtering
        if self.label is not None and len(self.label) == 1:
            self.data_table = self.data_table.dropna(subset=self.label)
            if f"{self.label[0]}_onset" in self.data_table.columns:
                self.data_table = self.data_table.loc[
                    (self.data_table[self.label[0]] == False)
                    | (self.data_table[f"{self.label[0]}_onset"] >= self.min_onset)
                ]

    def _process_static_features(self):
        """Handle static feature imputation and processing."""
        # Forward/backward fill within stays
        self.data_table[self.statics_set] = (
            self.data_table.groupby("stay_id")[self.statics_set]
            .apply(lambda x: x.ffill().bfill())
            .values
        )

        # Handle missing statics based on strategy
        if self.manage_missing_statics == "impute":
            self.data_table[self.statics_set] = self.data_table[
                self.statics_set
            ].fillna(np.nanmedian(self.data_table[self.statics_set]))
        elif self.manage_missing_statics == "drop":
            self.data_table = self.data_table.dropna(subset=self.statics_set)

    def _sample_data(self):
        """Sample a subset of stays if requested."""
        try:
            samples = np.random.choice(
                self.data_table.stay_id.unique(), self.samples, replace=False
            )
            self.data_table = self.data_table.loc[self.data_table.stay_id.isin(samples)]
        except ValueError:
            warnings.warn(
                "Number of samples exceeds unique stay IDs. Returning all samples."
            )

    def _create_tensors(self):
        """Convert processed data to ragged tensors (variable-length sequences)."""
        # Get stay metadata
        self.stay_sources = self.data_table.groupby("stay_id").centre.first().values
        self.lengths = np.minimum(
            self.data_table.groupby("stay_id").los.first().values, self.max_length
        )
        max_length = self.lengths.max().astype(int)

        if self.max_length is None:
            self.max_length = max_length

        # Create feature tensor as list of variable-length sequences
        X = self.data_table[["stay_id"] + self.features].copy()
        X["time_id"] = X.groupby("stay_id").cumcount()

        # Group by stay_id and create ragged sequences
        X_ragged = []
        times_ragged = []
        missing_masks_ragged = []

        for stay_id in X["stay_id"].unique():
            stay_data = X[X["stay_id"] == stay_id]
            stay_times = self.data_table[self.data_table["stay_id"] == stay_id][
                "stay_time"
            ].values

            # Get sequence for this stay (no padding)
            seq_len = min(len(stay_data), self.max_length)
            stay_features = stay_data[self.features].values[:seq_len].astype(float)

            # Create tensor and missing mask for this sequence
            stay_tensor = torch.tensor(stay_features)
            missing_mask = ~torch.isnan(stay_tensor)
            stay_tensor[~missing_mask] = 0.0

            # Store as individual tensors
            X_ragged.append(stay_tensor)
            times_ragged.append(torch.tensor(stay_times[:seq_len]).float())
            missing_masks_ragged.append(missing_mask)

        # Store as lists of tensors (ragged format)
        self.X = X_ragged
        self.times = times_ragged
        self.missing_mask = missing_masks_ragged

        # Create static features tensor (this remains the same)
        statics = (
            self.data_table.groupby("stay_id")
            .first()
            .reset_index()[self.statics_set]
            .values.astype(float)
        )

        # Create labels tensor
        label_columns = ["patient_uid" if self.return_patient_ids else "stay_id"]
        if self.label is not None:
            for label in self.label:
                label_columns.append(label)
                if f"{label}_onset" in self.data_table.columns:
                    label_columns.append(f"{label}_onset")

        y = (
            self.data_table.groupby("stay_id")
            .first()
            .reset_index()[label_columns]
            .fillna(0)
            .values.astype(int)
        )
        # Convert to tensors (static data and labels remain padded tensors)
        self.statics = torch.tensor(statics).float()
        self.lengths = torch.tensor(self.lengths)
        self.y = torch.tensor(y)

    def _transfer_from_loaded_dataset(self, filtered_dset):
        """Transfer attributes from a loaded dataset, converting to ragged format."""
        self.data_table = filtered_dset.data_table
        self.features = filtered_dset.features
        self.statics_set = filtered_dset.statics_set

        if hasattr(filtered_dset, "num_bins"):
            self.num_bins = int(filtered_dset.num_bins)
        elif hasattr(filtered_dset, "tokenizer") and hasattr(
            filtered_dset.tokenizer, "num_bins"
        ):
            self.num_bins = int(filtered_dset.tokenizer.num_bins)

        if hasattr(filtered_dset, "X_value_ids"):
            self.X_value_ids = [
                torch.as_tensor(x, dtype=torch.long) for x in filtered_dset.X_value_ids
            ]
            self.X = self.X_value_ids
            if hasattr(filtered_dset, "X_feature_ids"):
                self.X_feature_ids = [
                    torch.as_tensor(x, dtype=torch.long)
                    for x in filtered_dset.X_feature_ids
                ]

            statics = getattr(filtered_dset, "processed_statics", filtered_dset.statics)
            self.processed_statics = torch.as_tensor(statics, dtype=torch.long)
            self.statics = self.processed_statics
            self.lengths = torch.as_tensor(filtered_dset.lengths, dtype=torch.long)
            self.y = torch.as_tensor(filtered_dset.y)
            self.stay_sources = np.asarray(filtered_dset.stay_sources)
            self.times = [torch.as_tensor(t).float() for t in filtered_dset.times]
            if getattr(filtered_dset, "missing_mask", None) is not None:
                self.missing_mask = [
                    torch.as_tensor(m) for m in filtered_dset.missing_mask
                ]
            else:
                self.missing_mask = [
                    torch.ones_like(x, dtype=torch.bool) for x in self.X_value_ids
                ]

            self._truncate_direct_loaded_sequences()

            if hasattr(filtered_dset, "X_FIDDLE_bins"):
                self.X_FIDDLE_bins = filtered_dset.X_FIDDLE_bins
            if hasattr(filtered_dset, "S_FIDDLE_bins"):
                self.S_FIDDLE_bins = filtered_dset.S_FIDDLE_bins
            self.static_names = getattr(filtered_dset, "static_names", self.statics_set)
            self.static_missing_token_id = getattr(
                filtered_dset, "static_missing_token_id", self.num_bins
            )
            self.static_pad_token_id = getattr(
                filtered_dset, "static_pad_token_id", self.num_bins + 1
            )
            self.is_direct_fiddle = True
            return

        # Convert loaded tensors to ragged format
        if hasattr(filtered_dset.X, "to_dense"):
            # Handle sparse tensors
            X_dense = filtered_dset.X.to_dense().to(dtype=torch.int32)
        else:
            X_dense = filtered_dset.X

        # Convert padded tensor to ragged lists
        self.X = []
        self.times = []
        self.missing_mask = []

        for i in range(X_dense.shape[0]):
            # Get actual length for this sequence
            seq_len = min(filtered_dset.lengths[i].item(), X_dense.shape[1])

            # Extract unpadded sequence
            x_seq = X_dense[i, :seq_len]
            times_seq = filtered_dset.times[i]
            mask_seq = (
                filtered_dset.missing_mask[i]
                if hasattr(filtered_dset, "missing_mask")
                else torch.ones_like(x_seq, dtype=torch.bool)
            )

            self.X.append(x_seq)
            self.times.append(times_seq)
            self.missing_mask.append(mask_seq)

        # Keep other attributes as-is
        self.statics = filtered_dset.statics
        self.lengths = filtered_dset.lengths
        self.stay_sources = filtered_dset.stay_sources

        # FIDDLE-specific attributes
        if hasattr(filtered_dset, "X_names"):
            self.X_names = filtered_dset.X_names
        if hasattr(filtered_dset, "X_FIDDLE_bins"):
            self.X_FIDDLE_bins = filtered_dset.X_FIDDLE_bins
        if hasattr(filtered_dset, "S_FIDDLE_bins"):
            self.S_FIDDLE_bins = filtered_dset.S_FIDDLE_bins
        if hasattr(filtered_dset, "static_names"):
            self.static_names = filtered_dset.static_names

        # Create labels tensor
        label_columns = ["patient_uid" if self.return_patient_ids else "stay_id"]
        if self.label is not None:
            for label in self.label:
                label_columns.append(label)
                if f"{label}_onset" in self.data_table.columns:
                    label_columns.append(f"{label}_onset")

        y = (
            self.data_table.groupby("stay_id")
            .first()
            .reset_index()[label_columns]
            .fillna(0)
            .values.astype(int)
        )
        self.y = torch.tensor(y)

    def _demographic_bins(self):
        """
        Demographic bins for the tokenizer, taken from the dataset.

        Direct FIDDLE datasets store each static as a value id: 0..num_bins-1
        for the binned statics, then static_missing_token_id and
        static_pad_token_id. female_static is binary. A 5-bin dataset therefore
        gives the historical 2/5/5/5. Other datasets keep the tokenizer defaults.
        """
        if not (
            getattr(self, "is_direct_fiddle", False) or hasattr(self, "X_value_ids")
        ):
            return None
        return {
            "female_static": 2,
            "age_static": int(self.num_bins),
            "height_static": int(self.num_bins),
            "weight_static": int(self.num_bins),
        }

    def _check_static_levels(self, statics_indices):
        """
        Fail if a non-missing static value has no demographic bin. Such values
        used to be clamped to the last bin by the tokenizer and imputed as
        missing by the one-hot.
        """
        values = torch.as_tensor(statics_indices).long()
        missing_ids = {
            int(getattr(self, "static_missing_token_id", self.num_bins)),
            int(getattr(self, "static_pad_token_id", self.num_bins + 1)),
        }
        for i, name in enumerate(self.static_names):
            n_bins = self.tokenizer.demographic_bins[name]
            col = values[:, i]
            is_missing = torch.zeros_like(col, dtype=torch.bool)
            for missing_id in missing_ids:
                is_missing |= col == missing_id
            out_of_range = (~is_missing) & ((col < 0) | (col >= n_bins))
            if out_of_range.any():
                bad = sorted(torch.unique(col[out_of_range]).tolist())
                raise ValueError(
                    f"{name}: values {bad} are not missing/pad ids {sorted(missing_ids)} "
                    f"and have no demographic bin (n_bins={n_bins})."
                )

    def _direct_statics_to_onehot(self, statics_indices):
        """
        Convert direct statics indices (N, 4) back to one-hot (N, sum of bins).
        Order and bin sizes inferred from static_names + tokenizer.demographic_bins.
        """
        demographic_bins = self.tokenizer.demographic_bins
        # e.g. {"female_static": 2, "age_static": 5, "height_static": 5, "weight_static": 5}

        one_hot_parts = []
        for i, name in enumerate(self.static_names):
            n_bins = demographic_bins[name]
            oh = torch.zeros(statics_indices.shape[0], n_bins, dtype=torch.float)
            idx = statics_indices[:, i].long()

            # Impute missing values to median bin (idx >= n_bins o idx < 0)
            median_bin = n_bins // 2
            is_missing = (idx < 0) | (idx >= n_bins)
            idx = idx.clone()
            idx[is_missing] = median_bin

            oh.scatter_(1, idx.unsqueeze(1), 1.0)
            one_hot_parts.append(oh)

        return torch.cat(one_hot_parts, dim=1)  # (N, sum of bins), 17 for 2/5/5/5

    def _truncate_direct_loaded_sequences(self):
        """Apply max_length consistently to saved direct FIDDLE sequences."""
        if self.max_length is None or not hasattr(self, "X_value_ids"):
            return

        limit = int(self.max_length)
        if limit <= 0:
            raise ValueError(
                f"max_length must be positive or None, got {self.max_length}"
            )

        self.X_value_ids = [x[:limit] for x in self.X_value_ids]
        self.X = self.X_value_ids
        if hasattr(self, "X_feature_ids"):
            self.X_feature_ids = [x[:limit] for x in self.X_feature_ids]
        if hasattr(self, "times"):
            self.times = [t[:limit] for t in self.times]
        if hasattr(self, "missing_mask") and self.missing_mask is not None:
            self.missing_mask = [m[:limit] for m in self.missing_mask]
        if hasattr(self, "lengths"):
            self.lengths = torch.clamp(self.lengths, max=limit)

    def _run_fiddle_processing(self):
        """Run FIDDLE processing pipeline."""

        # Store original get_item_mode and switch temporarily
        original_mode = self.get_item_mode
        self.get_item_mode = "tuples"

        if run_fiddle is None:
            raise ImportError(
                "FIDDLE preprocessing dependencies are not available; "
                "load a preprocessed pickle or install the full preprocessing stack."
            )

        # Run FIDDLE processing
        (
            self.X,
            self.statics,
            self.X_names,
            self.static_names,
            self.X_FIDDLE_bins,
            self.S_FIDDLE_bins,
        ) = run_fiddle(
            self,
            X_discretization_bins=self.X_FIDDLE_bins,
            S_discretization_bins=self.S_FIDDLE_bins,
            verbose=self.verbose,
        )

        # Convert back to tensors
        self.X = torch.tensor(self.X)
        self.statics = torch.tensor(self.statics)

        # Restore mode
        self.get_item_mode = original_mode

    def _resolve_truncation_for_centre(self, centre_name):
        """Resolve truncation_time (int or per-centre dict/DictConfig) for one centre."""
        truncation_time = self.truncation_time
        if isinstance(truncation_time, (dict, DictConfig)):
            if isinstance(truncation_time, DictConfig):
                truncation_time = OmegaConf.to_container(truncation_time, resolve=True)
            return truncation_time.get(
                str(centre_name).lower(), max(truncation_time.values())
            )
        return truncation_time

    def _chunk_long_sequences(self, overlap_ratio: float = 0.15):
        """
        Split stays whose linearized token sequence exceeds the truncation cap
        into multiple training chunks, so the full stay contributes to
        training instead of being randomly windowed/truncated.

        Chunk 0:  [SOS, demo tokens..., GAP] + timepoint blocks... (+ EOS if this
                   is also the last chunk)
        Chunk k>0: [overlap: whole trailing timepoint blocks from chunk k-1,
                    NO <SOS>, NO demographics] + new timepoint blocks...
                   (+ EOS if this is the last chunk)

        The absence of <SOS> at the start of a continuation chunk is itself
        the signal that this is NOT the start of a new stay — the model never
        sees this token pattern (raw timepoint content with no SOS) except as
        a continuation, so it learns to treat it accordingly. No new special
        token is introduced.

        Conditioning (class_embed/center_embed) is NOT part of the token
        sequence — it's added per-sample in the model's forward() from
        y/stay_sources — so it automatically applies to every chunk, matching
        what causal attention to the prefix would see during unchunked
        generation. Demographics (which ARE tokens) are only ever in chunk 0.

        time_positions retain their true absolute value across chunks (never
        reset to 0), preserving meaningful sinusoidal time embeddings.

        Metadata (y, stay_sources, statics, times, lengths) is duplicated per
        chunk. (centre, stay_id) is identical across a stay's chunks, keeping
        them in the same train/val/test split without touching _split().
        """
        if self.truncation_time is None or not hasattr(self, "X_tokens"):
            return

        gap_id = self.tokenizer.vocab["<GAP>"]
        eos_id = self.tokenizer.vocab["<EOS>"]
        prefix_len = (
            1 + len(self.tokenizer.demographic_features) + 1
            if self.include_demographics
            else 1
        )

        has_time_positions = (
            hasattr(self, "X_time_positions") and self.X_time_positions is not None
        )
        has_statics = hasattr(self, "statics") and self.statics is not None
        has_times = hasattr(self, "times") and self.times is not None

        new_X_tokens = []
        new_X_time_positions = [] if has_time_positions else None
        new_stay_sources = []
        new_y_rows = []
        new_statics_rows = [] if has_statics else None
        new_times = [] if has_times else None

        y_is_tensor = torch.is_tensor(self.y)
        y_source = self.y.numpy() if y_is_tensor else np.asarray(self.y)

        for i, seq in enumerate(self.X_tokens):
            centre = self.stay_sources[i]
            cap = self._resolve_truncation_for_centre(centre)
            tp = self.X_time_positions[i] if has_time_positions else None
            seq_len = seq.size(0)

            if cap is None or seq_len <= cap:
                new_X_tokens.append(seq)
                if has_time_positions:
                    new_X_time_positions.append(tp)
                new_stay_sources.append(centre)
                new_y_rows.append(y_source[i])
                if has_statics:
                    new_statics_rows.append(self.statics[i])
                if has_times:
                    new_times.append(self.times[i])
                continue

            seq_np = seq.numpy()
            timeseries_start = prefix_len
            body = seq_np[timeseries_start:-1]  # exclude prefix and final EOS
            body_tp = tp[timeseries_start:-1].numpy() if has_time_positions else None

            gap_positions = np.where(body == gap_id)[0]
            blocks, block_tps = [], []
            start = 0
            for gp in gap_positions:
                blocks.append(body[start : gp + 1].tolist())
                if has_time_positions:
                    block_tps.append(body_tp[start : gp + 1].tolist())
                start = gp + 1
            if start < len(body):
                blocks.append(body[start:].tolist())
                if has_time_positions:
                    block_tps.append(body_tp[start:].tolist())

            prefix_tokens = seq_np[:timeseries_start].tolist()
            prefix_tp = tp[:timeseries_start].tolist() if has_time_positions else None

            target_overlap_len = int(round(cap * overlap_ratio))

            chunks_tokens, chunks_tp = [], []
            current_chunk = list(prefix_tokens)  # chunk 0 starts with SOS+demo+GAP
            current_tp = list(prefix_tp) if has_time_positions else None
            chunk_block_start = 0
            is_first_chunk = True

            def flush_and_start_next(next_block_idx, is_last):
                nonlocal current_chunk, current_tp, chunk_block_start, is_first_chunk

                closed = list(current_chunk)
                closed_tp = list(current_tp) if has_time_positions else None
                if is_last:
                    closed.append(eos_id)
                    if has_time_positions:
                        closed_tp.append(closed_tp[-1] if closed_tp else 0)
                chunks_tokens.append(closed)
                if has_time_positions:
                    chunks_tp.append(closed_tp)

                if is_last:
                    return

                overlap_block_idxs = []
                acc_len = 0
                b = next_block_idx - 1
                while b >= chunk_block_start and acc_len < target_overlap_len:
                    overlap_block_idxs.insert(0, b)
                    acc_len += len(blocks[b])
                    b -= 1

                # NO <SOS>, no demographics — start directly with overlap content.
                current_chunk = []
                current_tp = [] if has_time_positions else None
                for ob in overlap_block_idxs:
                    current_chunk.extend(blocks[ob])
                    if has_time_positions:
                        current_tp.extend(block_tps[ob])
                chunk_block_start = next_block_idx
                is_first_chunk = False

            b_idx = 0
            while b_idx < len(blocks):
                block = blocks[b_idx]
                is_last_block = b_idx == len(blocks) - 1
                projected_len = (
                    len(current_chunk) + len(block) + (1 if is_last_block else 0)
                )

                has_own_content = b_idx > chunk_block_start or len(current_chunk) > (
                    prefix_len if is_first_chunk else 0
                )
                if has_own_content and projected_len > cap:
                    flush_and_start_next(next_block_idx=b_idx, is_last=False)
                    continue

                min_seed_len = prefix_len if is_first_chunk else 0
                if (
                    len(current_chunk) + len(block) + 1 > cap
                    and len(current_chunk) <= min_seed_len
                ):
                    # Safety fallback: a single timepoint block bigger than cap alone.
                    block = block[: max(0, cap - len(current_chunk) - 1)]
                    if has_time_positions:
                        block_tps[b_idx] = block_tps[b_idx][: len(block)]

                current_chunk.extend(block)
                if has_time_positions:
                    current_tp.extend(block_tps[b_idx])
                b_idx += 1

            flush_and_start_next(next_block_idx=len(blocks), is_last=True)

            for chunk, chunk_tp in zip(
                chunks_tokens,
                chunks_tp if has_time_positions else [None] * len(chunks_tokens),
            ):
                new_X_tokens.append(torch.tensor(chunk, dtype=torch.long))
                if has_time_positions:
                    new_X_time_positions.append(
                        torch.tensor(chunk_tp, dtype=torch.long)
                    )
                new_stay_sources.append(centre)
                new_y_rows.append(y_source[i])
                if has_statics:
                    new_statics_rows.append(self.statics[i])
                if has_times:
                    new_times.append(self.times[i])

        self.X_tokens = new_X_tokens
        self.X = self.X_tokens
        if has_time_positions:
            self.X_time_positions = new_X_time_positions
        self.stay_sources = np.asarray(new_stay_sources)
        self.y = (
            torch.tensor(np.stack(new_y_rows), dtype=self.y.dtype)
            if y_is_tensor
            else np.stack(new_y_rows)
        )
        if has_statics:
            self.statics = torch.stack(new_statics_rows)
        if has_times:
            self.times = new_times
        if hasattr(self, "lengths"):
            self.lengths = torch.tensor(
                [t.size(0) for t in self.X_tokens], dtype=torch.long
            )

    def tokenize(self, split="", fine_tune=True, num_bins=5):
        """
        Tokenize the dataset for transformer-based models using linear sequences.

        Args:
            split: Split identifier (for compatibility)
            fine_tune: Whether in fine-tuning mode
            num_bins: Number of bins for FIDDLE tokenization
        """
        self.tokenize_data = True
        self.fine_tune = fine_tune

        # Create tokenizer with demographics support
        self.tokenizer = FIDDLETokenizer(
            features=self.features,
            num_bins=self.num_bins,
            shuffle_tokens_within_timepoint=self.shuffle_tokens_within_timepoint,
            add_time_point_positional_embeddings=self.add_time_point_positional_embeddings,
            label=self.return_label,
            include_demographics=self.include_demographics,
            demographic_bins=self._demographic_bins(),
        )

        if not hasattr(self, "X_value_ids"):
            raise ValueError(
                "Tokenization now requires direct ragged FIDDLE value IDs "
                "(`X_value_ids`). Regenerate or load a direct FIDDLE dataset; "
                "dense one-hot FIDDLE tokenization is no longer an active fallback."
            )

        self._tokenize_direct_fiddle()

    def _tokenize_direct_fiddle(self):
        """Linearize direct ragged FIDDLE value IDs without dense one-hot expansion."""
        self.X_tokens = []
        self.X_time_positions = (
            [] if self.add_time_point_positional_embeddings else None
        )

        statics = (
            self.processed_statics
            if getattr(self, "processed_statics", None) is not None
            else self.statics
        )
        if (
            self.include_demographics
            and getattr(self, "processed_statics", None) is not None
        ):
            self._check_static_levels(self.processed_statics)

        tokenized_data = Parallel(n_jobs=self.num_workers, backend="threading")(
            delayed(self.tokenizer.encode_direct_sequence)(
                x,
                statics[x_idx] if self.include_demographics else None,
                self.statics_set,
            )
            for x_idx, x in tqdm(
                enumerate(self.X_value_ids),
                total=len(self.X_value_ids),
                desc="Linearizing direct FIDDLE tokens",
            )
        )

        for result in tokenized_data:
            if self.add_time_point_positional_embeddings:
                token_sequence, time_positions = result
                self.X_tokens.append(torch.tensor(token_sequence, dtype=torch.long))
                self.X_time_positions.append(
                    torch.tensor(time_positions, dtype=torch.long)
                )
            else:
                self.X_tokens.append(torch.tensor(result, dtype=torch.long))

        if getattr(self, "processed_statics", None) is not None:
            self.statics = self._direct_statics_to_onehot(self.processed_statics)

    def impute_FIDDLE_statics(self, statics_vector):
        """Impute missing continuous statics in FIDDLE setting."""
        static_ids = torch.tensor(
            [
                next(
                    i
                    for i, feature_type in enumerate(self.statics_set)
                    if feature_type in name
                )
                for name in self.static_names
            ]
        )

        for i in set(static_ids):
            if (
                len(statics_vector[static_ids == i]) == 5
                and sum(statics_vector[static_ids == i]) == 0
            ):
                statics_vector[torch.argmax((static_ids == i).int()) + 2] = 1

        return statics_vector

    def __len__(self):
        """Return number of samples in dataset."""
        if isinstance(self.X, list):
            return len(self.X)
        else:
            return self.X.shape[0]

    def __repr__(self):
        """String representation of the dataset."""
        object_str = (
            f"ICU Dataset | {len(self)} samples | sources: {'-'.join(self.sources)}"
        )
        if self.label is not None:
            object_str += f" | outcomes: {'-'.join(self.label)}"
        return object_str

    def _get_tuple_item(self, idx):
        """Get tuple format item."""
        return (
            format_set(self.X[idx], self.missing_mask[idx], self.times[idx]),
            self.statics[idx],
            self.y[idx],
        )

    def retrieve_labels_from_id(self, stay_id, label, source=None):
        if hasattr(stay_id, "numel"):
            stay_id = (
                int(stay_id[0].item()) if stay_id.numel() > 1 else int(stay_id.item())
            )
        elif hasattr(stay_id, "__len__") and len(stay_id) > 1:
            stay_id = int(
                stay_id[0].item() if hasattr(stay_id[0], "item") else stay_id[0]
            )
        elif hasattr(stay_id, "item"):
            stay_id = int(stay_id.item())
        else:
            stay_id = int(stay_id)

        try:
            if source is not None and isinstance(self.data_table.index, pd.MultiIndex):
                row = self.data_table.loc[(self._normalize_source(source), stay_id)]
            else:
                row = self.data_table.loc[stay_id]
        except KeyError:
            raise KeyError(f"No metadata found for source={source}, stay_id={stay_id}")

        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        label_cols = [col for col in self.data_table.columns if label in col]
        label_vals = row[label_cols].values

        try:
            if np.isnan(label_vals[1]):
                label_vals[1] = 0.0
        except IndexError:
            label_vals = np.concatenate([label_vals, [0.0]])

        return np.concatenate([[stay_id], label_vals])

    def _item_label(self, idx):
        """Label row of sample idx with its centre appended (float32).

        With return_label set, row idx of y_full is retrieve_labels_from_id for
        sample idx, cast to float32 at setup (_build_label_table), so no
        metadata lookup runs per item.
        """
        label = self.y[idx]
        if self.return_label is not None:
            label = self.y_full[idx]
        centre = torch.tensor(
            CENTRE_MAP.get(str(self.stay_sources[idx]).lower(), 0), dtype=torch.long
        )
        return torch.cat([label, centre.unsqueeze(0).float()])

    def __getitem__(self, idx):
        if self.get_item_mode == "tuples":
            return self._get_tuple_item(idx)

        # If tokenized, return tokenized data
        if hasattr(self, "X_tokens"):
            tokens = self.X_tokens[
                idx
            ].clone()  # Make a copy to avoid modifying stored data
            time_positions = None

            if self.add_time_point_positional_embeddings and hasattr(
                self, "X_time_positions"
            ):
                time_positions = self.X_time_positions[idx].clone()

            label = self._item_label(idx)  # [stay_id, outcome, onset, centre_id]

            # # Truncate randomly if needed
            # if (
            #     self.truncation_time is not None
            #     and tokens.size(0) > self.truncation_time
            # ):
            #     max_start = max(0, tokens.size(0) - self.truncation_time)
            #     start_idx = np.random.randint(0, max_start + 1)
            #     end_idx = start_idx + self.truncation_time
            #     tokens = torch.cat(
            #         [tokens[0].unsqueeze(dim=0), tokens[start_idx:end_idx]]
            #     )
            #     if time_positions is not None:
            #         time_positions = torch.cat(
            #             [
            #                 time_positions[0].unsqueeze(dim=0),
            #                 time_positions[start_idx:end_idx],
            #             ]
            #         )
            #     else:
            #         time_positions = None

            if self.add_time_point_positional_embeddings and hasattr(
                self, "X_time_positions"
            ):
                return (
                    tokens,  # Linear token sequence (possibly shuffled)
                    time_positions,  # Time position sequence (shuffled accordingly)
                    self.statics[idx],
                    self.times[idx],  # Already variable length
                    label,
                )
            else:
                return (
                    tokens,  # Linear token sequence (possibly shuffled)
                    self.statics[idx],
                    self.times[idx],  # Already variable length
                    label,
                )

        return (
            self.X[idx],  # Already variable length
            self.statics[idx],
            self.times[idx],  # Already variable length
            label,
        )


class ICUDatamodule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        pretrain: bool = False,
        val_fraction: float = 0.05,
        test_fraction: float = 0.2,
        batch_size: int = 32,
        tokenize: bool = False,
        num_workers: int = 7,
        pin_memory: bool = True,
        drop_last: bool = False,
        truncation_time: int = 1300,
        random_state: int = 42,
        verbose=False,
        shuffle_tokens_within_timepoint: bool = False,
        add_time_point_positional_embeddings: bool = False,
        return_label="death_icu",
        demo=False,
        demo_fraction: float = 0.1,
        p_shuffle: float = 0.3,
        min_length: Optional[int] = 6,
        max_length: Optional[int] = 196,
        max_gap: int = 6,
        samples: Optional[int] = None,
        ids_to_keep: Optional[List] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.pretrain = pretrain
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.tokenize = tokenize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.truncation_time = truncation_time
        self.random_state = random_state
        self.verbose = verbose
        self.shuffle_tokens_within_timepoint = shuffle_tokens_within_timepoint
        self.add_time_point_positional_embeddings = add_time_point_positional_embeddings
        self.return_label = return_label
        self.demo = demo
        self.demo_fraction = demo_fraction
        self.p_shuffle = p_shuffle
        self.min_length = min_length
        self.max_length = max_length
        self.max_gap = max_gap
        self.samples = samples
        self.ids_to_keep = ids_to_keep

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._setup()

    @property
    def get_rep(self):
        object_str = f"ICU DataModule with {len(self.base_icu_dataset)} samples"
        return object_str

    def __str__(self):
        return self.get_rep

    def __repr__(self):
        return self.get_rep

    def _split(self):
        all_ids = self.base_icu_dataset.y[:, 0].numpy()
        all_sources = np.asarray(self.base_icu_dataset.stay_sources)
        all_keys = np.asarray(
            [
                f"{str(source).lower()}\x1f{int(stay_id)}"
                for source, stay_id in zip(all_sources, all_ids)
            ]
        )
        unique_keys = np.unique(all_keys)

        # Create splits
        train_val_keys, test_keys = train_test_split(
            unique_keys,
            test_size=self.test_fraction,
            random_state=getattr(self, "random_state", 42),
        )
        train_keys, val_keys = train_test_split(
            train_val_keys,
            test_size=self.val_fraction / (1 - self.test_fraction),
            random_state=getattr(self, "random_state", 42),
        )

        # Get indices for each split
        self.train_indices = np.where(np.isin(all_keys, train_keys))[0]
        self.val_indices = np.where(np.isin(all_keys, val_keys))[0]
        self.test_indices = np.where(np.isin(all_keys, test_keys))[0]

        self.train_ids = all_ids[self.train_indices]
        self.val_ids = all_ids[self.val_indices]
        self.test_ids = all_ids[self.test_indices]

        # Create subsets
        self.train_icu_dataset = Subset(self.base_icu_dataset, self.train_indices)
        self.val_icu_dataset = Subset(self.base_icu_dataset, self.val_indices)
        self.test_icu_dataset = Subset(self.base_icu_dataset, self.test_indices)

    def _setup(self):
        self.base_icu_dataset = ICUDataset(
            pkl_path=self.data_path,
            sources=getattr(
                self,
                "sources",
                ["eicu", "miiv", "aumc", "mimic", "hirid", "sic"],
            ),
            feature_set=getattr(self, "feature_set", "all"),
            statics_set=getattr(
                self,
                "statics_set",
                ["female_static", "age_static", "height_static", "weight_static"],
            ),
            num_workers=self.num_workers,
            shuffle_tokens_within_timepoint=self.shuffle_tokens_within_timepoint,
            add_time_point_positional_embeddings=self.add_time_point_positional_embeddings,
            return_label=self.return_label,
            truncation_time=self.truncation_time,
            min_length=self.min_length,
            max_length=self.max_length,
            max_gap=self.max_gap,
            samples=self.samples,
            ids_to_keep=self.ids_to_keep,
            demo=self.demo,
            demo_fraction=self.demo_fraction,
            verbose=self.verbose,
            include_demographics=getattr(self, "include_demographics", True),
        )
        self.tokenizer = self.base_icu_dataset.tokenizer
        self._split()

    def train_dataloader(self):
        return DataLoader(
            self.train_icu_dataset,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                pad_collate,
                vocabulary=self.base_icu_dataset.tokenizer.vocab,
                label=self.return_label,
                is_train=True,
                shuffle_tokens_within_timepoint=self.shuffle_tokens_within_timepoint,
                p_shuffle=self.p_shuffle,
                include_demographics=self.base_icu_dataset.include_demographics,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_icu_dataset,
            drop_last=self.drop_last,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                pad_collate,
                vocabulary=self.base_icu_dataset.tokenizer.vocab,
                label=self.return_label,
                include_demographics=self.base_icu_dataset.include_demographics,
                is_train=False,
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_icu_dataset,
            drop_last=self.drop_last,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                pad_collate,
                vocabulary=self.base_icu_dataset.tokenizer.vocab,
                label=self.return_label,
                include_demographics=self.base_icu_dataset.include_demographics,
                is_train=False,
            ),
        )


class EarlyWarningDataset(ICUDataset, torch.utils.data.Dataset):
    """
    Early Warning Dataset that inherits from ICUDataset and specializes in
    early warning system preprocessing with sliding windows and lookback filtering.

    This class focuses purely on early warning specific transformations while
    inheriting all ICU data processing capabilities from the parent class.
    """

    def __init__(
        self,
        label: str,
        feature_set: str = "all",
        lookback: Union[int, Tuple[int, int]] = 0,
        syn_lookback: Optional[int] = None,
        binarize_outcome: bool = True,
        # Sliding window parameters
        window_size: int = None,
        include_missing_mask: bool = False,
        missingness_threshold: Union[str, float] = "all",
        min_length: int = 6,
        real_path: str = None,
        syn_path: str = None,
        **kwargs,
    ):
        """
        Initialize Early Warning Dataset.

        Args:
            label: Single outcome label for early warning (required)
            lookback: Hours before onset after which data is discarded.
                - sliding_window: int only. Every stay, in all splits, is cut at
                  lookback hours before its onset (positives) or last observed
                  hour (negatives); stays left with fewer than window_size hours
                  produce no windows and drop out.
                - full sequences: int or (min, max) tuple for random sampling,
                  applied to positives in __getitem__ of each subset.
            syn_lookback: Lookback for synthetic stays in the training split only
                (sliding_window only). Synthetic stays are
                built with min(lookback, syn_lookback) and
                EarlyWarningDatamodule drops the extra windows per split.
            window_size: Size of windows for sliding_window mode (defaults to 4)
            include_missing_mask: If True, include missingness indicators (sliding_window only)
            missingness_threshold: Threshold for filtering missingness indicators (sliding_window only)
                                  - float: Remove indicators with missingness > threshold
                                  - str: Remove indicators for specified feature set (e.g., "vitals")
            min_length: Minimum length of stay to include
            real_path: Path to real dataset file
            syn_path: Path to synthetic dataset file
            **kwargs: All ICUDataset parameters (core_path, sources, etc.)
        """
        # Validate early warning specific parameters
        assert isinstance(
            label, str
        ), "A single label should be provided as outcome for the early warning task."

        # Validate lookback parameter
        if isinstance(lookback, tuple):
            assert (
                len(lookback) == 2
            ), "Lookback tuple must have exactly 2 elements (min, max)."
            assert (
                lookback[0] >= 0 and lookback[1] >= 0
            ), "Lookback values cannot be negative."
            assert lookback[0] <= lookback[1], "Lookback minimum must be <= maximum."
        else:
            assert lookback >= 0, "Lookback cannot be negative."

        # Validate missingness parameters
        self.use_sliding_windows = window_size is not None
        if self.use_sliding_windows:
            assert window_size > 0, "Window size must be positive."
            assert not isinstance(lookback, tuple), (
                "Sliding windows cut every stay at a single lookback; "
                "a (min, max) range is only supported for full sequences."
            )
        assert syn_lookback is None or (
            self.use_sliding_windows and syn_lookback >= 0
        ), "syn_lookback must be non-negative and needs sliding windows."
        if not self.use_sliding_windows and include_missing_mask:
            warnings.warn(
                "Missingness mask is only relevant for sliding_window mode. Ignoring."
            )

        # Store early warning specific parameters first
        self.label = label
        self.feature_set = feature_set
        self.binarize_outcome = binarize_outcome

        # Handle lookback parameter - store original and set working value
        self._original_lookback = lookback  # Store original (could be tuple)
        if isinstance(lookback, tuple):
            # During initialization, use the minimum value to avoid issues
            self.lookback = lookback[0]
        else:
            self.lookback = lookback
        self.syn_lookback = self.lookback if syn_lookback is None else syn_lookback

        self.min_length = min_length
        self.binarize_outcome = binarize_outcome
        self.window_size = window_size
        self.impute_missing = kwargs.get("impute_missing", "forwardfill")

        # Store missingness parameters (only relevant for sliding_window)
        self.include_missing_mask = (
            include_missing_mask if self.use_sliding_windows else False
        )
        self.missingness_threshold = (
            missingness_threshold if self.use_sliding_windows else None
        )

        # For random lookback augmentation (when lookback is tuple)
        self._is_random_lookback = isinstance(lookback, tuple)

        # For synthetic data
        if syn_path and not real_path:
            # with real data the bin count comes from it; synthetic tokens alone
            # only show the bins that happened to be sampled, so it must be given
            if "num_bins" not in kwargs:
                raise ValueError(
                    "synthetic-only data needs num_bins (the K the model was trained on)"
                )
            self.num_bins = int(kwargs["num_bins"])
            self.synthetic = True
            self.return_label = label
            if isinstance(self.feature_set, str):
                self.features = sorted(get_feature_set(self.feature_set))
            else:
                self.features = sorted(self.feature_set)
            print(f"[EarlyWarningDataset] Loading synthetic-only data from: {syn_path}")
            self._load_synthetic_data(syn_path)
            self._prepare_setup()

        if real_path and not syn_path:
            # Prepare default kwargs for parent class initialization
            self.synthetic = False
            kwargs["run_fiddle"] = True
            kwargs["get_item_mode"] = "FIDDLE"
            kwargs["tokenize_data"] = not self.use_sliding_windows
            kwargs["pkl_path"] = real_path
            kwargs["return_label"] = label

            print(f"[EarlyWarningDataset] Loading real-only data from: {real_path}")
            # Initialize parent ICUDataset with basic processing only
            super().__init__(
                label=label, feature_set=feature_set, min_length=min_length, **kwargs
            )
            print(
                f"[EarlyWarningDataset] After super().__init__: self.X type={type(getattr(self, 'X', None))}"
            )
            if hasattr(self, "X") and hasattr(self.X, "shape"):
                print(
                    f"[EarlyWarningDataset] After super().__init__: self.X.shape={self.X.shape}"
                )
            self._prepare_setup()

        if real_path and syn_path:
            print(
                f"[EarlyWarningDataset] Loading real+syn data from: real={real_path}, syn={syn_path}"
            )

            self.synthetic = False
            kwargs["run_fiddle"] = True
            kwargs["get_item_mode"] = "FIDDLE"
            kwargs["tokenize_data"] = not self.use_sliding_windows
            kwargs["pkl_path"] = real_path
            kwargs["return_label"] = label
            super().__init__(
                label=label, feature_set=feature_set, min_length=min_length, **kwargs
            )
            self._prepare_setup()

            if isinstance(self.feature_set, str):
                requested_features = sorted(get_feature_set(self.feature_set))
            else:
                requested_features = sorted(self.feature_set)

            if set(requested_features) != set(self.features):
                all_features_sorted = sorted(self.features)
                feature_indices = [
                    all_features_sorted.index(f) for f in requested_features
                ]

                # window_tensor shape: (window_size, n_features) → filter dim 1
                self.window_tensor = [w[:, feature_indices] for w in self.window_tensor]

                self.features = requested_features
                self._compute_sliding_window_missingness()
                self._apply_missingness_threshold_filter()
                self._update_feature_names_with_missingness()

            real_ids = self.ids.clone()
            real_X = (
                self.window_tensor.copy() if hasattr(self, "window_tensor") else None
            )
            real_statics = self.statics.clone()
            real_y = self.y.clone()
            real_stay_sources = pd.Series(self.stay_sources.copy())
            # Only exists in sliding_window mode (see
            # _setup_sliding_window_sequences); merge it the same way as
            # stay_sources itself below.
            real_patient_stay_sources = (
                pd.Series(np.asarray(self._patient_stay_sources)).copy()
                if hasattr(self, "_patient_stay_sources")
                else None
            )
            real_times = self.times.clone()
            real_onset = self.onset.clone()
            real_reference_hours = self.reference_hours.clone()
            real_lengths = self.lengths.clone()
            real_y_full = self.y_full.clone()
            # Save real missingness tensor before it gets reset
            real_missingness = (
                list(self.missingness_tensor)
                if self.missingness_tensor is not None
                else None
            )

            self.synthetic = True
            self.return_label = label
            if isinstance(self.feature_set, str):
                self.features = sorted(get_feature_set(self.feature_set))
            else:
                self.features = sorted(self.feature_set)
            self.is_direct_fiddle = False
            if hasattr(self, "X_value_ids"):
                del self.X_value_ids

            self._load_synthetic_data(syn_path)
            self._prepare_setup()

            self.ids = torch.cat([torch.tensor(real_ids), self.ids])
            self.window_tensor = (
                real_X + self.window_tensor
                if isinstance(real_X, list)
                else torch.cat([real_X, self.window_tensor])
            )
            self.statics = torch.cat([real_statics, self.statics])
            self.y = torch.cat([real_y, self.y])
            self.stay_sources = pd.concat(
                [real_stay_sources, self.stay_sources]
            ).reset_index(drop=True)
            if real_patient_stay_sources is not None and hasattr(
                self, "_patient_stay_sources"
            ):
                self._patient_stay_sources = pd.concat(
                    [
                        real_patient_stay_sources,
                        pd.Series(np.asarray(self._patient_stay_sources)),
                    ]
                ).reset_index(drop=True)
            self.times = torch.cat([real_times, self.times])
            self.onset = torch.cat([real_onset, self.onset])
            self.reference_hours = torch.cat(
                [real_reference_hours, self.reference_hours]
            )
            self.lengths = torch.cat([real_lengths, self.lengths])
            self.y_full = torch.cat([real_y_full, self.y_full])
            # Concatenate missingness tensors: real first, then synthetic
            if real_missingness is not None and self.missingness_tensor is not None:
                self.missingness_tensor = real_missingness + list(
                    self.missingness_tensor
                )
            elif real_missingness is not None:
                self.missingness_tensor = real_missingness
            # else: keep synthetic-only missingness_tensor as is

    def _prepare_setup(self):
        # Set flags to indicate we're in early warning mode
        self.fine_tune = True
        self._early_warning_setup_complete = False

        # Initialize missingness-related attributes
        self.missingness_tensor = None
        self.feature_names_with_missingness = None

        # Run early warning specific setup
        self._setup_early_warning()

        # After setup is complete, handle tuple lookback if needed
        if isinstance(self._original_lookback, tuple):
            self.set_lookback(self._original_lookback)

        # Mark setup as complete
        self._early_warning_setup_complete = True

    def _map_tok_to_idx(self, pat_tokens, special_token=["<GAP>", "<EOS>"]):
        # the synthetic tokens use the same K as the real data they were trained on
        n_cols = len(self.features) * self.num_bins
        gap_tok, eos_tok = special_token
        feature_col = {f: i * self.num_bins for i, f in enumerate(self.features)}
        patients_matrix = []
        lengths = []

        for tokens in pat_tokens:
            tokens_np = np.array(tokens, dtype=object)
            eos_positions = np.where(tokens_np == eos_tok)[0]
            if len(eos_positions) > 0:
                eos_pos = eos_positions[0]
                tokens_np = tokens_np[:eos_pos]  # cut before EOS

            gap_positions = np.where(tokens_np == gap_tok)[0]

            # Separe tokens by timesteps
            timesteps = []
            start = 0
            for gap_pos in gap_positions:
                if start < gap_pos:
                    timesteps.append(tokens_np[start:gap_pos])
                start = gap_pos + 1
            if start < len(tokens_np):
                timesteps.append(tokens_np[start:])

            if len(gap_positions) == 0:
                if len(eos_positions) > 0:
                    timesteps = [tokens_np] if len(tokens_np) > 0 else [[]]
                else:
                    timesteps = [tokens_np]

            mat = np.zeros((len(timesteps), n_cols), dtype=np.uint8)

            # Fill the matrix per timestep
            # Skip static/demographic features as they're not in the time-series X matrix
            static_prefixes = (
                "age_static",
                "female_static",
                "height_static",
                "weight_static",
            )
            for t, toks in enumerate(timesteps):
                for tok in toks:
                    if tok in special_token:
                        continue
                    if not isinstance(tok, str) or "_bin_" not in tok:
                        continue
                    f_name, bin_str = tok.rsplit("_bin_", 1)
                    # Skip static/demographic tokens
                    if f_name.startswith(static_prefixes):
                        continue
                    if f_name not in feature_col:
                        continue  # Skip unknown features
                    bin_idx = int(bin_str)
                    if not 0 <= bin_idx < self.num_bins:
                        raise ValueError(
                            f"synthetic token {tok!r} has bin {bin_idx}, but the data has "
                            f"{self.num_bins} bins: corpus and real data come from different K"
                        )
                    mat[t, feature_col[f_name] + bin_idx] = 1

            patients_matrix.append(mat)
            lengths.append(len(timesteps))

        return patients_matrix, np.array(lengths)

    def _load_synthetic_data(self, path):
        tokens_list, labels_list, centre_list = joblib.load(path)
        self.X, lengths = self._map_tok_to_idx(tokens_list)

        print(f"[Synthetic Data] Loaded {len(tokens_list)} sequences from {path}")
        print(
            f"[Synthetic Data] Length distribution: min={np.min(lengths)}, max={np.max(lengths)}, mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}"
        )

        n_patients = len(labels_list)
        self.y_full = np.zeros((n_patients, 3), dtype=int)

        for i, label in enumerate(labels_list):
            self.y_full[i, 0] = (
                i + 300000
            )  # Assign unique stay_id starting from 300000 to avoid overlap with real data
            self.y_full[i, 1] = label
            self.y_full[i, 2] = 0.0

        n_statics = len(self.statics_set) if hasattr(self, "statics_set") else 4
        self.statics = np.zeros((n_patients, n_statics), dtype=float)
        self.times = [torch.arange(1, l + 1) for l in lengths]

        non_empty_idx = [
            i for i, t in enumerate(self.times) if t.numel() > self.min_length
        ]

        n_filtered = len(lengths) - len(non_empty_idx)
        if n_filtered > 0:
            print(
                f"[Synthetic Data] Filtered out {n_filtered}/{len(lengths)} sequences "
                f"with length <= {self.min_length} timepoints."
            )

        if len(non_empty_idx) == 0:
            raise ValueError(
                f"All {len(lengths)} synthetic sequences were filtered out because they have "
                f"<= {self.min_length} timepoints. Sequence length distribution: "
                f"min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}. "
                f"Consider lowering min_length or generating longer sequences."
            )

        self.X = [self.X[i] for i in non_empty_idx]
        self.y_full = torch.tensor(self.y_full[non_empty_idx], dtype=torch.float32)
        self.statics = torch.tensor(self.statics[non_empty_idx], dtype=torch.float32)
        self.times = [self.times[i] for i in non_empty_idx]
        self.lengths = torch.tensor(
            [lengths[i] for i in non_empty_idx], dtype=torch.int32
        )
        self.stay_sources = pd.Series(pd.array(["syn"] * len(self.X)))

        assert (
            len(self.X)
            == len(self.times)
            == len(self.statics)
            == len(self.y_full)
            == len(self.lengths)
            == len(self.stay_sources)
        ), "Mismatch in lengths of loaded synthetic data."

    def _get_effective_lookback(self, los=None, lookback=None):
        """Get the effective lookback value, sampling randomly if lookback is a tuple."""

        if lookback is None:
            lookback = self.lookback

        if isinstance(lookback, tuple):
            max_lookback = min(lookback[1], los - 1) if los is not None else lookback[1]
            max_lookback = max(max_lookback, lookback[0])  # Ensure valid range
            return np.random.randint(lookback[0], max_lookback + 1)

        return min(lookback, los - 1) if los is not None else lookback

    def _setup_early_warning(self):
        """Setup early warning specific preprocessing pipeline."""
        # Create windows based on the specified mode
        if self.use_sliding_windows:
            self._cut_stays_before_end()
            self._create_sliding_windows()
            self._apply_feature_set_filter()

            if self.include_missing_mask:
                self._compute_sliding_window_missingness(
                    feature_indices=self._feature_subset_indices
                )
                if self.missingness_tensor is not None:
                    m_width = self.missingness_tensor[0].shape[1]
                    n_feat = len(self.features)
                    assert m_width == n_feat, (
                        f"missingness_tensor width ({m_width}) != len(self.features) "
                        f"({n_feat}). feature_indices={self._feature_subset_indices}."
                    )
                self._apply_missingness_threshold_filter()
                self._update_feature_names_with_missingness()
        # Prepare sequences and labels for early warning
        self._prepare_sequences()

    def _stay_onset_hours(self, stay_end_hours):
        """Onset hour of every stay; 0 for negative stays.

        Death ends the stay, so its onset is the last observed hour. Every other
        outcome takes its onset from the {label}_onset column, which synthetic
        data does not have.
        """
        positive = self.y_full[:, 1].bool()
        if self.return_label.startswith("death"):
            onset = stay_end_hours.clone()
        else:
            has_onset_column = not self.synthetic and (
                f"{self.return_label}_onset" in self.data_table.columns
            )
            if max(self.lookback, self.syn_lookback) > 0 and not has_onset_column:
                raise ValueError(
                    f"lookback needs a {self.return_label}_onset column, which this "
                    f"{'synthetic' if self.synthetic else 'real'} data does not have."
                )
            onset = self.y_full[:, 2].float()
        onset[~positive] = 0
        return onset

    def _cut_stays_before_end(self):
        """Keep only the hours up to lookback hours before each stay's reference hour.

        The reference hour is the onset for positive stays and the last observed
        hour for negative ones, so controls are cut the same way as cases. Window
        creation and missingness read the first lengths[i] rows of X, so shortening
        lengths and times cuts the windows as well.
        """
        lengths = (
            self.lengths
            if isinstance(self.lengths, list)
            else [int(l) for l in self.lengths]
        )
        assert len(self.times) == len(lengths) == len(self.X) == len(self.y_full)

        times_unpadded = []
        for times, length in zip(self.times, lengths):
            times = times[: torch.argmax(times) + 1]
            # X rows and times must describe the same hours, or windows and their
            # end times would point at different hours of the stay
            assert len(times) == length, f"{len(times)} times for {length} rows"
            assert torch.all(times[1:] > times[:-1]), "stay times are not increasing"
            times_unpadded.append(times)

        stay_end_hours = torch.tensor(
            [t[-1].item() for t in times_unpadded], dtype=torch.float32
        )
        self.stay_onset_hours = self._stay_onset_hours(stay_end_hours)
        positive = self.y_full[:, 1].bool()
        self.stay_reference_hours = torch.where(
            positive, self.stay_onset_hours, stay_end_hours
        )

        # Synthetic stays keep the hours either split may need; the datamodule
        # drops the rest once it knows which split each stay is in
        cut_hours = (
            min(self.lookback, self.syn_lookback) if self.synthetic else self.lookback
        )
        self.times = [
            t[t <= reference - cut_hours]
            for t, reference in zip(times_unpadded, self.stay_reference_hours.tolist())
        ]
        self.lengths = [len(t) for t in self.times]

        n_dropped = sum(l < self.window_size for l in self.lengths)
        print(
            f"[EarlyWarningDataset] {'synthetic' if self.synthetic else 'real'} "
            f"stays cut {cut_hours} h before their reference hour: "
            f"{n_dropped}/{len(self.lengths)} stays have fewer than "
            f"window_size={self.window_size} hours left and are dropped."
        )

    def _setup_sliding_window_sequences(self):
        """Prepare sequences for sliding window approach."""
        # Check for empty dataset
        if len(self.times) == 0 or len(self.statics) == 0:
            raise ValueError(
                f"Cannot setup sliding window sequences: dataset is empty. "
                f"times has {len(self.times)} entries, statics has {len(self.statics) if hasattr(self.statics, '__len__') else 0} entries. "
                f"Check that synthetic data was loaded correctly and sequences meet min_length requirements."
            )

        # Calculate lengths for sliding windows (number of windows per sequence);
        # times are already unpadded and cut by _cut_stays_before_end
        self.lengths = [max(0, len(t) - self.window_size + 1) for t in self.times]

        # Check if any valid windows can be created
        total_windows = sum(self.lengths)
        if total_windows == 0:
            raise ValueError(
                f"Cannot create any sliding windows: all {len(self.times)} sequences are shorter than window_size={self.window_size}. "
                f"Sequence lengths after padding removal: {[len(t) for t in self.times][:10]}... "
                f"Consider reducing window_size or increasing min_length filter."
            )

        # Create times tensor for each window (use the end time of each window)
        window_times = []
        for i, time_seq in enumerate(self.times):
            num_windows = self.lengths[i]
            if num_windows > 0:
                # For each window, use the time of the last element in the window
                for w in range(num_windows):
                    window_end_idx = w + self.window_size - 1
                    window_times.append(time_seq[window_end_idx])

        self.times = torch.stack(window_times) if window_times else torch.tensor([])

        # Repeat statics for each time step/window
        time_lengths = torch.tensor(self.lengths)

        self._patient_stay_sources = pd.Series(
            np.asarray(self.stay_sources)
        ).reset_index(drop=True)
        self.stay_sources = pd.Series(
            np.repeat(np.asarray(self._patient_stay_sources), time_lengths.numpy())
        ).reset_index(drop=True)

        if self.synthetic is False:
            statics = (
                self.processed_statics
                if getattr(self, "processed_statics", None) is not None
                else torch.stack(
                    [
                        torch.where(self.impute_FIDDLE_statics(static_vec))[0]
                        for static_vec in self.statics
                    ]
                )
            )
            self.statics = torch.repeat_interleave(statics, time_lengths, dim=0)
        else:
            self.statics = torch.repeat_interleave(self.statics, time_lengths, dim=0)

        # Prepare labels and metadata
        y_tensor = torch.tensor(np.array(self.y_full))
        y_repeated = torch.repeat_interleave(y_tensor, time_lengths, dim=0)

        # Extract IDs and times
        self.ids = y_repeated[
            :, 0
        ]  # TODO: replace with a large enough hash to avoid accidental overlap with real IDs

        self.onset = torch.repeat_interleave(self.stay_onset_hours, time_lengths)
        self.reference_hours = torch.repeat_interleave(
            self.stay_reference_hours, time_lengths
        )

        # Repeat lengths
        self.lengths = torch.cat(
            [torch.repeat_interleave(torch.tensor(l), l) for l in self.lengths]
        )

        # Extract binary labels
        self.y = y_repeated[:, 1].float()

        if self.binarize_outcome and not self.synthetic:
            if self.return_label == "death":
                self.y = (self.y > 1).float()
                self.y_full[:, 1] = (self.y_full[:, 1] > 1).long()
            else:
                self.y = (self.y > 0).float()
                self.y_full[:, 1] = (self.y_full[:, 1] > 0).long()

    def _setup_full_sequences(self):
        """Prepare full sequences with all timestep labels for parallel prediction."""
        # Process times and lengths based on real timesteps with data,
        # ignoring trailing empty hours (no measurements)
        if hasattr(self, "X_time_positions") and self.X_time_positions is not None:
            real_lengths = [
                tp.max().item() if len(tp) > 0 else 0 for tp in self.X_time_positions
            ]
            self.times = [
                t[: torch.argmax(t) + 1][:l] for t, l in zip(self.times, real_lengths)
            ]
            self.lengths = real_lengths
        else:
            self.times = [t[: torch.argmax(t) + 1] for t in self.times]
            self.lengths = [len(t) for t in self.times]

        # For sequence-level approach: return full sequences, not windows
        # Each sample is one complete patient sequence

        # Keep times as list of sequences
        # times will be processed during collation

        if self.synthetic is False:
            statics = (
                self.processed_statics
                if getattr(self, "processed_statics", None) is not None
                else torch.stack(
                    [
                        torch.where(self.impute_FIDDLE_statics(static_vec))[0]
                        for static_vec in self.statics
                    ]
                )
            )
            self.statics = statics  # Keep original shape, no repetition
        else:
            self.statics = self.statics

        # Prepare labels and metadata for each sequence
        y_tensor = torch.tensor(np.array(self.y_full))

        # For sequence-level: keep original y structure
        self.ids = y_tensor[:, 0]
        self.y = y_tensor[:, 1].float()  # Binary outcome labels
        if self.binarize_outcome and not self.synthetic:
            if self.return_label == "death":
                self.y = (self.y > 1).float()
                self.y_full[:, 1] = (self.y_full[:, 1] > 1).long()
            else:
                self.y = (self.y > 0).float()
                self.y_full[:, 1] = (self.y_full[:, 1] > 0).long()

        # Handle onset information
        if y_tensor.shape[1] > 2:
            self.onset = y_tensor[:, 2].float()  # Use provided onset
        else:
            # Use sequence length as onset for cases without explicit onset
            self.onset = torch.tensor(
                [length - 1 for length in self.lengths], dtype=torch.float
            )
            # Set onset to 0 for negative cases
            self.onset[~self.y.bool()] = 0.0

        # Store full tokenized sequences. Direct FIDDLE pickle loads are tokenized
        # by ICUDataset when full-sequence early warning is requested.
        self.full_sequences = (
            self.X_tokens if hasattr(self, "X_tokens") else self.X_value_ids
        )

    def _direct_values_for_flat_decoding(self, values):
        """Convert direct FIDDLE value IDs to ordinal bins for flat decoders."""
        values = np.asarray(values, dtype=float).copy()
        missing_id = self.num_bins
        middle_bin = self.num_bins // 2
        missing = values == missing_id

        if self.impute_missing == "forwardfill":
            values[missing] = np.nan
            values = pd.DataFrame(values).ffill().to_numpy()
            values[np.isnan(values)] = middle_bin
        else:
            values[missing] = middle_bin

        return np.clip(values, 0, self.num_bins - 1)

    def _create_sliding_windows(self):
        """Create sliding windows from tokenized sequences for ragged tensor support."""
        sliding_windows = []

        # Debug: Print info about self.X before processing
        print(f"[DEBUG _create_sliding_windows] self.X type: {type(self.X)}")
        if hasattr(self.X, "shape"):
            print(f"[DEBUG _create_sliding_windows] self.X shape: {self.X.shape}")
        elif isinstance(self.X, list):
            print(
                f"[DEBUG _create_sliding_windows] self.X is list with {len(self.X)} elements"
            )
        else:
            print(f"[DEBUG _create_sliding_windows] self.X: {self.X}")

        # Handle ragged X format
        X_list = (
            self.X
            if isinstance(self.X, list)
            else [self.X[i] for i in range(len(self.X))]
        )

        # Check for empty data
        if len(X_list) == 0:
            # Print more diagnostic info
            print(f"[DEBUG] self.synthetic: {getattr(self, 'synthetic', 'N/A')}")
            print(
                f"[DEBUG] self.times type: {type(getattr(self, 'times', None))}, len: {len(getattr(self, 'times', []))}"
            )
            print(f"[DEBUG] self.lengths type: {type(getattr(self, 'lengths', None))}")
            if hasattr(self, "lengths"):
                print(f"[DEBUG] self.lengths: {self.lengths}")
            raise ValueError(
                "Cannot create sliding windows: no sequences available. "
                "All sequences may have been filtered out due to min_length requirements. "
                f"self.X type={type(self.X)}, synthetic={getattr(self, 'synthetic', 'N/A')}"
            )

        lengths_list = (
            self.lengths
            if isinstance(self.lengths, list)
            else [self.lengths[i].item() for i in range(len(self.lengths))]
        )

        for i, X in tqdm(
            enumerate(X_list), desc="Creating sliding windows", total=len(X_list)
        ):
            # Handle both tensor and numpy array formats
            if isinstance(X, torch.Tensor):
                X_np = X.detach().cpu().numpy()
            else:
                X_np = np.asarray(X)

            # Get actual sequence length and remove padding
            seq_len = lengths_list[i]
            X_np = X_np[:seq_len]

            if getattr(self, "is_direct_fiddle", False) or hasattr(self, "X_value_ids"):
                X_tokens = self._direct_values_for_flat_decoding(X_np)
                if X_tokens.shape[0] >= self.window_size:
                    for w in range(X_tokens.shape[0] - self.window_size + 1):
                        window = torch.tensor(
                            X_tokens[w : w + self.window_size], dtype=torch.float32
                        )
                        sliding_windows.append(window)
                continue

            # Legacy FIDDLE one-hot path: reshape to make quantiles explicit.
            X_quantiles = X_np.astype(float).reshape(
                X_np.shape[0], X_np.shape[1] // self.num_bins, self.num_bins
            )

            if self.impute_missing == "forwardfill":
                missing_indices = np.where(np.sum(X_quantiles, axis=2) == 0)
                X_quantiles[missing_indices] = np.nan

                for j in range(X_quantiles.shape[1]):
                    X_quantiles[:, j, :] = (
                        pd.DataFrame(X_quantiles[:, j, :]).ffill().values
                    )

                X_quantiles[np.isnan(X_quantiles)] = 0.0

            if self.impute_missing:
                missing_indices = np.where(np.sum(X_quantiles, axis=2) == 0)
                X_quantiles[
                    missing_indices[0], missing_indices[1], self.num_bins // 2
                ] = 1.0

            # Get ordinal representation of quintiles
            X_quantiles = X_quantiles.argmax(axis=2)

            # Create sliding window view - store as list of variable-length windows
            if X_quantiles.shape[0] >= self.window_size:
                for w in range(X_quantiles.shape[0] - self.window_size + 1):
                    window = torch.tensor(X_quantiles[w : w + self.window_size]).float()
                    sliding_windows.append(window)

        # Check if any windows were created
        if len(sliding_windows) == 0:
            raise ValueError(
                f"No sliding windows could be created from {len(X_list)} sequences. "
                f"All sequences may be shorter than window_size={self.window_size}. "
                f"Sequence lengths: {lengths_list[:10]}... "
                f"Consider reducing window_size or min_length."
            )

        # Store as list for ragged format (padding will be done in collate function)
        self.window_tensor = sliding_windows

    def _compute_sliding_window_missingness(self, feature_indices=None):
        """Compute missingness indicators for sliding windows only."""
        if not self.use_sliding_windows or not self.include_missing_mask:
            return

        missingness_windows = []
        X_list = (
            self.X
            if isinstance(self.X, list)
            else [self.X[i] for i in range(len(self.X))]
        )
        lengths_list = (
            self.lengths
            if isinstance(self.lengths, list)
            else [self.lengths[i].item() for i in range(len(self.lengths))]
        )

        for i, X in enumerate(X_list):
            X_tensor = (
                X.detach().cpu() if isinstance(X, torch.Tensor) else torch.tensor(X)
            )
            seq_len = lengths_list[i]
            X_tensor = X_tensor[:seq_len]

            if getattr(self, "is_direct_fiddle", False) or hasattr(self, "X_value_ids"):
                missing_mask = (X_tensor == self.num_bins).float()
            else:
                X_quantiles = X_tensor.float().reshape(
                    X_tensor.shape[0],
                    X_tensor.shape[1] // self.num_bins,
                    self.num_bins,
                )
                missing_mask = (X_quantiles.sum(axis=2) == 0).float()

            if feature_indices is not None:
                missing_mask = missing_mask[:, feature_indices]

            if missing_mask.shape[0] >= self.window_size:
                for w in range(missing_mask.shape[0] - self.window_size + 1):
                    missingness_windows.append(
                        missing_mask[w : w + self.window_size].detach()
                    )

        self.missingness_tensor = missingness_windows if missingness_windows else None

    def _apply_missingness_threshold_filter(self):
        """Apply missingness threshold filtering (sliding_window only)."""
        if (
            not self.use_sliding_windows
            or not self.include_missing_mask
            or self.missingness_tensor is None
        ):
            return

        if isinstance(self.missingness_threshold, float):
            self._filter_by_frequency_threshold()
        elif isinstance(self.missingness_threshold, str):
            self._filter_by_feature_set_threshold()

    def _filter_by_frequency_threshold(self):
        """Remove missingness indicators with frequency > threshold."""
        if not self.missingness_tensor:
            return

        # Calculate missingness frequency across all windows and time steps
        # Convert list of tensors to stacked tensor for computation
        stacked_missingness = torch.stack(self.missingness_tensor)
        missingness_freq = stacked_missingness.mean(
            dim=(0, 1)
        )  # Average across samples and time

        # Find features to keep (below threshold)
        features_to_keep = missingness_freq <= self.missingness_threshold

        # Filter missingness tensor - keep as list but filter features
        self.missingness_tensor = [
            window[:, features_to_keep] for window in self.missingness_tensor
        ]

    def _filter_by_feature_set_threshold(self):
        """Remove missingness indicators for specified feature set (e.g., 'vitals')."""
        if not self.missingness_tensor:
            return

        frequent_features = sorted(get_feature_set(self.missingness_threshold))
        base_features = self._get_base_feature_names_from_fiddle()

        # Find which features to keep missingness indicators for
        features_to_keep_mask = torch.tensor(
            [feat not in frequent_features for feat in base_features], dtype=torch.bool
        )

        # Filter missingness tensor - keep as list but filter features
        self.missingness_tensor = [
            window[:, features_to_keep_mask] for window in self.missingness_tensor
        ]

    def _get_base_feature_names_from_fiddle(self):
        """Get base feature names from FIDDLE processing."""
        # Extract base feature names from FIDDLE features
        fiddle_features = sorted(self.features)
        base_features = (
            pd.Series(fiddle_features)
            .str.split("_")
            .str[:2]  # Take first two parts (feature_type)
            .str.join("_")
            .drop_duplicates()
            .values
        )
        return base_features

    def _update_feature_names_with_missingness(self):
        """Update feature names to include missingness indicators (sliding_window only)."""
        if (
            not self.use_sliding_windows
            or not self.include_missing_mask
            or self.missingness_tensor is None
        ):
            self.feature_names_with_missingness = self.features
            return

        # Get base feature names from FIDDLE processing
        base_features = self._get_base_feature_names_from_fiddle()

        # Determine which features have missingness indicators after filtering
        if isinstance(self.missingness_threshold, float):
            # Get features that passed frequency threshold
            missingness_source = (
                torch.stack(self.missingness_tensor)
                if isinstance(self.missingness_tensor, list)
                else self.missingness_tensor
            )
            missingness_freq = missingness_source.mean(dim=(0, 1))
            kept_features = [
                feat
                for i, feat in enumerate(base_features)
                if i < len(missingness_freq)  # Safety check
            ]
        elif isinstance(self.missingness_threshold, str):
            # Get features not in the excluded set
            frequent_features = sorted(get_feature_set(self.missingness_threshold))
            kept_features = [
                feat for feat in base_features if feat not in frequent_features
            ]
        else:
            kept_features = base_features

        # Create missingness feature names
        missingness_features = [f"{feat}__mask" for feat in kept_features]

        # Combine original features with missingness indicators
        self.feature_names_with_missingness = list(self.features) + missingness_features

    def _apply_feature_set_filter(self):
        """
        Restrict window_tensor to self.feature_set, if it differs from what
        was actually loaded. Must run right after _create_sliding_windows()
        and before _prepare_sequences(), since the latter mutates self.lengths
        from per-sequence to per-window values — any later recomputation from
        self.X would silently use the wrong lengths.
        """
        self._feature_subset_indices = None
        if self.feature_set is None:
            return

        if isinstance(self.feature_set, str):
            requested_features = sorted(get_feature_set(self.feature_set))
        else:
            requested_features = sorted(self.feature_set)

        if set(requested_features) == set(self.features):
            return

        all_features_sorted = sorted(self.features)
        missing = set(requested_features) - set(all_features_sorted)
        assert not missing, (
            f"Requested feature_set includes features not present in the "
            f"loaded dataset: {sorted(missing)}."
        )

        feature_indices = [all_features_sorted.index(f) for f in requested_features]
        self.window_tensor = [w[:, feature_indices] for w in self.window_tensor]
        self.features = requested_features
        self._feature_subset_indices = feature_indices

    def _prepare_sequences(self):
        """Prepare sequences, statics, and labels for early warning task."""
        if self.use_sliding_windows:
            self._setup_sliding_window_sequences()

        else:
            self._setup_full_sequences()

    def __getitem__(self, index):
        """
        Get item for early warning training.

        Returns:
            Tuple of (sequence_with_eos, static_features, label, onset)
        """
        # Check if early warning setup is complete
        if not getattr(self, "_early_warning_setup_complete", False):
            # During initialization, fall back to parent class behavior
            if self.get_item_mode == "tuples":
                return self._get_tuple_item(index)
            else:
                return (
                    super().__getitem__(index)
                    if hasattr(super(), "__getitem__")
                    else None
                )

        if self.use_sliding_windows:
            # lookback is already applied: _cut_stays_before_end removed the
            # windows after it when the dataset was built
            # Base feature tensor for sliding window
            if isinstance(self.window_tensor, list):
                # Handle ragged format - window is already the right shape
                feature_tensor = self.window_tensor[index]
                if feature_tensor.dim() == 2:
                    feature_tensor = feature_tensor.transpose(1, 0)
            else:
                # Legacy padded format
                feature_tensor = self.window_tensor[index].transpose(1, 0)

            # Conditionally add missingness indicators (only for sliding_window)
            if (
                self.include_missing_mask
                and hasattr(self, "missingness_tensor")
                and self.missingness_tensor is not None
            ):
                if isinstance(self.missingness_tensor, list):
                    missingness_tensor = self.missingness_tensor[index]
                    if missingness_tensor.dim() == 2:
                        missingness_tensor = missingness_tensor.transpose(1, 0)
                else:
                    missingness_tensor = self.missingness_tensor[index].transpose(1, 0)
                # Concatenate features with missingness indicators
                feature_tensor = torch.cat([feature_tensor, missingness_tensor], dim=0)

            return (
                feature_tensor,
                self.statics[index],
                self.y[index],
                self.times[index],
                self.onset[index],
                self.ids[index],
            )

        else:
            # Return full sequence data, compatible with vectorised early warning processing
            sequence_data = self.full_sequences[index]
            sequence_data = sequence_data[:-1]  # Remove EOS token

            # Process individual elements
            label = self.y[index]
            times = F.pad(self.times[index], (1, 0), value=-1)  # Add SOS time
            onset = self.onset[index]

            if label.item() > 0:
                # For positive cases, ensure we correctly filter using the lookback parameter
                effective_lookback = self._get_effective_lookback(
                    los=int(self.lengths[index].item()) - self.min_length + 1
                )
                sequence_data = sequence_data[times <= onset - effective_lookback]
                times = times[times <= onset - effective_lookback]

            return (
                sequence_data,  # Full tokenized sequence
                self.statics[index],  # Static features
                label.expand(len(times)),  # Outcome label
                times,  # Time points for each position
                onset,  # Onset time
                self.ids[index],  # Patient ID
            )

    @property
    def y_series(self):
        if self.use_sliding_windows:
            y = self.y.bool()
            index_ids, index_times, index_onset = (
                np.array(self.ids),
                np.array(self.times),
                np.array(self.onset),
            )
            # stay_sources is expanded per-window above, already aligned here.
            index_sources = np.array(self.stay_sources)
        else:
            y = (
                torch.repeat_interleave(
                    self.y, torch.tensor([len(t) for t in self.times]), dim=0
                )
                .bool()
                .numpy()
            )
            index_ids = torch.repeat_interleave(
                self.ids, torch.tensor([len(t) for t in self.times]), dim=0
            ).numpy()
            index_times = torch.cat(self.times).numpy()
            index_onset = torch.repeat_interleave(
                self.onset, torch.tensor([len(t) for t in self.times]), dim=0
            ).numpy()
            # stay_sources is still per-patient here (same granularity as ids
            # before expansion); expand it the same way.
            index_sources = np.repeat(
                np.asarray(self.stay_sources), [len(t) for t in self.times]
            )

        y = pd.Series(y, dtype=float)
        y.index = pd.MultiIndex.from_arrays(
            [index_sources, index_ids, index_times, index_onset],
            names=["source", "ids", "times", "onset"],
        )

        # Filter to only include controls and times before onset for cases
        y = y[
            (y.index.get_level_values("times") >= 0)
            & (
                (y == 0)
                | (
                    y.index.get_level_values("times")
                    <= y.index.get_level_values("onset")
                )
            )
        ]

        return y

    def set_lookback(self, lookback: Union[int, Tuple[int, int]]):
        """
        Set the lookback value for filtering.

        Args:
            lookback: Lookback value (int) or tuple of (min, max) for random augmentation
        """
        if isinstance(lookback, tuple):
            assert (
                len(lookback) == 2
            ), "Lookback tuple must have exactly 2 elements (min, max)."
            assert (
                lookback[0] >= 0 and lookback[1] >= 0
            ), "Lookback values cannot be negative."
            assert lookback[0] <= lookback[1], "Lookback minimum must be <= maximum."

            self._is_random_lookback = True
            self.lookback = lookback
        else:
            assert lookback >= 0, "Lookback cannot be negative."

            self._is_random_lookback = False
            self.lookback = lookback

    def get_current_lookback_info(self):
        """Get information about current lookback configuration for debugging."""
        if self._is_random_lookback:
            min_lookback, max_lookback = self.lookback
            return {
                "type": "random",
                "range": self.lookback,
                "available_lookbacks": list(range(min_lookback, max_lookback + 1)),
            }
        else:
            return {"type": "fixed", "value": self.lookback}

    def _get_tuple_item(self, idx):
        """Get tuple format item - used during FIDDLE processing."""
        return (
            format_set(self.X[idx], self.missing_mask[idx], self.times[idx]),
            self.statics[idx],
            self.y[idx],
        )

    def __len__(self):
        """Return number of samples in the dataset."""
        # Handle case where dataset is still being initialized
        if not getattr(self, "_early_warning_setup_complete", False):
            # Fallback to parent class length during initialization
            return super().__len__()

        if self.use_sliding_windows:
            # Return length based on unfiltered dataset IDs
            return len(self.ids.unique())
        else:
            return len(self.full_sequences)

    def __repr__(self):
        """String representation of the dataset."""
        base_repr = super().__repr__()
        mode = "sliding window" if self.use_sliding_windows else "full autoregressive"
        return base_repr.replace(f"ICU Dataset", f"Early Warning Dataset | {mode} mode")

    def __str__(self):
        """String representation of the dataset."""
        return self.__repr__()


class EarlyWarningSubset:
    """
    Custom subset class for EarlyWarningDataset, compatible with on-the-fly lookback filtering.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self._subset_lookback = dataset.lookback

    def __getitem__(self, index):
        """Get item by forwarding to the main dataset with the correct index."""
        # Temporarily set the dataset's lookback to this subset's lookback
        original_lookback = self.dataset.lookback
        self.dataset.lookback = self._subset_lookback
        try:
            return self.dataset.__getitem__(self.indices[index])
        finally:
            # Restore original lookback
            self.dataset.lookback = original_lookback

    def __len__(self):
        """Return the number of samples in this subset."""
        return len(self.indices)

    def set_lookback(self, lookback: Union[int, Tuple[int, int]]):
        """Set lookback for this subset."""
        self._subset_lookback = lookback

    @property
    def lookback(self):
        """Get this subset's lookback value."""
        return self._subset_lookback

    def _get_effective_lookback(self, los=None, lookback=None):
        """Get the effective lookback value for this subset."""
        if lookback is None:
            lookback = self._subset_lookback

        return self.dataset._get_effective_lookback(los=los, lookback=lookback)

    @property
    def y_series(self):
        """Get subset of y_series."""
        # Temporarily set the dataset's lookback to this subset's lookback
        original_lookback = self.dataset.lookback
        self.dataset.lookback = self._subset_lookback

        try:
            full_y_series = self.dataset.y_series
            subset_ids = self.dataset.ids[self.indices].numpy()
            subset_sources = np.asarray(self.dataset.stay_sources)[self.indices]

            subset_keys = pd.MultiIndex.from_arrays(
                [subset_sources, subset_ids]
            ).unique()
            index_keys = pd.MultiIndex.from_arrays(
                [
                    full_y_series.index.get_level_values("source"),
                    full_y_series.index.get_level_values("ids"),
                ]
            )
            return full_y_series.loc[index_keys.isin(subset_keys)]
        finally:
            self.dataset.lookback = original_lookback

    # Forward other commonly needed attributes
    def __getattr__(self, name):
        """Forward attribute access to the underlying dataset."""
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        # raise AttributeError(
        #     f"'{self.__class__.__name__}' object has no attribute '{name}'"
        # )

    # def __repr__(self):
    #     original_length = re.findall(r"\| (\d+) samples", self.dataset.__repr__())[0]
    #     base_repr = self.dataset.__repr__().replace(original_length, str(len(self)))
    #     return base_repr + f" (Subset with lookback = {self._subset_lookback})"

    def __str__(self):
        """String representation of the dataset."""
        return self.__repr__()


class EarlyWarningDatamodule(L.LightningDataModule):
    """
    Early Warning DataModule that handles dataset creation and splitting
    """

    def __init__(self, **kwargs):
        """Same parameters as EarlyWarningDataset."""
        super().__init__()
        # Store all the same parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Set defaults
        if not hasattr(self, "real_path"):
            self.real_path = None
        if not hasattr(self, "val_set"):
            self.val_set = True
        if not hasattr(self, "test_on_syn"):
            self.test_on_syn = False
        if (
            not hasattr(self, "percentage_synthetic")
            or self.percentage_synthetic is None
        ):
            self.percentage_synthetic = 0.0
        if not hasattr(self, "sources") or self.sources is None:
            self.sources = [["syn"], ["eicu", "miiv", "hirid", "mimic", "aumc", "sic"]]
        if not hasattr(self, "split_fractions"):
            self.split_fractions = np.array([0.7, 0.1, 0.2])
        else:
            self.split_fractions = np.array(self.split_fractions)
        print(f"[EarlyWarningDatamodule] Initial sources: {self.sources}")

        if not hasattr(self, "n_real_train"):
            self.n_real_train = None
        if not hasattr(self, "augment_with_all_syn"):
            self.augment_with_all_syn = False
        # Splitting parameters
        self.split_fractions = self.split_fractions / np.sum(self.split_fractions)
        self.split_mode = kwargs.get("split_mode") or "stay_id"  # Use "stay_id" if None

        if self.split_mode == "centre":
            assert isinstance(
                self.sources[0], list
            ), "For centre split, sources must be a list of lists including training/validation and test sources"
            assert (
                len(self.sources) == 2
            ), "For centre split, sources must be a list of two lists including training/validation and test sources"
            self.train_sources = self.sources[0]
            self.test_sources = self.sources[-1]
            self.sources = self.train_sources + self.test_sources
        else:
            # Flatten sources if it's a nested list
            if (
                isinstance(self.sources, list)
                and self.sources
                and isinstance(self.sources[0], list)
            ):
                self.sources = [item for sublist in self.sources for item in sublist]
            self.train_sources = self.test_sources = self.sources

        print(
            f"[EarlyWarningDatamodule] Flattened sources for filtering: {self.sources}"
        )

        self.full_dataset = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def compute_samples_by_prevalence(
        self, data_ids, labels, prevalence, target_size=None
    ):
        """Compute number of case and control samples based on prevalence in the data."""

        case_ids = data_ids[labels == 1]
        control_ids = data_ids[labels == 0]
        # Compute number of samples
        total = target_size if target_size is not None else len(data_ids)
        num_case = max(1, int(round(total * prevalence))) if len(case_ids) > 0 else 0
        num_control = max(1, total - num_case) if len(control_ids) > 0 else 0

        replace_case = num_case > len(case_ids)
        replace_control = num_control > len(control_ids)

        rng = np.random.default_rng(getattr(self, "random_state", 42))
        case_sampled = (
            rng.choice(case_ids, size=num_case, replace=replace_case)
            if num_case > 0
            else np.array([], dtype=data_ids.dtype)
        )
        control_sampled = (
            rng.choice(control_ids, size=num_control, replace=replace_control)
            if num_control > 0
            else np.array([], dtype=data_ids.dtype)
        )

        return np.concatenate([case_sampled, control_sampled])

    def split(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset into train/val/test.

        Four modes:
        - CASE 1 (centre):    train on one centre, test on another. Optionally augment train with synthetic.
        - CASE 2 (full syn):  no real data, use synthetic only.
        - CASE 3 (test on syn): train on real, test on synthetic.
        - CASE 4 (real + syn):  fixed 12k patients with % substitution (exp 2) or fixed n_real + augmentation (exp 1).
        """

        all_ids = self.full_dataset.ids.numpy().astype(int)
        all_sources = np.array(self.full_dataset.stay_sources)
        labels_id_full = np.asarray(self.full_dataset.y).astype(int)

        source_codes, _ = pd.factorize(
            all_sources
        )  # ej: eicu→0, miiv→1, syn→2, mimic→3...
        OFFSET = 10**9
        all_keys = (
            source_codes * OFFSET + all_ids
        )  # combine source and patient ID to create unique keys

        unique_df = (
            pd.DataFrame(
                {"key": all_keys, "source": all_sources, "label": labels_id_full}
            )
            .drop_duplicates(subset="key")
            .reset_index(drop=True)
        )
        unique_patient_ids = unique_df["key"].values
        patient_sources = unique_df["source"].values
        labels_id = unique_df["label"].values

        print("Total unique patient IDs:", len(unique_patient_ids))
        print("Overall labels distribution:", np.bincount(labels_id))

        # Helper available to all cases
        def get_dist(patients):
            if len(patients) == 0:
                return []
            labs = labels_id[np.isin(unique_patient_ids, patients)]
            return np.bincount(labs)

        # Separate real and synthetic patients
        real_patient_ids = np.unique(unique_patient_ids[patient_sources != "syn"])
        synthetic_patient_ids = np.unique(unique_patient_ids[patient_sources == "syn"])

        print("Total real patient IDs available:", len(real_patient_ids))
        print("Total synthetic patient IDs available:", len(synthetic_patient_ids))

        # real prevalence
        if len(real_patient_ids) > 0:
            real_labels_for_prev = labels_id[
                np.isin(unique_patient_ids, real_patient_ids)
            ]
            real_prevalence = np.mean(real_labels_for_prev == 1)
        else:
            syn_labels_for_prev = labels_id[
                np.isin(unique_patient_ids, synthetic_patient_ids)
            ]
            real_prevalence = (
                np.mean(syn_labels_for_prev == 1)
                if len(syn_labels_for_prev) > 0
                else 0.5
            )

        # CASE 1: CENTRE-BASED SPLIT
        # Train on one centre, test on another.
        # Val comes from the test centre (merged with test if val_set=False).
        # Synthetic can be added to train as augmentation.

        if self.split_mode == "centre" and self.real_path is not None:
            print("\n=== CENTRE-BASED SPLIT ===")

            # Mask on unique_patient_sources (same length as unique_patient_ids),
            train_mask = np.isin(patient_sources, self.train_sources)
            test_mask = np.isin(patient_sources, self.test_sources)

            train_real_ids = np.unique(unique_patient_ids[train_mask])
            test_real_ids = np.unique(unique_patient_ids[test_mask])

            test_labels = labels_id[np.isin(unique_patient_ids, test_real_ids)]

            print(f"Train centre {self.train_sources}: {len(train_real_ids)} patients")
            print(f"Test centre  {self.test_sources}:  {len(test_real_ids)} patients")

            # All train-centre patients go to train
            train_patients = train_real_ids

            # Split test-centre patients into val / test
            val_patients, test_patients = train_test_split(
                test_real_ids,
                test_size=self.split_fractions[2]
                / (self.split_fractions[1] + self.split_fractions[2]),
                random_state=getattr(self, "random_state", 42),
                stratify=test_labels if len(np.unique(test_labels)) > 1 else None,
            )

            print(f"Val patients  (test centre): {len(val_patients)}")
            print(f"Test patients (test centre): {len(test_patients)}")

            # Optionally augment train with synthetic data
            if (
                hasattr(self, "syn_path")
                and self.syn_path is not None
                and self.percentage_synthetic > 0
                and len(synthetic_patient_ids) > 0
            ):
                syn_labels_all = labels_id[
                    np.isin(unique_patient_ids, synthetic_patient_ids)
                ]
                if self.percentage_synthetic >= 1.0:
                    # percentage_synthetic=1.0 in centre mode → add ALL synthetic
                    train_syn_sample = self.compute_samples_by_prevalence(
                        synthetic_patient_ids,
                        syn_labels_all,
                        real_prevalence,
                        target_size=len(synthetic_patient_ids),
                    )
                    print(
                        f"percentage_synthetic=1.0 in centre mode → adding ALL {len(train_syn_sample)} synthetic patients"
                    )
                else:
                    n_train_syn = int(
                        len(train_patients)
                        * self.percentage_synthetic
                        / (1 - self.percentage_synthetic)
                    )
                    train_syn_sample = self.compute_samples_by_prevalence(
                        synthetic_patient_ids,
                        syn_labels_all,
                        real_prevalence,
                        target_size=n_train_syn,
                    )
                    print(f"Added {n_train_syn} synthetic patients to train")

                train_patients = np.concatenate([train_patients, train_syn_sample])

            print("\nFinal distributions →")
            print(f"  Train: {len(train_patients)} {get_dist(train_patients)}")
            print(f"  Val:   {len(val_patients)}   {get_dist(val_patients)}")
            print(f"  Test:  {len(test_patients)}  {get_dist(test_patients)}")

        # CASE 2: FULL SYNTHETIC
        # No real data available. Use synthetic only.

        elif self.real_path is None:
            print("\n=== FULL SYNTHETIC DATA ===")

            target_patients = min(12000, len(synthetic_patient_ids))
            syn_labels_all = labels_id[
                np.isin(unique_patient_ids, synthetic_patient_ids)
            ]
            sampled_synthetic = self.compute_samples_by_prevalence(
                synthetic_patient_ids,
                syn_labels_all,
                real_prevalence,
                target_size=target_patients,
            )

            mask_syn = np.isin(unique_patient_ids, sampled_synthetic)
            syn_labels = labels_id[mask_syn]

            print(f"Using {len(sampled_synthetic)} synthetic patients")
            print(f"Labels distribution: {np.bincount(syn_labels)}")

            train_val_patients, test_patients = train_test_split(
                sampled_synthetic,
                test_size=self.split_fractions[2],
                random_state=getattr(self, "random_state", 42),
                stratify=syn_labels if len(np.unique(syn_labels)) > 1 else None,
            )

            labels_tval = labels_id[np.isin(unique_patient_ids, train_val_patients)]

            train_patients, val_patients = train_test_split(
                train_val_patients,
                test_size=self.split_fractions[1] / self.split_fractions[:-1].sum(),
                random_state=getattr(self, "random_state", 42),
                stratify=labels_tval if len(np.unique(labels_tval)) > 1 else None,
            )

            print("\nFinal distributions →")
            print(f"  Train: {len(train_patients)} {get_dist(train_patients)}")
            print(f"  Val:   {len(val_patients)}   {get_dist(val_patients)}")
            print(f"  Test:  {len(test_patients)}  {get_dist(test_patients)}")

        # CASE 3: TEST ON SYNTHETIC
        # Train on real, test on synthetic.

        elif self.test_on_syn and self.syn_path is not None:
            print("\n=== TRAIN ON REAL, TEST ON SYNTHETIC ===")

            target_train = min(12000, len(real_patient_ids))
            mask_real = np.isin(unique_patient_ids, real_patient_ids)
            real_labels = labels_id[mask_real]

            train_patients, _ = train_test_split(
                real_patient_ids,
                train_size=target_train,
                random_state=getattr(self, "random_state", 42),
                stratify=real_labels if len(np.unique(real_labels)) > 1 else None,
            )

            val_patients = np.array([], dtype=int)

            print(f"Train patients (real): {len(train_patients)}")
            print(f"Train labels: {get_dist(train_patients)}")

            mask_syn = np.isin(unique_patient_ids, synthetic_patient_ids)
            syn_labels = labels_id[mask_syn]

            test_size = self.split_fractions[2] / (
                self.split_fractions[1] + self.split_fractions[2]
            )

            val_patients, test_patients = train_test_split(
                synthetic_patient_ids,
                test_size=test_size,
                random_state=getattr(self, "random_state", 42),
                stratify=syn_labels if len(np.unique(syn_labels)) > 1 else None,
            )

            print(f"Val patients  (synthetic): {len(val_patients)}")
            print(f"Test patients (synthetic): {len(test_patients)}")

            print("\nFinal distributions →")
            print(f"  Train: {len(train_patients)} {get_dist(train_patients)}")
            print(f"  Val:   {len(val_patients)}   {get_dist(val_patients)}")
            print(f"  Test:  {len(test_patients)}  {get_dist(test_patients)}")

        # CASE 4: REAL + SYNTHETIC MIX
        # Two sub-modes:
        #   - Experiment 1: fixed n_real_train + optionally all synthetic as augmentation
        #   - Experiment 2: fixed 12k pool with percentage_synthetic substitution
        # Test is always 100% real and fixed.

        else:
            print("\n=== REAL + SYNTHETIC MIX ===")

            N_TOTAL = 12000
            n_test = int(N_TOTAL * self.split_fractions[2])
            n_val = int(N_TOTAL * self.split_fractions[1])

            real_ids = np.unique(unique_patient_ids[patient_sources != "syn"])
            syn_ids = np.unique(unique_patient_ids[patient_sources == "syn"])

            mask_real_all = np.isin(unique_patient_ids, real_ids)
            real_labels_all = labels_id[mask_real_all]

            # Fixed real split for val and test
            train_pool_real, test_patients = train_test_split(
                real_ids,
                test_size=min(n_test, len(real_ids) - 1),
                stratify=(
                    real_labels_all if len(np.unique(real_labels_all)) > 1 else None
                ),
                random_state=getattr(self, "random_state", 42),
            )

            labels_train_pool = labels_id[np.isin(unique_patient_ids, train_pool_real)]
            val_size = min(n_val, len(train_pool_real) - 1)

            if val_size > 0:
                train_pool_real, val_patients = train_test_split(
                    train_pool_real,
                    test_size=val_size,
                    stratify=(
                        labels_train_pool
                        if len(np.unique(labels_train_pool)) > 1
                        else None
                    ),
                    random_state=getattr(self, "random_state", 42),
                )
            else:
                val_patients = np.array([], dtype=int)

            print(
                f"Fixed splits → val: {len(val_patients)}, test: {len(test_patients)}"
            )

            # EXPERIMENT 1: fixed n_real_train + optionally all synthetic

            if getattr(self, "n_real_train", None) is not None:
                print(f"\n--- Experiment 1: n_real_train={self.n_real_train} ---")

                n_real = min(self.n_real_train, len(train_pool_real))
                labels_pool = labels_id[np.isin(unique_patient_ids, train_pool_real)]

                if n_real < len(train_pool_real):
                    minority_cls = np.argmin(np.bincount(labels_pool))
                    majority_cls = 1 - minority_cls

                    minority_patients = train_pool_real[labels_pool == minority_cls]
                    majority_patients = train_pool_real[labels_pool == majority_cls]

                    rng = np.random.default_rng(getattr(self, "random_state", 42))

                    # natural proportion of minority in the training pool determines the split, but we ensure at least 1 minority sample if there are any
                    n_minority = max(
                        1, int(n_real * (len(minority_patients) / len(train_pool_real)))
                    )
                    n_minority = min(n_minority, len(minority_patients))
                    n_majority = min(n_real - n_minority, len(majority_patients))

                    sampled_minority = rng.choice(
                        minority_patients, size=n_minority, replace=False
                    )
                    sampled_majority = rng.choice(
                        majority_patients, size=n_majority, replace=False
                    )

                    train_real_sample = np.concatenate(
                        [sampled_minority, sampled_majority]
                    )
                    print(
                        f"Stratified manual sample → minority ({minority_cls}): {n_minority}, majority ({majority_cls}): {n_majority}"
                    )
                else:
                    train_real_sample = train_pool_real

                train_patients = train_real_sample
                print(
                    f"Real patients in train: {len(train_patients)} {get_dist(train_patients)}"
                )

                # Augment with ALL synthetic if requested
                if getattr(self, "augment_with_all_syn", False) and len(syn_ids) > 0:
                    syn_labels_all = labels_id[np.isin(unique_patient_ids, syn_ids)]
                    syn_sample = self.compute_samples_by_prevalence(
                        syn_ids,
                        syn_labels_all,
                        real_prevalence,
                        target_size=len(syn_ids),
                    )
                    train_patients = np.concatenate([train_patients, syn_sample])
                    print(f"Augmented with ALL {len(syn_ids)} synthetic patients")

            # EXPERIMENT 2: % substitution within fixed 12k pool
            else:
                print(
                    f"\n--- Experiment 2: percentage_synthetic={self.percentage_synthetic} ---"
                )

                n_train = N_TOTAL - n_test - n_val
                n_train_syn = int(n_train * self.percentage_synthetic)
                n_train_real = n_train - n_train_syn

                print(f"Train composition → real: {n_train_real}, syn: {n_train_syn}")

                train_real_sample = np.random.choice(
                    train_pool_real,
                    size=min(n_train_real, len(train_pool_real)),
                    replace=len(train_pool_real) < n_train_real,
                )

                if n_train_syn > 0 and len(syn_ids) > 0:
                    syn_labels_all = labels_id[np.isin(unique_patient_ids, syn_ids)]
                    syn_prevalence = np.mean(syn_labels_all == 1)
                    # Cap real prevalence to what is actually available in synthetic
                    sampling_prevalence = min(real_prevalence, syn_prevalence)
                    train_syn_sample = self.compute_samples_by_prevalence(
                        syn_ids,
                        syn_labels_all,
                        sampling_prevalence,
                        target_size=n_train_syn,
                    )
                else:
                    train_syn_sample = np.array([], dtype=int)

                train_patients = np.concatenate([train_real_sample, train_syn_sample])

            print("\nFinal distributions →")
            print(f"  Train: {len(train_patients)} {get_dist(train_patients)}")
            print(f"  Val:   {len(val_patients)}   {get_dist(val_patients)}")
            print(f"  Test:  {len(test_patients)}  {get_dist(test_patients)}")

        # COMMON: assign indices
        self.train_indices = np.where(np.isin(all_keys, train_patients))[0]
        self.val_indices = np.where(np.isin(all_keys, val_patients))[0]
        self.test_indices = np.where(np.isin(all_keys, test_patients))[0]

        self.train_ids = all_ids[self.train_indices]
        self.val_ids = all_ids[self.val_indices]
        self.test_ids = all_ids[self.test_indices]

        print("\n=== FINAL SPLIT ===")
        print("Training samples:  ", len(self.train_indices))
        print("Validation samples:", len(self.val_indices))
        print("Test samples:      ", len(self.test_indices))

        if self.val_set is False and self.test_on_syn is False:
            self.test_indices = np.concatenate([self.test_indices, self.val_indices])
            self.test_ids = np.concatenate([self.test_ids, self.val_ids])
            print("Test samples (incl. val):", len(self.test_indices))

        return self.train_indices, self.val_indices, self.test_indices
        # holi :)
        # creo que va eh

    def _apply_lookback_per_split(self):
        """Drop the windows each split must not see.

        Synthetic stays were built with min(lookback, syn_lookback), so their
        windows are cut to syn_lookback in train and to lookback in val/test.
        """
        dataset = self.full_dataset
        hours_before_reference = (dataset.reference_hours - dataset.times).numpy()
        is_synthetic = np.asarray(dataset.stay_sources) == "syn"
        assert len(hours_before_reference) == len(is_synthetic)

        def keep(indices, synthetic_lookback):
            indices = np.asarray(indices)
            min_hours = np.where(
                is_synthetic[indices], synthetic_lookback, dataset.lookback
            )
            kept = indices[hours_before_reference[indices] >= min_hours]
            n_stays_left = len(np.unique(dataset.ids.numpy()[kept]))
            return kept, n_stays_left

        self.train_indices, n_train = keep(self.train_indices, dataset.syn_lookback)
        self.test_indices, n_test = keep(self.test_indices, dataset.lookback)
        if self.val_indices is not None:
            self.val_indices, _ = keep(self.val_indices, dataset.lookback)
        print(
            f"[EarlyWarningDatamodule] lookback={dataset.lookback} h "
            f"(synthetic train: {dataset.syn_lookback} h): "
            f"{len(self.train_indices)} train windows from {n_train} stays, "
            f"{len(self.test_indices)} test windows from {n_test} stays."
        )

    def setup(self, stage: Optional[str] = None):
        """Setup using lazy filtering approach."""
        if self.full_dataset is None:
            # Create the full dataset once with training lookback
            self.full_dataset = EarlyWarningDataset(
                label=self.label,
                lookback=self.lookback,
                syn_lookback=getattr(self, "syn_lookback", None),
                real_path=getattr(self, "real_path", None),
                syn_path=getattr(self, "syn_path", None),
                pkl_path=getattr(self, "pkl_path", None),
                sources=self.sources,
                ids_to_keep=getattr(self, "ids_to_keep", None),
                min_length=getattr(self, "min_length", 6),
                max_length=getattr(self, "max_length", 192),
                max_gap=getattr(self, "max_gap", 6),
                min_onset=getattr(self, "min_onset", 6),
                samples=getattr(self, "samples", None),
                X_FIDDLE_bins=getattr(self, "X_FIDDLE_bins", None),
                S_FIDDLE_bins=getattr(self, "S_FIDDLE_bins", None),
                match_cases_on=getattr(self, "match_cases_on", None),
                feature_set=getattr(self, "feature_set", "all"),
                statics_set=getattr(
                    self,
                    "statics_set",
                    ["female_static", "age_static", "height_static", "weight_static"],
                ),
                # Sliding window parameters
                window_size=getattr(self, "window_size", None),
                include_missing_mask=getattr(self, "include_missing_mask", True),
                missingness_threshold=getattr(self, "missingness_threshold", "vitals"),
                # forwarded only when given, so the dataset's own handling applies
                # otherwise (num_bins is read from the real data when there is some)
                **{
                    k: getattr(self, k)
                    for k in ("num_bins", "impute_missing")
                    if hasattr(self, k)
                },
            )
            # Get indices for each split
            self.split()
            if self.full_dataset.use_sliding_windows:
                self._apply_lookback_per_split()

            # Create subsets
            self.train_dataset = EarlyWarningSubset(
                self.full_dataset, self.train_indices
            )
            self.val_dataset = (
                EarlyWarningSubset(self.full_dataset, self.val_indices)
                if self.val_set
                else None
            )
            self.test_dataset = EarlyWarningSubset(self.full_dataset, self.test_indices)

            # Full sequences filter per subset, so val/test keep every hour up to
            # onset. Sliding windows were already cut for every split when the
            # dataset was built, and this has no effect on them.
            if self.val_dataset is not None:
                self.val_dataset.set_lookback(0)
            self.test_dataset.set_lookback(0)

            # Assign original dataset attributes to subsets
            datasets_to_update = [self.train_dataset, self.test_dataset]
            if self.val_dataset is not None:
                datasets_to_update.append(self.val_dataset)

            for attr in [
                "feature_set",
                "statics_set",
                "feature_names_with_missingness",
                "use_sliding_windows",
                "include_missing_mask",
                "features",
            ]:
                if hasattr(self.full_dataset, attr):
                    value = getattr(self.full_dataset, attr)
                    for ds in datasets_to_update:
                        setattr(ds, attr, value)

    def train_dataloader(self, shuffle=True) -> DataLoader:
        """Create training DataLoader with lookback filtering."""
        return DataLoader(
            self.train_dataset,
            collate_fn=partial(
                fiddle_collate,
                fine_tune=True,
                pad_token_id=293,  # self.tokenizer.get_pad_token_id(),
            ),
            batch_size=getattr(self, "batch_size", 512),
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(self, "num_workers", 1),
            pin_memory=False,
        )

    def val_dataloader(self, shuffle=False) -> DataLoader:
        """Create validation DataLoader without lookback filtering (lookback=0)."""
        return DataLoader(
            self.val_dataset,
            collate_fn=partial(
                fiddle_collate,
                fine_tune=True,
                pad_token_id=293,  # self.tokenizer.get_pad_token_id(),
            ),
            batch_size=getattr(self, "batch_size", 512),
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(self, "num_workers", 1),
            pin_memory=False,
        )

    def test_dataloader(self, shuffle=False) -> DataLoader:
        """Create test DataLoader without lookback filtering (lookback=0)."""
        return DataLoader(
            self.test_dataset,
            collate_fn=partial(
                fiddle_collate,
                fine_tune=True,
                pad_token_id=293,  # self.tokenizer.get_pad_token_id(),
            ),
            batch_size=getattr(self, "batch_size", 512),
            shuffle=shuffle,
            drop_last=False,
            num_workers=getattr(self, "num_workers", 1),
            pin_memory=False,
        )

    # def __str__(self):
    #     """String representation of the DataModule."""
    #     return (
    #         f"EarlyWarningDatamodule with {len(self.train_dataset)} training samples, "
    #         f"{len(self.val_dataset)} validation samples, and {len(self.test_dataset)} test samples."
    #     )

    def __repr__(self):
        """Official string representation of the DataModule."""
        return self.__str__()
