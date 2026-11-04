#!/usr/bin/env python

"""utils.py: Auxiliary functions for the SyntheticICU project."""

import copy
import numpy as np
import pandas as pd

try:
    import sparse
except ImportError:
    sparse = None
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def get_feature_set(feature_set):
    """Outputs the set of feature names to use (vitals, labs, all).

    Args:
        feature_set (str): name of the feature set to use

    Returns:
        list: list of feature names
    """
    feature_dict = {
        "vitals": [
            "temp_raw",
            "hr_raw",
            "resp_raw",
            "sbp_raw",
            "dbp_raw",
            "map_raw",
            "o2sat_raw",
            "urine_raw",
        ],
        "labs": [
            "crea_raw",
            "lact_raw",
            "bili_raw",
            "glu_raw",
            "k_raw",
            "be_raw",
            "bicar_raw",
            "fio2_raw",
            "lymph_raw",
            "pco2_raw",
            "ph_raw",
            "po2_raw",
            "plt_raw",
            "rbc_raw",
            "etco2_raw",
            "cl_raw",
            "mg_raw",
            "phos_raw",
            "ast_raw",
            "bun_raw",
            "alp_raw",
            "ca_raw",
            "bildir_raw",
            "tri_raw",
            "hct_raw",
            "hgb_raw",
            "ptt_raw",
            "wbc_raw",
            "fgn_raw",
            "cai_raw",
            "na_raw",
            "alb_raw",
            "alt_raw",
            "ck_raw",
            "crp_raw",
            "tnt_raw",
            "basos_raw",
            "bnd_raw",
            "eos_raw",
            "esr_raw",
            "hbco_raw",
            "inrpt_raw",
            "mch_raw",
            "mchc_raw",
            "mcv_raw",
            "methb_raw",
            "neut_raw",
            "pt_raw",
            "rdw_raw",
            "tco2_raw",
        ],
    }

    if feature_set == "all":
        return feature_dict["vitals"] + feature_dict["labs"]

    return feature_dict[feature_set]


class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def format_set(X, missing_mask, times):
    """Format a raw sample as timestamp, value and feature-index tuples.

    Args:
        X (torch.Tensor): The time series data.
        missing_mask (torch.Tensor): The missing data mask.
        times (torch.Tensor): The time indices.

    Returns:
        torch.Tensor: The formatted sample, of shape (n_observations x 3), where each observation
        is a triple containing the timestamp, the measurement, and a modality index.

    """
    # Get the number of time points and features
    T, C = X.shape[0], X.shape[1]

    # Create the time index
    time_idx = times.repeat(C, 1).T.reshape(T * C, 1)

    # Create the feature index
    feature_idx = torch.arange(C).repeat(T, 1).reshape(T * C, 1)

    # Flatten the time series data
    X = X.reshape(T * C, 1)
    missing_mask = missing_mask.reshape(T * C, 1)

    # Concatenate the time, feature, and data indices
    return torch.stack(
        (time_idx[missing_mask], X[missing_mask], feature_idx[missing_mask]), dim=1
    )


## Preprocessing functions


def parse_icu_data(
    core: str,
    meta: str,
    min_length: int = None,
    feature_columns: list = None,
    extra_columns: list = [],
    sources: list = None,
    patient_id: bool = False,
    include_ed: bool = False,
) -> pd.DataFrame:
    """Parses core and metadata from .parquet files. Returns a single formatted pd.DataFrame with relevant columns

    Args:
        core (str): path to the core data file
        meta (str): path to the metadata file
        min_length (int): minimum length of ICU stay to include
        feature_columns (list): list of columns to use as features
        extra_columns (list): list of columns to use as labels
        sources (list): list of dataset sources to use
        patient_id (bool): if True, use patient ID instead of stay ID
        include_ed (bool): if True, include the emergency department data

    Returns:
        pd.DataFrame: formatted DataFrame with relevant columns

    """
    # Load data
    core = pd.read_parquet(core)
    meta = pd.read_parquet(meta)
    # Convert to a pandas DataFrame if necessary
    core = core.replace({None: np.nan})
    meta = meta.replace({None: np.nan})

    base_columns = ["stay_uid", "stay_time", "centre"]
    core = pd.merge(core, meta, how="left")

    if feature_columns is None:
        feature_columns = [col for col in core.columns if col.endswith("_raw")]

    columns = base_columns + feature_columns + extra_columns

    # Add label onsets when available
    for column in extra_columns:
        if f"{column}_onset" in core.columns:
            columns.append(f"{column}_onset")

    if feature_columns is not None or extra_columns is not None:
        core = core.loc[:, list(set([col for col in columns if col is not None]))]

    core = core.sort_values(base_columns).reset_index(drop=True)

    try:
        core.rename(columns={"stay_uid": "stay_id"}, inplace=True)
    except KeyError:
        pass

    # Process each ICU stay individually
    def process_stay_id(group):
        first_measurement_index = (
            group.loc[:, feature_columns].isnull().all(axis=1).argmin()
        )

        # If no measurements are available, remove the entire stay
        if group.loc[:, feature_columns].isnull().all().all():
            return pd.DataFrame()

        # If only one time point is available, remove the entire stay
        if first_measurement_index == group.shape[0] - 1:
            return pd.DataFrame()

        group = group.iloc[first_measurement_index:, :]

        # Apply extra processing steps, as requested
        if not include_ed:
            group = group.loc[group.stay_time >= 0]
        if min_length is not None:
            if group.stay_time.max() < min_length:
                return pd.DataFrame(columns=group.columns)

        return group

    results = Parallel(n_jobs=-1)(
        delayed(process_stay_id)(group)
        for _, group in tqdm(core.groupby("stay_id"), desc="Processing ICU stays")
    )
    core = pd.concat([x for x in results if len(x)], ignore_index=True).reset_index(
        drop=True
    )

    if isinstance(sources, str):
        sources = [sources]

    if isinstance(sources, list):
        core = core.loc[core.centre.isin(sources)]
    return core


def pad_batch_to_max_length(
    X,
    pad_value,
    gap_token=None,
    sos_tokens=None,
    eos_token=None,
    how="right",
    return_attention_mask=False,
):
    """Pads the input sequences in a batch to the same length.

    Args:
        X (list): nested list of input sequences
        pad_value (int): value used for padding
        gap_token (int, optional): int value representing the <GAP> token. If None, no gap token is used.
        sos_tokens (list, optional): int value representing the <SOS> and <SOS_labels> (if enabled) tokens. If None, no SOS token is used.
        eos_token (int, optional): int value representing the <EOS> token. If None, no EOS token is used.
        how (str, optional): padding direction, either "right" or "left". Defaults to "right".
        return_attention_mask (bool, optional): whether to return the attention mask. Defaults to False.
    Returns:
        np.ndarray: padded input sequences
    """
    max_len_input = max([len(x) for x in X])

    # Pad the sequences in the batch to the length of the longest sequence
    if how == "right":
        padded_input = [
            F.pad(x, (0, max_len_input - len(x)), "constant", value=pad_value)
            for x in X
        ]
    elif how == "left":
        padded_input = [
            F.pad(x, (max_len_input - len(x), 0), "constant", value=pad_value)
            for x in X
        ]
    else:
        raise ValueError(f"how must be 'right' or 'left', got {how!r}")

    padded = torch.stack(padded_input)
    if return_attention_mask:
        # One vectorized comparison: a Python loop over every token cost about
        # half a second of worker CPU per batch of 32 x 3000 tokens.
        attention_mask = (padded != pad_value).long()
        assert attention_mask.shape == padded.shape
        return padded, attention_mask

    return padded


def shuffle_within_timeblocks(
    tokens_batch, time_positions=None, vocab=None, p=0.3, skip_demo_block=False
):
    """
    Shuffle tokens within each time block defined by the <GAP> token, while keeping
    the special tokens ("<PAD>", "<SOS>", "<EOS>") in place.
    Optionally, shuffle the corresponding time positions as well.

    Args:
        tokens_batch (torch.Tensor): shape (B, L), batch of token IDs
        time_positions (torch.Tensor or None): shape (B, L), same as tokens_batch
        vocab (dict): token IDs for "<PAD>", "<SOS>", "<EOS>", "<GAP>"
        p (float): probability to shuffle each sequence
        skip_demo_block (bool): if True, the first block (between <SOS> and the first
            <GAP>) is not shuffled. Set to True when demographics are prepended to the
            sequence so their fixed order is preserved.

    Returns:
        tokens_batch (torch.Tensor), time_positions (torch.Tensor or None)
    """
    # The block bookkeeping runs in numpy on views of the CPU tensors: per block
    # it replaces about eight small torch ops. This runs in DataLoader workers.
    assert tokens_batch.device.type == "cpu", "collate-time shuffle expects CPU"
    B = tokens_batch.shape[0]
    device = tokens_batch.device

    shuffled_tokens = tokens_batch.clone()
    shuffled_time = None
    if time_positions is not None:
        shuffled_time = time_positions.clone().to(
            device=device, dtype=time_positions.dtype
        )
        assert (
            shuffled_tokens.shape == shuffled_time.shape
        ), f"{shuffled_tokens.shape} vs {shuffled_time.shape}"

    # The RNG calls (this torch.rand, then one torch.randperm per block with
    # more than one token, in row and block order) are the same as before the
    # numpy rewrite, so a given seed produces the same permutations.
    mask_shuffle = torch.rand(B, device=device) < p

    is_special = (
        (tokens_batch == vocab["<PAD>"])
        | (tokens_batch == vocab["<SOS>"])
        | (tokens_batch == vocab["<EOS>"])
    )
    is_special_rows = is_special.numpy()

    for b in mask_shuffle.nonzero(as_tuple=True)[0].tolist():
        # numpy views: the helper's writes land in shuffled_tokens / shuffled_time.
        shuffle_within_timeblocks_row(
            token_row=shuffled_tokens[b].numpy(),
            time_row=shuffled_time[b].numpy() if shuffled_time is not None else None,
            is_special_row=is_special_rows[b],
            gap_token=vocab["<GAP>"],
            skip_demo_block=skip_demo_block,
        )

    # Special tokens are never moved, so their positions must be unchanged.
    assert torch.equal(
        is_special,
        (shuffled_tokens == vocab["<PAD>"])
        | (shuffled_tokens == vocab["<SOS>"])
        | (shuffled_tokens == vocab["<EOS>"]),
    )
    return shuffled_tokens, shuffled_time


def shuffle_within_timeblocks_row(
    token_row, time_row, is_special_row, gap_token, skip_demo_block
):
    """Permute, in place, the non-special tokens of each GAP-delimited block of a row.

    Draws one torch.randperm per block with more than one such token, in block
    order, which is the RNG call sequence shuffle_within_timeblocks relies on.
    `time_row` (or None) is permuted with the tokens.
    """
    assert token_row.shape == is_special_row.shape
    if time_row is not None:
        assert time_row.shape == token_row.shape
    # GAP tokens never move: blocks lie strictly between consecutive GAPs.
    boundaries = np.concatenate(
        ([-1], np.flatnonzero(token_row == gap_token), [len(token_row)])
    )
    for block_index in range(len(boundaries) - 1):
        # The first block holds the demographics, whose order is fixed.
        if skip_demo_block and block_index == 0:
            continue
        start = boundaries[block_index] + 1
        end = boundaries[block_index + 1]
        valid_positions = np.arange(start, end)[~is_special_row[start:end]]
        if len(valid_positions) > 1:
            perm = torch.randperm(len(valid_positions), device="cpu").numpy()
            # The right side is a copy (fancy indexing), so this permutes.
            token_row[valid_positions] = token_row[valid_positions[perm]]
            if time_row is not None:
                time_row[valid_positions] = time_row[valid_positions[perm]]


def pad_collate(
    batch,
    vocabulary,
    is_train,
    shuffle_tokens_within_timepoint=False,
    p_shuffle=0.3,
    label=None,
    include_demographics=False,
):
    """Pads the input sequences in a batch to the same length.

    Args:
        batch (list): list of input sequences
        vocabulary (dict): dictionary containing the vocabulary (including the "<PAD>" token)

    Returns:
        tuple: tuple containing the padded input sequences and their lengths
    """

    sos_tokens = [vocabulary["<SOS>"]]
    if label is not None:
        sos_label_token = f"<SOS_{label.upper()}>"
        if sos_label_token in vocabulary:
            sos_tokens.append(vocabulary[sos_label_token])

    # Check if batch contains time positions (5 elements) or not (4 elements)
    if len(batch[0]) == 5:
        # Batch contains time positions: (X, time_positions, statics, times, labels)
        X, time_positions, S, times, IDs = zip(*batch)

        padded_X, attention_mask = pad_batch_to_max_length(
            X,
            vocabulary["<PAD>"],
            vocabulary["<GAP>"],
            sos_tokens,
            vocabulary["<EOS>"],
            return_attention_mask=True,
        )

        # Pad time positions with 0 (same padding value as for time position 0)
        padded_time_positions = [
            F.pad(seq_time, (0, padded_X.shape[1] - len(seq_time)))
            for seq_time in time_positions
        ]
        padded_time_positions = torch.stack(padded_time_positions)

        # shuffle toks withing timeblocks ignoring special toks
        if is_train and shuffle_tokens_within_timepoint:
            padded_X, padded_time_positions = shuffle_within_timeblocks(
                padded_X,
                padded_time_positions,
                vocabulary,
                p_shuffle,
                skip_demo_block=include_demographics,
            )
        assert (
            padded_time_positions.shape == padded_X.shape == attention_mask.shape
        ), f"{padded_time_positions.shape} vs {padded_X.shape}"
        return (
            padded_X,
            padded_time_positions,
            attention_mask,
            torch.tensor(np.stack(S)),
            torch.from_numpy(np.array(IDs, dtype=np.float32)),
        )
    else:
        # Standard batch without time positions: (X, S, times, IDs)
        X, S, times, IDs = zip(*batch)

        padded_X, attention_mask = pad_batch_to_max_length(
            X,
            vocabulary["<PAD>"],
            vocabulary["<GAP>"],
            sos_tokens,
            vocabulary["<EOS>"],
            return_attention_mask=True,
        )

        # shuffle toks withing timeblocks ignoring special toks
        if is_train and shuffle_tokens_within_timepoint:
            padded_X, _ = shuffle_within_timeblocks(
                padded_X, None, vocabulary, p_shuffle
            )

        return (
            padded_X,
            attention_mask,
            torch.tensor(np.stack(S)),
            torch.from_numpy(np.array(IDs, dtype=np.float32)),
        )


## Functions to visualize the token sequence (during generation)


## Functions to convert the token sequence to nd array (during evaluation)


## Functions to compute the tokenBEP metric (during evaluation)


def subset_icu_dataset(self, indices):
    """Subset the dataset based on per-stay indices."""
    if hasattr(indices, "to_numpy"):
        indices = indices.to_numpy()
    indices = np.asarray(indices, dtype=np.int64)

    def take(values):
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

    subset_dataset = copy.deepcopy(self)

    if hasattr(self, "y"):
        y = (
            self.y.detach().cpu().numpy()
            if torch.is_tensor(self.y)
            else np.asarray(self.y)
        )
        keep_stay_ids = set(y[indices, 0].astype(int).tolist())
    else:
        subset_ids = subset_dataset.data_table.stay_id.unique()
        keep_stay_ids = set(np.asarray(subset_ids)[indices].astype(int).tolist())

    subset_dataset.data_table = subset_dataset.data_table.loc[
        subset_dataset.data_table.stay_id.astype(int).isin(keep_stay_ids)
    ].copy()

    for attr in [
        "X",
        "X_value_ids",
        "X_feature_ids",
        "statics",
        "processed_statics",
        "times",
        "lengths",
        "y",
        "y_full",  # __getitem__ reads labels from it, so it must stay aligned.
        "stay_sources",
        "missing_mask",
        "X_tokens",
        "X_time_positions",
    ]:
        if hasattr(self, attr):
            setattr(subset_dataset, attr, take(getattr(self, attr)))

    return subset_dataset


# Functions to evaluate classification tasks


def early_warning_scorer(
    estimator=None,
    X=None,
    y=None,
    threshold=None,
    metric="roc_auc",
    bootstraps=1000,
    precomputed_predictions=None,
    disable_tqdm=False,
    **kwargs,
):
    """Score one maximum-probability window per stay for the mortality task.

    ROC and PR curves retain the classifier's saved return convention. Confidence
    intervals are computed by the paper's separate stay-bootstrap exporter.
    """
    metrics = {"roc_auc", "average_precision", "roc_curve", "pr_curve"}
    if metric not in metrics:
        raise ValueError("Unsupported paper classification metric")
    if threshold is not None:
        raise ValueError("The paper scorer uses maximum probability per stay")
    if isinstance(y, pd.DataFrame) and "prediction" in y.columns:
        precomputed_predictions = y["prediction"].values
        y = y.drop("prediction", axis=1)
    if isinstance(precomputed_predictions, (np.ndarray, torch.Tensor)):
        predictions = precomputed_predictions.flatten()
    elif isinstance(precomputed_predictions, list):
        predictions = np.concatenate(precomputed_predictions).flatten()
    else:
        predictions = estimator.predict_proba(X.astype(float))[:, 1]
    warnings = pd.DataFrame(
        dict(
            ids=y.index.get_level_values("ids").values,
            times=y.index.get_level_values("times").values,
            y_true=y.values.flatten(),
            y_pred=predictions,
            onset=y.index.get_level_values("onset").values,
        )
    ).sort_values(["ids", "times"])
    selected = warnings.groupby("ids")["y_pred"].idxmax().values
    grouped = warnings.loc[selected].reset_index(drop=True)
    if len(grouped) != warnings.ids.nunique():
        raise ValueError("Classification aggregation lost a stay")
    if metric == "roc_auc":
        return roc_auc_score(grouped.y_true, grouped.y_pred)
    if metric == "average_precision":
        return average_precision_score(grouped.y_true, grouped.y_pred)
    if metric == "roc_curve":
        return roc_curve(grouped.y_true, grouped.y_pred)
    precision, recall, thresholds = precision_recall_curve(
        grouped.y_true, grouped.y_pred
    )
    return recall, precision, thresholds


def fiddle_collate(batch, fine_tune=True, pad_token_id=None):
    """
    Custom collate function for FIDDLE compatible datasets with ragged tensor support.

    Handles variable-length sequences by applying padding only at collation time,
    significantly reducing memory usage compared to storing pre-padded tensors.

    Args:
        batch: List of dataset items
        fine_tune: Whether this is for fine-tuning (supervised) or pretraining
        pad_token_id: The padding token ID from the tokenizer (defaults to 0 if None)

    Returns:
        For fine_tune=True (Early Warning/Supervised):
            X_values: (B, T, F) - padded time series features
            static: (B, S) - static features
            y: (B, L) - labels
            times: (B, T) - padded time indices
            onset: (B,) - onset times
            ids: (B,) - patient/stay IDs

        For fine_tune=False (Self-supervised):
            X_source: (B, T, F) - padded source sequences
            X_target: (B, T, F) - padded target sequences
            statics: (B, S) - static features
    """

    # Use proper pad token or fallback to 0
    padding_value = pad_token_id if pad_token_id is not None else 0.0

    if fine_tune:
        # Fine-tuning mode (supervised learning / early warning)
        X_values, static, y, times, onset, ids = zip(*batch)

        # Handle variable-length sequences by padding
        if isinstance(X_values[0], torch.Tensor) and X_values[0].dim() >= 2:
            # Check if we have sequence data or scalar data
            if all(hasattr(t, "shape") and len(t.shape) == 0 for t in times):
                # Scalar mode: times, onset, ids are scalars (growing_window mode)
                X_values_padded = pad_sequence(
                    X_values, batch_first=True, padding_value=padding_value
                )
                times_padded = torch.stack(times, dim=0)  # Stack scalars into 1D tensor
            else:
                # Sequence mode: times, onset, ids are sequences (sliding_window mode)
                # Debug: Check for empty sequences
                def get_seq_len(seq):
                    if hasattr(seq, "shape"):
                        return seq.shape[0] if len(seq.shape) > 0 else 1
                    elif hasattr(seq, "__len__"):
                        return len(seq)
                    else:
                        return 0

                empty_X_indices = [
                    i for i, seq in enumerate(X_values) if get_seq_len(seq) == 0
                ]
                empty_times_indices = [
                    i for i, seq in enumerate(times) if get_seq_len(seq) == 0
                ]

                if empty_X_indices or empty_times_indices:
                    print(
                        f"Warning: Found empty sequences - X_values: {empty_X_indices}, times: {empty_times_indices}"
                    )
                    # Filter out empty sequences
                    valid_indices = [
                        i
                        for i in range(len(X_values))
                        if get_seq_len(X_values[i]) > 0 and get_seq_len(times[i]) > 0
                    ]

                    if not valid_indices:
                        raise ValueError("All sequences in batch are empty!")

                    # Keep only valid sequences
                    X_values = [X_values[i] for i in valid_indices]
                    static = [static[i] for i in valid_indices]
                    y = [y[i] for i in valid_indices]
                    times = [times[i] for i in valid_indices]
                    onset = [onset[i] for i in valid_indices]
                    ids = [ids[i] for i in valid_indices]

                # Pad sequences to max length in batch
                X_values_padded = pad_sequence(
                    X_values, batch_first=True, padding_value=padding_value
                )
                times_padded = pad_sequence(
                    times, batch_first=True, padding_value=0.0
                )  # Times use 0.0 padding
        else:
            # Handle case where sequences are already uniform (fallback)
            X_values_padded = torch.stack(X_values, dim=0)
            times_padded = torch.stack(times, dim=0)

        # Stack static features and other uniform tensors
        static = torch.stack(static, dim=0)
        onset = torch.stack(onset, dim=0)
        ids = torch.stack(ids, dim=0)

        # Handle labels: check if they are sequences (full_sequence mode) or scalars
        # Check if ANY label in the batch is a sequence (not just the first one)
        has_sequences = any(
            isinstance(label, torch.Tensor) and label.dim() > 0 and len(label) >= 1
            for label in y
        )

        if has_sequences:
            # Variable-length label sequences (full_sequence mode) - need padding
            y = pad_sequence(y, batch_first=True, padding_value=0.0).float()
        else:
            # Uniform labels (sliding_window mode or single labels) - can stack directly
            y = torch.stack(y, dim=0).float()

        return X_values_padded, static, y, times_padded, onset, ids

    else:
        # Self-supervised pretraining mode
        X_source, X_target, statics = zip(*batch)

        # Pad source and target sequences separately
        if isinstance(X_source[0], torch.Tensor) and X_source[0].dim() >= 2:
            X_source_padded = pad_sequence(
                X_source, batch_first=True, padding_value=padding_value
            )
            X_target_padded = pad_sequence(
                X_target, batch_first=True, padding_value=-100
            )  # -100 for loss masking
        else:
            # Fallback for uniform sequences
            X_source_padded = torch.stack(X_source, dim=0)
            X_target_padded = torch.stack(X_target, dim=0)

        # Stack static features
        statics = torch.stack(statics, dim=0)

        return X_source_padded, X_target_padded, statics
