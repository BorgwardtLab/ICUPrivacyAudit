"""
Shared test fixtures and configuration for the SyntheticICU test suite.
"""

import pytest
import torch
import numpy as np
from collections import OrderedDict
from unittest.mock import MagicMock, patch
import tempfile
import os


# ============================================================================
# Hypothesis Configuration
# ============================================================================

from hypothesis import settings, Verbosity

settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("debug", max_examples=5, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# ============================================================================
# Feature Fixtures
# ============================================================================


@pytest.fixture
def vitals_features():
    """Standard vitals feature set."""
    return [
        "temp_raw",
        "hr_raw",
        "resp_raw",
        "sbp_raw",
        "dbp_raw",
        "map_raw",
        "o2sat_raw",
        "urine_raw",
    ]


@pytest.fixture
def labs_features():
    """Standard labs feature set (subset for testing)."""
    return [
        "crea_raw",
        "lact_raw",
        "bili_raw",
        "glu_raw",
        "k_raw",
    ]


@pytest.fixture
def minimal_features():
    """Minimal feature set for fast testing."""
    return ["temp_raw", "hr_raw", "resp_raw"]


@pytest.fixture
def statics_set():
    """Standard static features."""
    return [
        "female_static",
        "age_static",
        "height_static",
        "weight_static",
    ]


# ============================================================================
# Tokenizer Fixtures
# ============================================================================


@pytest.fixture
def mock_tokenizer(minimal_features):
    """Create a FIDDLE tokenizer for testing (without demographics for simpler tests)."""
    from syntheticicu.data_classes import FIDDLETokenizer

    return FIDDLETokenizer(
        features=minimal_features,
        num_bins=5,
        shuffle_tokens_within_timepoint=False,
        add_time_point_positional_embeddings=True,
        label=None,
        include_demographics=False,  # Disable for simpler test vocab
    )


@pytest.fixture
def tokenizer_with_shuffle(minimal_features):
    """Create a tokenizer with shuffling enabled (without demographics for simpler tests)."""
    from syntheticicu.data_classes import FIDDLETokenizer

    return FIDDLETokenizer(
        features=minimal_features,
        num_bins=5,
        shuffle_tokens_within_timepoint=True,
        add_time_point_positional_embeddings=True,
        label=None,
        include_demographics=False,  # Disable for simpler test vocab
    )


@pytest.fixture
def tokenizer_with_label(minimal_features):
    """Create a tokenizer with label configured (without demographics for simpler tests)."""
    from syntheticicu.data_classes import FIDDLETokenizer

    return FIDDLETokenizer(
        features=minimal_features,
        num_bins=5,
        shuffle_tokens_within_timepoint=False,
        add_time_point_positional_embeddings=True,
        label="death",
        include_demographics=False,  # Disable for simpler test vocab
    )


@pytest.fixture
def tokenizer_with_demographics(minimal_features):
    """Create a tokenizer with demographics enabled."""
    from syntheticicu.data_classes import FIDDLETokenizer

    return FIDDLETokenizer(
        features=minimal_features,
        num_bins=5,
        shuffle_tokens_within_timepoint=False,
        add_time_point_positional_embeddings=True,
        label=None,
        include_demographics=True,
    )


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_fiddle_timestep(minimal_features):
    """Create a sample FIDDLE-encoded timestep (one-hot encoded)."""
    num_features = len(minimal_features)
    num_bins = 5
    timestep = np.zeros(num_features * num_bins)
    # Set bin 2 for first feature (temp_raw)
    timestep[2] = 1
    # Set bin 3 for second feature (hr_raw)
    timestep[1 * num_bins + 3] = 1
    # Set bin 4 for third feature (resp_raw)
    timestep[2 * num_bins + 4] = 1
    return timestep


@pytest.fixture
def sample_fiddle_sequence(sample_fiddle_timestep):
    """Create a sample sequence of FIDDLE-encoded timesteps."""
    # Create 5 timesteps with slight variations
    sequence = []
    for i in range(5):
        timestep = sample_fiddle_timestep.copy()
        # Modify some bins randomly
        sequence.append(timestep)
    return sequence


@pytest.fixture
def sample_vocabulary(minimal_features):
    """Create a sample vocabulary for testing."""
    vocab = OrderedDict()
    idx = 0

    # Add feature-bin tokens
    for feature in sorted(minimal_features):
        for bin_idx in range(5):
            token = f"{feature}_bin_{bin_idx}"
            vocab[token] = idx
            idx += 1

    # Add special tokens
    for token in ["<SOS>", "<GAP>", "<EOS>", "<PAD>"]:
        vocab[token] = idx
        idx += 1

    return vocab


# ============================================================================
# Model Configuration Fixtures
# ============================================================================


@pytest.fixture
def llama_xs_config(sample_vocabulary):
    """Configuration for llama-xs model."""
    return {
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "intermediate_size": 4 * 512,
        "vocab_size": len(sample_vocabulary),
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-05,
        "attn_implementation": "sdpa",
        "bos_token_id": sample_vocabulary["<SOS>"],
        "eos_token_id": sample_vocabulary["<EOS>"],
        "pad_token_id": sample_vocabulary["<PAD>"],
    }


@pytest.fixture
def llama_s_config(sample_vocabulary):
    """Configuration for llama-s model."""
    return {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 4 * 768,
        "vocab_size": len(sample_vocabulary),
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-05,
        "attn_implementation": "sdpa",
        "bos_token_id": sample_vocabulary["<SOS>"],
        "eos_token_id": sample_vocabulary["<EOS>"],
        "pad_token_id": sample_vocabulary["<PAD>"],
    }


# ============================================================================
# Mock Dataset Fixtures
# ============================================================================


@pytest.fixture
def mock_icu_data(minimal_features, statics_set):
    """Create mock ICU data for testing without loading real files."""
    num_patients = 10
    max_length = 20

    # Create mock X tensors (ragged format)
    X = []
    lengths = []
    times = []
    missing_masks = []

    for i in range(num_patients):
        seq_len = np.random.randint(5, max_length)
        lengths.append(seq_len)

        # Create random feature values
        x = torch.randn(seq_len, len(minimal_features))
        X.append(x)

        # Create time indices
        times.append(torch.arange(seq_len, dtype=torch.float32))

        # Create missing masks (all not missing for simplicity)
        missing_masks.append(
            torch.ones(seq_len, len(minimal_features), dtype=torch.bool)
        )

    # Create statics
    statics = torch.randn(num_patients, len(statics_set))

    # Create labels (stay_id, label, onset)
    y = torch.zeros(num_patients, 3)
    y[:, 0] = torch.arange(num_patients)  # stay_ids
    y[:, 1] = torch.randint(0, 2, (num_patients,))  # binary labels
    y[:, 2] = torch.tensor([l // 2 for l in lengths])  # onset times

    return {
        "X": X,
        "lengths": torch.tensor(lengths),
        "times": times,
        "missing_masks": missing_masks,
        "statics": statics,
        "y": y,
        "features": minimal_features,
        "statics_set": statics_set,
    }


@pytest.fixture
def mock_tokenized_data(mock_icu_data, mock_tokenizer):
    """Create mock tokenized data for testing."""
    X_tokens = []
    X_time_positions = []

    for i in range(len(mock_icu_data["X"])):
        # Create random token sequence
        seq_len = mock_icu_data["lengths"][i].item()
        num_tokens_per_timestep = 3  # Average tokens per timestep

        tokens = [mock_tokenizer.get_sos_token_id()]
        time_positions = [0]

        for t in range(seq_len):
            # Add random feature tokens
            for _ in range(np.random.randint(1, 4)):
                tokens.append(np.random.randint(0, len(mock_tokenizer.vocab) - 4))
                time_positions.append(t + 1)
            # Add GAP token
            tokens.append(mock_tokenizer.get_gap_token_id())
            time_positions.append(t + 1)

        # Remove last GAP and add EOS
        tokens = tokens[:-1] + [mock_tokenizer.get_eos_token_id()]
        time_positions = time_positions[:-1] + [len(time_positions)]

        X_tokens.append(torch.tensor(tokens, dtype=torch.long))
        X_time_positions.append(torch.tensor(time_positions, dtype=torch.long))

    return {
        **mock_icu_data,
        "X_tokens": X_tokens,
        "X_time_positions": X_time_positions,
        "tokenizer": mock_tokenizer,
    }


# ============================================================================
# Batch Fixtures
# ============================================================================


@pytest.fixture
def sample_batch_with_time_positions(mock_tokenized_data):
    """Create a sample batch with time positions (5-element format)."""
    batch_size = 3
    X = mock_tokenized_data["X_tokens"][:batch_size]
    time_positions = mock_tokenized_data["X_time_positions"][:batch_size]
    statics = mock_tokenized_data["statics"][:batch_size]
    times = mock_tokenized_data["times"][:batch_size]
    IDs = mock_tokenized_data["y"][:batch_size]

    return list(zip(X, time_positions, statics, times, IDs))


@pytest.fixture
def sample_batch_without_time_positions(mock_tokenized_data):
    """Create a sample batch without time positions (4-element format)."""
    batch_size = 3
    X = mock_tokenized_data["X_tokens"][:batch_size]
    statics = mock_tokenized_data["statics"][:batch_size]
    times = mock_tokenized_data["times"][:batch_size]
    IDs = mock_tokenized_data["y"][:batch_size]

    return list(zip(X, statics, times, IDs))


# ============================================================================
# Early Warning Dataset Fixtures
# ============================================================================


@pytest.fixture
def sample_early_warning_batch():
    """Create a sample batch for early warning dataset (sliding window format)."""
    batch_size = 4
    window_size = 4
    num_features = 10

    batch = []
    for i in range(batch_size):
        # Feature tensor (F x T format, will be transposed)
        feature_tensor = torch.randn(num_features, window_size)
        # Static features
        static = torch.randn(20)
        # Label
        y = torch.tensor(float(i % 2))
        # Time
        times = torch.tensor(float(i + 5))
        # Onset
        onset = torch.tensor(float(10 + i))
        # ID
        ids = torch.tensor(float(i + 1000))

        batch.append((feature_tensor, static, y, times, onset, ids))

    return batch


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_checkpoint_path(temp_dir):
    """Create a mock checkpoint file path."""
    return os.path.join(temp_dir, "mock_model.ckpt")


# ============================================================================
# Device Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for consistent testing."""
    return torch.device("cpu")
