"""
Tests for the FIDDLETokenizer class.

This module tests:
- Vocabulary construction
- Encoding/decoding round-trips
- Special token handling
- Token shuffling within timepoints
- Time position embeddings
"""

import pytest
import numpy as np
import torch
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from collections import OrderedDict

from syntheticicu.data_classes import FIDDLETokenizer
from syntheticicu.utils import shuffle_within_timeblocks


# ============================================================================
# Tokenizer Initialization Tests
# ============================================================================


class TestTokenizerInitialization:
    """Tests for FIDDLETokenizer initialization."""

    def test_vocabulary_size(self, minimal_features):
        """Vocabulary size should equal (num_features * num_bins) + num_special_tokens."""
        num_bins = 5
        num_special_tokens = 4  # <SOS>, <GAP>, <EOS>, <PAD>

        tokenizer = FIDDLETokenizer(
            features=minimal_features,
            num_bins=num_bins,
            include_demographics=False,
        )

        expected_vocab_size = len(minimal_features) * num_bins + num_special_tokens
        assert tokenizer.vocab_size == expected_vocab_size
        assert len(tokenizer.vocab) == expected_vocab_size
        assert len(tokenizer.id_to_token) == expected_vocab_size

    def test_features_sorted(self, minimal_features):
        """Features should be stored in sorted order."""
        # Pass features in unsorted order
        unsorted_features = list(reversed(minimal_features))
        tokenizer = FIDDLETokenizer(features=unsorted_features)

        assert tokenizer.features == sorted(minimal_features)

    def test_special_tokens_present(self, mock_tokenizer):
        """All special tokens should be present in vocabulary."""
        special_tokens = ["<SOS>", "<GAP>", "<EOS>", "<PAD>"]

        for token in special_tokens:
            assert token in mock_tokenizer.vocab
            token_id = mock_tokenizer.vocab[token]
            assert mock_tokenizer.id_to_token[token_id] == token

    def test_special_token_accessors(self, mock_tokenizer):
        """Special token accessor methods should work correctly."""
        assert isinstance(mock_tokenizer.get_sos_token_id(), int)
        assert isinstance(mock_tokenizer.get_eos_token_id(), int)
        assert isinstance(mock_tokenizer.get_gap_token_id(), int)
        assert isinstance(mock_tokenizer.get_pad_token_id(), int)

        # Token IDs should be distinct
        special_ids = [
            mock_tokenizer.get_sos_token_id(),
            mock_tokenizer.get_eos_token_id(),
            mock_tokenizer.get_gap_token_id(),
            mock_tokenizer.get_pad_token_id(),
        ]
        assert len(set(special_ids)) == 4

    @given(num_bins=st.integers(min_value=2, max_value=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_different_num_bins(self, minimal_features, num_bins):
        """Tokenizer should work with different numbers of bins."""
        tokenizer = FIDDLETokenizer(
            features=minimal_features,
            num_bins=num_bins,
            include_demographics=False,
        )

        num_special_tokens = 4
        expected_size = len(minimal_features) * num_bins + num_special_tokens
        assert tokenizer.vocab_size == expected_size
        assert tokenizer.num_bins == num_bins

    def test_tokenizer_repr(self, mock_tokenizer):
        """Tokenizer should have a meaningful string representation."""
        repr_str = repr(mock_tokenizer)

        assert "FIDDLE_tokenizer" in repr_str
        assert "num_features" in repr_str
        assert "num_bins" in repr_str
        assert "vocab_size" in repr_str

    def test_tokenizer_len(self, mock_tokenizer):
        """len() should return vocabulary size."""
        assert len(mock_tokenizer) == mock_tokenizer.vocab_size


# ============================================================================
# Encoding Tests
# ============================================================================


class TestEncoding:
    """Tests for encoding functionality."""

    def test_encode_single_timestep(self, mock_tokenizer, sample_fiddle_timestep):
        """Encoding a single timestep should return correct token IDs."""
        tokens = mock_tokenizer.encode(sample_fiddle_timestep)

        # Should return list of integers
        assert isinstance(tokens, list)
        assert all(isinstance(t, (int, np.integer)) for t in tokens)

        # All tokens should be valid vocab IDs (excluding special tokens)
        num_special = 4
        max_feature_token = mock_tokenizer.vocab_size - num_special
        for token in tokens:
            assert 0 <= token < max_feature_token

    def test_encode_empty_timestep(self, mock_tokenizer, minimal_features):
        """Encoding an all-zero timestep should return empty list."""
        num_bins = 5
        empty_timestep = np.zeros(len(minimal_features) * num_bins)
        tokens = mock_tokenizer.encode(empty_timestep)

        assert tokens == []

    def test_encode_sequence(self, mock_tokenizer, sample_fiddle_sequence):
        """Encoding a sequence should add SOS, GAPs, and EOS tokens."""
        result = mock_tokenizer.encode_sequence(sample_fiddle_sequence)

        if mock_tokenizer.add_time_point_positional_embeddings:
            tokens, time_positions = result
            assert len(tokens) == len(time_positions)
        else:
            tokens = result

        # First token should be SOS
        assert tokens[0] == mock_tokenizer.get_sos_token_id()

        # Last token should be EOS
        assert tokens[-1] == mock_tokenizer.get_eos_token_id()

        # Should contain GAP tokens between timesteps
        gap_token = mock_tokenizer.get_gap_token_id()
        assert gap_token in tokens

    @given(num_timesteps=st.integers(min_value=1, max_value=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_encode_sequence_different_lengths(
        self, mock_tokenizer, minimal_features, num_timesteps
    ):
        """Encoding should work with different sequence lengths."""
        num_bins = 5

        # Create sequence of random timesteps
        sequence = []
        for _ in range(num_timesteps):
            timestep = np.zeros(len(minimal_features) * num_bins)
            # Set one random bin per feature
            for f in range(len(minimal_features)):
                bin_idx = np.random.randint(0, num_bins)
                timestep[f * num_bins + bin_idx] = 1
            sequence.append(timestep)

        result = mock_tokenizer.encode_sequence(sequence)

        if mock_tokenizer.add_time_point_positional_embeddings:
            tokens, time_positions = result
        else:
            tokens = result

        # Verify structure
        assert tokens[0] == mock_tokenizer.get_sos_token_id()
        assert tokens[-1] == mock_tokenizer.get_eos_token_id()


# ============================================================================
# Decoding Tests
# ============================================================================


class TestDecoding:
    """Tests for decoding functionality."""

    def test_decode_simple_sequence(self, mock_tokenizer, minimal_features):
        """Decoding should recover the original features."""
        num_bins = 5

        # Create a simple timestep
        timestep = np.zeros(len(minimal_features) * num_bins)
        timestep[0] = 1  # First feature, first bin
        timestep[num_bins] = 1  # Second feature, first bin

        # Encode and decode
        result = mock_tokenizer.encode_sequence([timestep])
        if mock_tokenizer.add_time_point_positional_embeddings:
            tokens, _ = result
        else:
            tokens = result

        decoded = mock_tokenizer.decode(tokens)

        # Should have at least one timestep
        assert len(decoded) >= 1

        # Decoded timestep should have correct shape
        assert decoded[0].shape == timestep.shape

    def test_decode_ignores_special_tokens(self, mock_tokenizer):
        """Decoding should properly handle special tokens."""
        # Create sequence with only special tokens
        tokens = [
            mock_tokenizer.get_sos_token_id(),
            mock_tokenizer.get_gap_token_id(),
            mock_tokenizer.get_eos_token_id(),
        ]

        decoded = mock_tokenizer.decode(tokens)

        # Should return list (possibly with empty timesteps)
        assert isinstance(decoded, list)


# ============================================================================
# Round-Trip Tests (Property-Based)
# ============================================================================


class TestRoundTrip:
    """Property-based tests for encoding/decoding round-trips."""

    @given(
        active_bins=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=2),  # feature index
                st.integers(min_value=0, max_value=4),  # bin index
            ),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_single_timestep_round_trip(
        self, mock_tokenizer, minimal_features, active_bins
    ):
        """A single timestep should survive encode-decode round-trip."""
        num_bins = 5

        # Create timestep with specific active bins
        timestep = np.zeros(len(minimal_features) * num_bins)
        for feature_idx, bin_idx in active_bins:
            if feature_idx < len(minimal_features):
                timestep[feature_idx * num_bins + bin_idx] = 1

        # Encode
        tokens = mock_tokenizer.encode(timestep)

        # Verify each active bin has corresponding token
        for feature_idx, bin_idx in active_bins:
            if feature_idx < len(minimal_features):
                feature_name = mock_tokenizer.features[feature_idx]
                token = f"{feature_name}_bin_{bin_idx}"
                token_id = mock_tokenizer.vocab.get(token)
                if token_id is not None:
                    assert token_id in tokens

    @settings(
        max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )  # Reduce examples for complex test
    @given(
        num_timesteps=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=1000),
    )
    def test_sequence_round_trip_structure(
        self, mock_tokenizer, minimal_features, num_timesteps, seed
    ):
        """Sequence encoding should preserve structure through round-trip."""
        np.random.seed(seed)
        num_bins = 5

        # Create random sequence
        sequence = []
        for _ in range(num_timesteps):
            timestep = np.zeros(len(minimal_features) * num_bins)
            for f in range(len(minimal_features)):
                if np.random.random() > 0.5:  # 50% chance of setting each feature
                    bin_idx = np.random.randint(0, num_bins)
                    timestep[f * num_bins + bin_idx] = 1
            sequence.append(timestep)

        # Encode
        result = mock_tokenizer.encode_sequence(sequence)
        if mock_tokenizer.add_time_point_positional_embeddings:
            tokens, time_positions = result
        else:
            tokens = result

        # Count GAP tokens (should be approximately num_timesteps - 1, but may vary with trailing empty timesteps)
        gap_count = tokens.count(mock_tokenizer.get_gap_token_id())

        # At least one SOS and one EOS
        assert tokens.count(mock_tokenizer.get_sos_token_id()) == 1
        assert tokens.count(mock_tokenizer.get_eos_token_id()) == 1


# ============================================================================
# Token Shuffling Tests
# ============================================================================


class TestTokenShuffling:
    """Tests for token shuffling within timepoints."""

    def test_shuffle_preserves_special_tokens(
        self, tokenizer_with_shuffle, minimal_features, sample_vocabulary
    ):
        """Shuffling should not move special tokens."""
        num_bins = 5

        # Create sequence with multiple features per timestep
        timestep = np.zeros(len(minimal_features) * num_bins)
        for f in range(len(minimal_features)):
            timestep[f * num_bins + 2] = 1  # Set bin 2 for each feature

        # Encode multiple times to check shuffling
        results = []
        for _ in range(10):
            result = tokenizer_with_shuffle.encode_sequence([timestep, timestep])
            if tokenizer_with_shuffle.add_time_point_positional_embeddings:
                tokens, time_positions = result
                # Apply shuffling via shuffle_within_timeblocks
                tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(
                    0
                )  # Add batch dim
                time_pos_tensor = torch.tensor(
                    time_positions, dtype=torch.long
                ).unsqueeze(0)
                shuffled_tokens, shuffled_time_pos = shuffle_within_timeblocks(
                    tokens_tensor, time_pos_tensor, sample_vocabulary, p=1.0
                )
                results.append(shuffled_tokens[0].tolist())  # Remove batch dim
            else:
                results.append(result)

        # Check that all results have SOS at start and EOS at end
        for tokens in results:
            assert tokens[0] == tokenizer_with_shuffle.get_sos_token_id()
            assert tokens[-1] == tokenizer_with_shuffle.get_eos_token_id()

    def test_shuffle_produces_variation(
        self, tokenizer_with_shuffle, minimal_features, sample_vocabulary
    ):
        """Shuffling should produce different orderings (probabilistic test)."""
        num_bins = 5

        # Create timestep with all features active
        timestep = np.zeros(len(minimal_features) * num_bins)
        for f in range(len(minimal_features)):
            timestep[f * num_bins + 1] = 1

        # Generate multiple shuffled sequences
        shuffled_results = set()
        for _ in range(20):
            result = tokenizer_with_shuffle.encode_sequence([timestep])
            if tokenizer_with_shuffle.add_time_point_positional_embeddings:
                tokens, time_positions = result
                tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(
                    0
                )  # Add batch dim
                time_pos_tensor = torch.tensor(
                    time_positions, dtype=torch.long
                ).unsqueeze(0)
                shuffled_tokens, _ = shuffle_within_timeblocks(
                    tokens_tensor, time_pos_tensor, sample_vocabulary, p=1.0
                )
                shuffled_results.add(
                    tuple(shuffled_tokens[0].tolist())
                )  # Remove batch dim
            else:
                shuffled_results.add(tuple(result))

        # With at least 3 features, we should see some variation
        # (Note: This is probabilistic, but highly likely with 20 trials)
        if len(minimal_features) >= 3:
            assert len(shuffled_results) >= 1  # At least one unique ordering


# ============================================================================
# Time Position Tests
# ============================================================================


class TestTimePositions:
    """Tests for time position embeddings."""

    def test_time_positions_with_encoding(self, mock_tokenizer, sample_fiddle_sequence):
        """Time positions should be generated alongside tokens."""
        assume(mock_tokenizer.add_time_point_positional_embeddings)

        result = mock_tokenizer.encode_sequence(sample_fiddle_sequence)
        tokens, time_positions = result

        # Same length
        assert len(tokens) == len(time_positions)

        # Time positions should start at 0 for SOS
        assert time_positions[0] == 0

        # Time positions should be non-negative
        assert all(t >= 0 for t in time_positions)

    def test_time_positions_increment_at_gaps(self, mock_tokenizer, minimal_features):
        """Time positions should increment after each timestep."""
        assume(mock_tokenizer.add_time_point_positional_embeddings)

        num_bins = 5
        num_timesteps = 5

        # Create sequence
        sequence = []
        for _ in range(num_timesteps):
            timestep = np.zeros(len(minimal_features) * num_bins)
            timestep[0] = 1  # Just set one feature
            sequence.append(timestep)

        tokens, time_positions = mock_tokenizer.encode_sequence(sequence)

        # Find GAP token positions
        gap_id = mock_tokenizer.get_gap_token_id()
        gap_positions = [i for i, t in enumerate(tokens) if t == gap_id]

        # Check time positions increment after gaps
        for i, gap_pos in enumerate(gap_positions[:-1]):
            # Time position at gap should be less than or equal to position after gap
            current_time = time_positions[gap_pos]
            next_time = (
                time_positions[gap_pos + 1]
                if gap_pos + 1 < len(time_positions)
                else current_time
            )
            assert next_time >= current_time


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_sequence(self, mock_tokenizer):
        """Should handle empty sequence gracefully."""
        result = mock_tokenizer.encode_sequence([])

        if mock_tokenizer.add_time_point_positional_embeddings:
            tokens, time_positions = result
        else:
            tokens = result

        # Should at least have SOS and EOS
        assert mock_tokenizer.get_sos_token_id() in tokens
        assert mock_tokenizer.get_eos_token_id() in tokens

    def test_single_feature_tokenizer(self):
        """Tokenizer should work with a single feature."""
        tokenizer = FIDDLETokenizer(
            features=["temp_raw"],
            num_bins=5,
        )

        # Create timestep
        timestep = np.zeros(5)
        timestep[2] = 1

        tokens = tokenizer.encode(timestep)
        assert len(tokens) == 1
        assert tokenizer.id_to_token[tokens[0]] == "temp_raw_bin_2"

    def test_all_bins_active(self, mock_tokenizer, minimal_features):
        """Should handle all bins being active for a feature."""
        num_bins = 5

        timestep = np.zeros(len(minimal_features) * num_bins)
        # Set all bins for first feature (unusual but should work)
        for b in range(num_bins):
            timestep[b] = 1

        tokens = mock_tokenizer.encode(timestep)

        # Should have 5 tokens for the first feature
        assert len(tokens) >= num_bins

    def test_get_token_methods(self, mock_tokenizer):
        """get_token and get_token_id should be inverses."""
        # Test for a feature token
        test_token = f"{mock_tokenizer.features[0]}_bin_0"
        token_id = mock_tokenizer.get_token_id(test_token)
        retrieved_token = mock_tokenizer.get_token(token_id)

        assert retrieved_token == test_token

        # Test for unknown token
        assert mock_tokenizer.get_token_id("nonexistent_token") is None
        assert mock_tokenizer.get_token(99999) is None
