"""Functional checks for the opt-in bin-plus-residual generator."""

from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig

from syntheticicu.model_numeric import BinResidualLlama, NumericalBins


@pytest.fixture
def numerical_model():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=7,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    bins = NumericalBins(
        lower=[0, 0, 0, 60, 100, 0, 0],
        upper=[0, 0, 0, 100, 180, 0, 0],
        numerical=[False, False, False, True, True, False, False],
    )
    return BinResidualLlama(config, bins, num_centers=2)


def test_value_roundtrip_and_invalid_bin_alignment(numerical_model):
    bins = numerical_model.bins
    ids = torch.tensor([[1, 3, 4, 2]])
    values = torch.tensor([[float("nan"), 80, 160, float("nan")]])
    residuals = bins.encode(ids, values)
    torch.testing.assert_close(residuals, torch.tensor([[0.0, 0.5, 0.75, 0.0]]))
    decoded = bins.decode(ids, residuals)
    torch.testing.assert_close(decoded[:, 1:3], values[:, 1:3])
    assert torch.isnan(decoded[:, [0, 3]]).all()
    with pytest.raises(ValueError, match="outside"):
        bins.encode(torch.tensor([[3]]), torch.tensor([[120.0]]))
    with pytest.raises(ValueError, match="finite"):
        bins.encode(torch.tensor([[3]]), torch.tensor([[float("nan")]]))


def test_values_affect_future_predictions_without_future_leakage(numerical_model):
    model = numerical_model.eval()
    ids = torch.tensor([[1, 3, 4, 2]])
    residuals = torch.tensor([[0.0, 0.2, 0.3, 0.0]])
    before = model(ids, residuals, torch.tensor([0]), torch.tensor([1]))
    residuals[0, 1] = 0.9
    after = model(ids, residuals, torch.tensor([0]), torch.tensor([1]))
    torch.testing.assert_close(before.logits[:, 0], after.logits[:, 0])
    assert not torch.allclose(before.logits[:, 1:], after.logits[:, 1:])


def test_training_generation_and_checkpoint_roundtrip(numerical_model, tmp_path):
    model = numerical_model
    ids = torch.tensor([[1, 3, 4, 2], [1, 4, 2, 0]])
    residuals = torch.tensor([[0.0, 0.2, 0.8, 0.0], [0.0, 0.7, 0.0, 0.0]])
    labels, centers = torch.tensor([0, 1]), torch.tensor([0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    output = model(ids, residuals, labels, centers, compute_loss=True)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.value_embedding.weight.grad.abs().sum() > 0
    assert model.value_head[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    reference = model(ids, residuals, labels, centers).logits.detach()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(model(ids, residuals, labels, centers).logits, reference)
    generated = model.generate(
        ids[:, :1], residuals[:, :1], labels, centers, max_new_tokens=4
    )
    assert generated["input_ids"].shape == generated["residuals"].shape
    assert generated["input_ids"].shape[1] <= 5
    numerical = model.bins.numerical[generated["input_ids"]]
    values = generated["values"][numerical]
    assert torch.isfinite(values).all()
    assert (values >= model.bins.lower[generated["input_ids"]][numerical]).all()
    assert (values <= model.bins.upper[generated["input_ids"]][numerical]).all()


def test_generation_samples_values_and_stops_at_eos(numerical_model):
    model = numerical_model.eval()

    def force_next(ids, scores):
        scores = torch.full_like(scores, -torch.inf)
        scores[:, 3 if ids.shape[1] == 1 else 2] = 0
        return scores

    result = model.generate(
        torch.tensor([[1]]),
        torch.zeros(1, 1),
        torch.tensor([0]),
        torch.tensor([0]),
        max_new_tokens=5,
        logits_processors=[force_next],
    )
    assert result["input_ids"].tolist() == [[1, 3, 2]]
    assert 60 < result["values"][0, 1] < 100
    assert 0 < result["residuals"][0, 1] < 1


def test_vocabulary_mapping_keeps_demographics_discrete():
    bins = NumericalBins.from_vocabulary(
        {"hr_raw_bin_0": 0, "hr_raw_bin_1": 1, "age_static_bin_0": 2},
        {"hr_raw": [60, 100, 180]},
    )
    assert bins.numerical.tolist() == [True, True, False]
    with pytest.raises(ValueError, match="positive width"):
        NumericalBins.from_vocabulary({"hr_raw_bin_0": 0}, {"hr_raw": [60, 60]})


def test_opt_in_hydra_config_instantiates_without_a_datamodule(numerical_model):
    cfg = OmegaConf.load(Path(__file__).parents[1] / "config/numerical_model.yaml")
    for key in [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
    ]:
        cfg.config[key] = getattr(numerical_model.config, key)
    cfg.gap_token_id = 6
    model = hydra.utils.instantiate(cfg, bins=numerical_model.bins)
    assert isinstance(model, BinResidualLlama)
    assert model.gap_token_id == 6
    assert model.bins.numerical.tolist() == numerical_model.bins.numerical.tolist()


def test_padding_does_not_change_training_loss(numerical_model):
    model = numerical_model.eval()
    ids = torch.tensor([[1, 3, 2]])
    residuals = torch.tensor([[0.0, 0.5, 0.0]])
    labels, centers = torch.tensor([0]), torch.tensor([0])
    original = model(ids, residuals, labels, centers, compute_loss=True)
    padded = model(
        torch.tensor([[1, 3, 2, 0, 0]]),
        torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0]]),
        labels,
        centers,
        compute_loss=True,
    )
    torch.testing.assert_close(original.loss, padded.loss)


def test_no_numerical_targets_and_endpoints_have_finite_loss(numerical_model):
    model = numerical_model.eval()
    labels, centers = torch.tensor([0]), torch.tensor([0])
    output = model(
        torch.tensor([[1, 5, 2]]), torch.zeros(1, 3), labels, centers, compute_loss=True
    )
    assert output.numerical_loss == 0
    endpoints = model(
        torch.tensor([[1, 3, 4, 2]]),
        torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        labels,
        centers,
        compute_loss=True,
    )
    assert torch.isfinite(endpoints.loss)


def test_numerical_loss_predicts_next_value_from_previous_context(numerical_model):
    model = numerical_model.eval()
    ids = torch.tensor([[1, 3, 4, 2]])
    residuals = torch.tensor([[0.0, 0.2, 0.8, 0.0]])
    output = model(
        ids, residuals, torch.tensor([0]), torch.tensor([0]), compute_loss=True
    )
    distribution = model.value_distribution(output.hidden_states[:, :2], ids[:, 1:3])
    expected = -distribution.log_prob(residuals[:, 1:3]).mean()
    torch.testing.assert_close(output.numerical_loss, expected)


def test_time_aware_generation_advances_after_gap(numerical_model):
    model = numerical_model.eval()
    model.gap_token_id = 6

    def force_next(ids, scores):
        scores = torch.full_like(scores, -torch.inf)
        scores[:, {1: 6, 2: 3, 3: 2}[ids.shape[1]]] = 0
        return scores

    result = model.generate(
        torch.tensor([[1]]),
        torch.zeros(1, 1),
        torch.tensor([0]),
        torch.tensor([0]),
        max_new_tokens=5,
        time_positions=torch.zeros(1, 1, dtype=torch.long),
        logits_processors=[force_next],
    )
    assert result["input_ids"].tolist() == [[1, 6, 3, 2]]
    assert result["time_positions"].tolist() == [[0, 0, 1, 1]]


def test_all_dropout_uses_explicit_null_conditions(numerical_model):
    model = numerical_model.train()
    model.outcome_dropout = model.center_dropout = 1.0
    ids, residuals = torch.tensor([[1, 3, 2]]), torch.tensor([[0.0, 0.5, 0.0]])
    dropped = model(ids, residuals, torch.tensor([0]), torch.tensor([0]))
    model.eval()
    explicit_null = model(ids, residuals, torch.tensor([2]), torch.tensor([2]))
    torch.testing.assert_close(dropped.logits, explicit_null.logits)


@pytest.mark.parametrize("invalid", [float("nan"), -0.1, 1.1])
def test_invalid_numerical_context_fails_loudly(numerical_model, invalid):
    with pytest.raises(ValueError):
        numerical_model(
            torch.tensor([[1, 3, 2]]),
            torch.tensor([[0.0, invalid, 0.0]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_full_backbone_bfloat16_long_context_backward(numerical_model):
    config = LlamaConfig(
        vocab_size=7,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=4096,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.33657881463675005,
    )
    model = BinResidualLlama(config, numerical_model.bins, num_centers=2).cuda().train()
    ids = torch.full((32, 3000), 3, dtype=torch.long, device="cuda")
    ids[:, 0], ids[:, -1] = 1, 2
    residuals = torch.full((32, 3000), 0.5, device="cuda")
    residuals[:, 0], residuals[:, -1] = 0, 0
    labels = torch.zeros(32, dtype=torch.long, device="cuda")
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(ids, residuals, labels, labels, compute_loss=True)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert torch.isfinite(model.value_head[-1].weight.grad).all()
        assert model.bins.lower.dtype == torch.float32
    finally:
        torch.use_deterministic_algorithms(deterministic)
