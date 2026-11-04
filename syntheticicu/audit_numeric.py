"""Likelihood scoring primitives for bin-plus-residual releases."""

import numpy as np
import torch

from syntheticicu.data_numeric import EOS, PAD, SOS


def _lower_tail(values, fraction):
    if not 0 < fraction <= 1 or len(values) == 0:
        raise ValueError(
            "A nonempty score vector and Min-K fraction in (0, 1] required"
        )
    count = max(1, int(len(values) * fraction))
    return float(np.partition(values, count - 1)[:count].mean())


@torch.inference_mode()
def batch_likelihood_scores(model, batch, fraction):
    """Score bin mass and numerical density separately under teacher forcing.

    Numerical Min-K uses density in training-range-normalized feature units:
    log p(residual | bin, history) - log(bin width) + log(feature range).
    This removes arbitrary measurement units. Endpoint density clipping matches
    training. Neither density nor categorical mass is pooled into a joint Min-K.
    """
    if model.training:
        raise ValueError("Likelihood auditing requires model.eval()")
    if not 0 < fraction <= 1:
        raise ValueError("Min-K fraction must lie in (0, 1]")
    output = model(**batch)
    targets = batch["input_ids"][:, 1:]
    token_logp = (
        output.logits[:, :-1]
        .float()
        .log_softmax(-1)
        .gather(-1, targets[..., None])[..., 0]
    )
    numeric = model.bins.numerical[targets]
    numeric_logp = torch.full_like(token_logp, float("nan"))
    density = model.value_distribution(
        output.hidden_states[:, :-1][numeric], targets[numeric]
    )
    residuals = batch["residuals"][:, 1:][numeric].float().clamp(1e-6, 1 - 1e-6)
    numeric_logp[numeric] = density.log_prob(residuals)
    width = model.bins.upper - model.bins.lower
    feature_ranges = _feature_ranges(model)
    normalized = numeric_logp.clone()
    normalized[numeric] += (
        feature_ranges[targets[numeric]] / width[targets[numeric]]
    ).log()
    content = (targets != PAD) & (targets != EOS) & (targets != SOS)
    categorical, numerical = [], []
    for row in range(len(targets)):
        categorical.append(
            _lower_tail(token_logp[row][content[row]].cpu().numpy(), fraction)
        )
        values = normalized[row][numeric[row]].cpu().numpy()
        numerical.append(_lower_tail(values, fraction) if len(values) else float("nan"))
    return {
        "token_logp": token_logp.cpu().numpy(),
        "numeric_logp": numeric_logp.cpu().numpy(),
        "numeric_mask": numeric.cpu().numpy(),
        "categorical_min_k": np.asarray(categorical),
        "numerical_min_k": np.asarray(numerical),
    }


def _feature_ranges(model):
    if not hasattr(model, "audit_feature_ranges"):
        raise ValueError("Attach schema-derived audit_feature_ranges before scoring")
    return model.audit_feature_ranges


def attach_feature_ranges(model, schema):
    """Attach per-token feature ranges explicitly, without guessing bin groups."""
    ranges = torch.ones_like(model.bins.lower)
    for specification in schema["features"].values():
        width = specification["edges"][-1] - specification["edges"][0]
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Feature ranges must be finite and positive")
        ids = [schema["vocabulary"][token] for token in specification["tokens"]]
        ranges[ids] = width
    model.audit_feature_ranges = ranges
