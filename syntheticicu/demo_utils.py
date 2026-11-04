import torch
import pandas as pd
import importlib
import os
import numpy as np

try:
    from sdv.metadata import SingleTableMetadata
except ImportError:
    SingleTableMetadata = None


# defaults per model
DEMO_MODEL_REGISTRY = {
    "tvae": {
        "_target_": "sdv.single_table.TVAESynthesizer",
        "epochs": 300,
        "batch_size": 500,
    },
}


def resolve_demo_model_cfg(gen_demo_cfg: str) -> dict:
    key = gen_demo_cfg.lower()
    if key not in DEMO_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown demo model '{key}'. Choose from: {list(DEMO_MODEL_REGISTRY)}"
        )
    return dict(DEMO_MODEL_REGISTRY[key])


# Functions to process demographics
def demographics_to_prompt_tensor(df: pd.DataFrame, vocab: dict) -> torch.Tensor:
    """
    Convert a DataFrame of demographic features into a token-ID tensor.

    Each row produces a fixed-length prompt:
        [female_token, age_token, height_token, weight_token, <GAP>]

    Args:
        df: DataFrame with columns female_static, age_static, height_static,
            weight_static (integer bin indices).
        vocab: vocabulary dict mapping token strings to IDs.

    Returns:
        LongTensor of shape (N, 5).
    """
    demographic_features = [
        "female_static",
        "age_static",
        "height_static",
        "weight_static",
    ]
    prompts = []
    for _, row in df.iterrows():
        tokens = []
        for feature in demographic_features:
            val = row[feature]
            bin_idx = int(val) if pd.notna(val) else 0
            token_key = f"{feature}_bin_{bin_idx}"
            tokens.append(vocab[token_key])
        tokens.append(vocab["<GAP>"])
        prompts.append(tokens)

    return torch.tensor(prompts, dtype=torch.long)


def sample_sex_prior(n_samples: int, p_female: float = 0.5) -> np.ndarray:
    """
    Sample sex from Bernoulli distribution.

    Args:
        n_samples: Number of samples to generate
        p_female: Probability of female (default: 0.5)

    Returns:
        Array of shape (n_samples,) with 0 (male) or 1 (female)
    """
    return np.random.binomial(1, p_female, size=n_samples)


def compute_sex_prevalence(datamodule) -> float:
    """
    Compute female prevalence from training data.

    Args:
        datamodule: DataModule with train_icu_dataset

    Returns:
        Probability of female in training data
    """
    try:
        indices = datamodule.train_icu_dataset.indices
        dataset = datamodule.train_icu_dataset.dataset
        statics = dataset.statics[indices].float()

        # Find female_static slice using static_names
        static_names = dataset.static_names
        female_start = next(
            i for i, name in enumerate(static_names) if name.startswith("female_static")
        )
        female_onehot = statics[:, female_start : female_start + 2]
        female_binary = female_onehot.argmax(dim=1)
        p_female = (female_binary == 1).float().mean().item()

        print(f"Computed sex prevalence from training data:")
        print(f"Female: {p_female:.1%}")
        print(f"Male:   {1-p_female:.1%}")
        return p_female

    except Exception as e:
        print(f"Warning: Could not compute sex prevalence: {e}")
        print(f"Using default: 50% female")
        return 0.5


def prepare_demographics(outcome: str, dataset_wrapper) -> tuple:
    """
    Extract and preprocess demographic static variables from the dataset.

    Args:
        outcome: name of the outcome column (e.g. 'death').
        dataset_wrapper: dataset wrapper with .dataset.statics and .dataset.y_full.

    Returns:
        x_indices: LongTensor of categorical indices, shape (N, 4).
        y_collapsed: LongTensor of outcome labels, shape (N,).
        df_demo: DataFrame ready for SDV training, with columns
                 [female_static, age_static, height_static, weight_static, outcome].
    """
    indices = dataset_wrapper.indices
    statics = dataset_wrapper.dataset.statics[indices].float()
    y_raw = dataset_wrapper.dataset.y_full[indices, 1]

    if outcome == "death":
        y_collapsed = torch.where(
            y_raw > 0,
            torch.tensor(1, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
        )
    else:
        y_collapsed = y_raw.long()

    dataset = dataset_wrapper.dataset
    tokenizer = dataset.tokenizer
    demographic_bins = tokenizer.demographic_bins  # {"female_static": 2, ...}
    static_names = dataset.static_names

    # Calculate start indices for each feature in the statics tensor
    feature_start = {}
    offset = 0
    for name in static_names:
        if name not in feature_start:
            feature_start[name] = offset
        offset += demographic_bins[name]

    # Extract categorical indices for each demographic feature
    desired_order = ["female_static", "age_static", "height_static", "weight_static"]
    indices_by_feature = {}
    for feat in desired_order:
        start = feature_start[feat]
        n_bins = demographic_bins[feat]
        feat_onehot = statics[:, start : start + n_bins]
        argmax_vals = feat_onehot.argmax(dim=1)
        indices_by_feature[feat] = argmax_vals

    x_indices = torch.stack([indices_by_feature[f] for f in desired_order], dim=1)
    df_demo = pd.DataFrame(x_indices.cpu().numpy(), columns=desired_order)
    df_demo[outcome] = y_collapsed.cpu().numpy()

    return x_indices, y_collapsed, df_demo


def train_demographic_model(
    gen_model_cfg: dict,
    df_train: pd.DataFrame,
    model_path: str = None,
):
    """
    Load a trained tabular synthesizer if a saved file exists; otherwise train
    one from scratch and optionally save it.

    Args:
        gen_model_cfg: plain dict with _target_ and constructor kwargs.
                       Use resolve_demo_model_cfg() before calling this.
        df_train: training DataFrame (output of prepare_demographics).
        model_path: path to save / load the serialised model. Pass None to
            skip persistence entirely.

    Returns:
        A fitted synthesizer instance.
    """
    if SingleTableMetadata is None:
        raise ImportError(
            "SDV is required to fit demographic synthesizers. "
            "Install the project with SDV support before calling fit_demo_model()."
        )

    cfg_dict = dict(gen_model_cfg)
    target_str = cfg_dict.pop("_target_")
    module_path, class_name = target_str.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)

    # Load existing model
    if model_path is not None and os.path.exists(model_path):
        print(f"Loading demographic model from '{model_path}'...")
        synthesizer = cls.load(model_path)
        print("Demographic model loaded.")
        return synthesizer

    # Build metadata
    print("Detecting metadata from training data...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    for col in df_train.columns:
        metadata.update_column(column_name=col, sdtype="categorical")

    # Instantiate + fit
    print(f"Training {class_name} on {len(df_train)} rows...")
    synthesizer = cls(metadata=metadata, **cfg_dict)
    synthesizer.fit(df_train)

    # Save
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        synthesizer.save(model_path)
        print(f"Model saved to '{model_path}'.")

    return synthesizer
