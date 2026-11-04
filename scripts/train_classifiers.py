import gc
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
import ast

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real
from tqdm import tqdm


from syntheticicu.utils import early_warning_scorer
from syntheticicu.data_classes import (
    EarlyWarningDatamodule,
)


def extract_tabular_data_single_pass(args, dataset, augmentation_seed=None):
    """
    Extract tabular data from an EarlyWarningDataset for a single pass.

    Args:
        args: Command line arguments from argparse
        dataset: EarlyWarningDataset instance
        augmentation_seed: Random seed for this augmentation pass

    Returns:
        tuple: (X, y) where X is extracted features (as a DataFrame with feature names as columns), and y is labels
    """
    if augmentation_seed is not None:
        np.random.seed(augmentation_seed)

    print(
        f"Extracting tabular data from {len(dataset)} samples..."
        + (
            f" (augmentation seed: {augmentation_seed})"
            if augmentation_seed is not None
            else ""
        )
    )

    X_list = []
    y_list = []
    static_list = []
    ids_list = []
    times_list = []
    onset_list = []

    for i in tqdm(range(len(dataset)), desc="Extracting samples"):
        try:
            sample = dataset[i]

            if dataset.use_sliding_windows:
                feature_tensor, statics, y, times, onset, ids = sample
                X_list.append(feature_tensor.numpy())  # Shape: (features, window_size)
                static_list.append(statics.numpy())
                y_list.append(y.numpy())
                ids_list.append(ids.item())
                times_list.append(times.item())
                onset_list.append(onset.item())

            else:
                raise NotImplementedError(
                    "Full sequence mode not yet implemented for this extraction"
                )

        except ValueError:
            print(f"Error processing sample {i}: {e}")
            continue

    if not X_list:
        raise ValueError("No valid samples found after filtering")

    X = np.stack(X_list)  # Shape: (n_samples, features, window_size)
    statics = np.stack(static_list)  # Shape: (n_samples, n_static_features)
    y = np.array(y_list)
    ids = np.array(ids_list)
    times = np.array(times_list)
    onset = np.array(onset_list)

    print(f"Collected {len(X)} samples after lookback filtering")

    include_missingness = (
        dataset.use_sliding_windows
        and dataset.include_missing_mask
        and dataset.missingness_tensor is not None
    )

    if include_missingness:
        total_features = X.shape[1]
        n_original_features = len(dataset.features)  # Original FIDDLE features
        n_missingness_features = total_features - n_original_features

        X_original = X[:, :n_original_features, :]  # Original features
        X_missingness = X[:, n_original_features:, :]  # Missingness indicators

        print(
            f"Split features: {n_original_features} original + {n_missingness_features} missingness"
        )
    else:
        X_original = X
        X_missingness = None
        n_original_features = X.shape[1]

    feature_extractor_dict = {
        feature: getattr(np, feature) for feature in args.extracted_features.split("-")
    }

    final_stats = []

    for stat in feature_extractor_dict.keys():
        final_stats.append(
            feature_extractor_dict[stat](X_original, axis=2)
        )  # Shape: (n_samples, n_original_features)

    X = np.concatenate(
        final_stats, axis=1
    )  # Shape: (n_samples, n_stats * n_original_features)

    if include_missingness:
        missingness_counts = np.any(
            X_missingness, axis=2
        )  # Shape: (n_samples, n_missingness_features)

        X = np.concatenate([X, missingness_counts], axis=1)

    if args.include_statics:
        X = np.concatenate([X, statics], axis=1)

    base_feature_names = sorted(dataset.features)

    feature_names = [
        f"{feature}_{aggregator}"
        for aggregator in args.extracted_features.split("-")
        for feature in base_feature_names
    ]

    print(f"{len(feature_names)} statistical features extracted.")

    if include_missingness:
        missingness_feature_names = [
            feat
            for feat in dataset.feature_names_with_missingness
            if feat.endswith("__mask")
        ]

        missingness_names = [f"{feat}_count" for feat in missingness_feature_names]
        feature_names += missingness_names

        print(f"{len(missingness_names)} missingness count features added.")

    if args.include_statics:
        feature_names += dataset.statics_set
        print(f"{len(dataset.statics_set)} static features added.")

    print(f"Total features: {len(feature_names)}")

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]} features "
            f"but {len(feature_names)} feature names provided."
        )

    X = pd.DataFrame(X, columns=feature_names)

    label_name = str(dataset.label) if hasattr(dataset, "label") else "label"
    if isinstance(dataset.label, list):
        label_name = dataset.label[0]  # Take first element if it's a list

    y_series = pd.Series(y.astype(float), name=label_name)
    y_series.index = pd.MultiIndex.from_arrays(
        [ids, times, onset], names=["ids", "times", "onset"]
    )

    return X, y_series


def extract_tabular_data(args, dataset, force_single_pass=False):
    """Extract the fixed clinical windows once, without resampling their clock."""
    if isinstance(args.lookback, tuple) or args.n_augmentations != 1:
        raise ValueError(
            "The paper uses a fixed clinical cutoff and one extraction pass"
        )
    features, labels = extract_tabular_data_single_pass(args, dataset)
    return features, labels, 1


def create_cross_validation_splits(train_X, train_y, train_ids, args):
    """
    Create cross-validation splits based on the training strategy.

    Args:
        train_X: Training features
        train_y: Training labels
        train_ids: Array of training patient IDs
        args: Command line arguments

    Returns:
        list: Cross-validation split indices
    """
    return list(
        StratifiedGroupKFold(
            n_splits=args.n_folds, shuffle=True, random_state=args.seed
        ).split(train_X, train_y, groups=train_ids)
    )


def main(args, datamodule_factory=None, max_parallel_jobs=None):
    """
    Main function to set up the EarlyWarningDatamodule, extract data, tune hyperparameters,
    and evaluate the model.
    """

    print("Setting up EarlyWarningDatamodule...")
    if datamodule_factory is None:
        datamodule_factory = EarlyWarningDatamodule

    test_frac = args.test_fraction
    val_frac = args.val_fraction * (1 - test_frac)  # Val fraction of remaining data
    train_frac = 1 - test_frac - val_frac

    print(
        f"Split fractions - Train: {train_frac:.2f}, Val: {val_frac:.2f}, Test: {test_frac:.2f}"
    )

    try:
        missingness_threshold = float(args.missingness_threshold)
    except ValueError:
        missingness_threshold = args.missingness_threshold
    print(args.sources)
    datamodule = datamodule_factory(
        label=args.label,
        lookback=args.lookback,
        syn_lookback=args.syn_lookback,
        syn_path=args.syn_path,
        real_path=args.real_path,
        n_real_train=args.n_real_train,
        augment_with_all_syn=args.augment_with_all_syn,
        test_on_syn=args.test_on_syn,
        percentage_synthetic=args.percentage_synthetic,
        sources=ast.literal_eval(args.sources) if args.sources is not None else None,
        ids_to_keep=None,  # Could load from args.train_ids/test_ids if needed
        min_length=args.min_length,
        max_length=args.max_length,
        max_gap=args.max_gap,
        min_onset=args.window_size,
        samples=(100 if args.debug else None),
        val_set=args.val_set,
        feature_set=args.data_size,
        batch_size=512,  # Not used for sklearn but required
        num_workers=args.num_workers,
        split_fractions=(train_frac, val_frac, test_frac),
        split_mode=args.split_mode,  # ("stay_id" if train_datasets == test_datasets else "centre"),
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        random_state=args.seed,
        window_size=args.window_size,
        impute_missing=args.impute_missing,
        include_missing_mask=args.include_missing_mask,
        missingness_threshold=missingness_threshold,
    )

    datamodule.setup("fit")

    print(f"Data splits created:")
    print(f"  Training: {len(datamodule.train_dataset)} samples")
    print(f"  Test: {len(datamodule.test_dataset)} samples")

    train_X, train_y, n_train_augmentations = extract_tabular_data(
        args,
        datamodule.train_dataset,
        force_single_pass=False,  # Allow augmentation for training
    )
    test_X, test_y, n_test_augmentations = extract_tabular_data(
        args,
        datamodule.test_dataset,
        force_single_pass=True,  # Force deterministic for testing
    )

    print("Train prevalence:", np.mean(train_y.values))
    print("Test prevalence:", np.mean(test_y.values))

    augmentation_info = {
        "train": n_train_augmentations,
        "test": n_test_augmentations,
    }
    print(
        f"Augmentation passes used - Train: {n_train_augmentations}, Test: {n_test_augmentations}"
    )

    train_ids = train_y.index.get_level_values("ids").values
    test_ids = test_y.index.get_level_values("ids").values

    splits = {
        "train_ids": np.unique(train_ids),
        "test_ids": np.unique(test_ids),
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
    }

    print("Setting up model pipeline and hyperparameter search...")

    if args.linear:
        classifier = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            max_iter=1000,
            class_weight=("balanced" if not args.match_controls else None),
        )
        param_grid = {
            "classifier__C": Real(0.0001, 1000.0, prior="log-uniform"),
        }
    else:
        raise ValueError("The paper utility model is logistic regression")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    cv_splits = create_cross_validation_splits(train_X, train_y, train_ids, args)

    scoring_metric = partial(
        early_warning_scorer,
        metric="roc_auc",
        threshold=None,
        bootstraps=False,
    )

    optimal_n_jobs = min(len(cv_splits), max(1, cpu_count() // 2))
    if max_parallel_jobs is not None:
        optimal_n_jobs = min(optimal_n_jobs, max_parallel_jobs)

    print(f"Running hyperparameter optimization...")

    tuner = BayesSearchCV(
        estimator=model,
        search_spaces=param_grid,
        cv=cv_splits,
        n_iter=args.hopt_iter,
        n_jobs=optimal_n_jobs,
        scoring=scoring_metric,
        refit=True,
    )

    cv_results = tuner.fit(train_X, train_y, groups=train_ids)

    print("Evaluating model performance...")

    performance_dict = defaultdict(lambda: defaultdict(dict))

    performance_dict["hopt_metadata"] = cv_results.cv_results_

    performance_dict["augmentation_info"] = augmentation_info
    performance_dict["n_augmentations"] = (
        args.n_augmentations if isinstance(args.lookback, tuple) else 1
    )

    if args.FIDDLE_encode and hasattr(datamodule.train_dataset, "X_FIDDLE_bins"):
        performance_dict["FIDDLE_X_bins"] = datamodule.train_dataset.X_FIDDLE_bins
        performance_dict["FIDDLE_S_bins"] = datamodule.train_dataset.S_FIDDLE_bins

    for training_samples in ["all"]:
        try:
            sub_X = train_X
            sub_y = train_y
            sub_ids = train_ids

            estimator = cv_results.best_estimator_

            performance_dict[training_samples]["estimator"] = estimator

            def update_performance_dict(
                performance_dict, samples, dataset_name, X, y, estimator
            ):
                """Compute ROC and PR curves for a dataset split."""
                metrics = ["roc_curve", "pr_curve"]
                for metric in metrics:
                    performance_dict[samples][f"{dataset_name}_{metric}"] = (
                        early_warning_scorer(
                            estimator,
                            X,
                            y,
                            threshold=None,
                            metric=metric,
                            bootstraps=args.bootstraps,
                        )
                    )

            update_performance_dict(
                performance_dict, training_samples, "train", sub_X, sub_y, estimator
            )
            update_performance_dict(
                performance_dict, training_samples, "test", test_X, test_y, estimator
            )

            performance_dict[training_samples]["full_train_input"] = sub_X
            performance_dict[training_samples]["full_train_labels"] = sub_y

            test_predictions = estimator.predict_proba(test_X)[:, -1]

            unique_test_ids = np.unique(test_ids)
            per_patient_predictions = []

            for stay_id in unique_test_ids:
                patient_mask = test_ids == stay_id
                patient_preds = test_predictions[patient_mask]
                per_patient_predictions.append(torch.tensor(patient_preds))

            performance_dict[training_samples][
                "full_test_predictions"
            ] = per_patient_predictions
            performance_dict[training_samples]["full_test_input"] = test_X
            performance_dict[training_samples]["full_test_labels"] = test_y

            gc.collect()

        except (ValueError, TypeError) as e:
            print(f"Error with {training_samples} samples: {e}. Stopping evaluation.")
            break

    return dict(performance_dict), splits
