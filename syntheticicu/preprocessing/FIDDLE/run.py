import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import syntheticicu.preprocessing.FIDDLE.config as FIDDLE_config
import syntheticicu.preprocessing.FIDDLE.steps as FIDDLE_steps
from syntheticicu.utils import AttrDict, get_feature_set


def get_fiddle_df(icu_dataset):
    """Get a FIDDLE-compatible set of dataframes from the ICUDataset.

    Args:
        icu_dataset (ICUDataset): ICUDataset object.

    """

    stay_ids = icu_dataset.data_table.stay_id.unique().tolist()
    fiddle_df = pd.DataFrame()
    max_length = icu_dataset.max_length

    def process_stay(stay_id, stay, statics_set):

        X, statics = stay[0], stay[1]
        # Convert to NumPy arrays for faster processing
        X_np = X.numpy()
        statics_np = statics.numpy()
        # Preallocate memory for the dynamic part (observations)
        num_obs = X_np.shape[0]
        ids = np.full(num_obs, stay_id)  # Stay ID repeated for all observations
        ts = X_np[:, 0]  # Time stamps
        variable_names = X_np[:, 2].astype(int)  # Variable names as integers
        variable_values = X_np[:, 1]  # Variable values
        # Preallocate memory for the static part (statics_set)
        num_statics = len(statics_set)
        static_ids = np.full(num_statics, stay_id)
        static_ts = np.full(num_statics, np.nan)  # Null timestamps
        static_variable_names = np.array(statics_set)
        static_variable_values = statics_np

        # Concatenate the dynamic and static parts
        ids = np.concatenate([ids, static_ids])
        ts = np.concatenate([ts, static_ts])
        variable_names = np.concatenate([variable_names, static_variable_names])
        variable_values = np.concatenate([variable_values, static_variable_values])

        return ids, ts, variable_names, variable_values

    # Run in parallel
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(process_stay)(stay_id, stay, icu_dataset.statics_set)
        for stay_id, stay in enumerate(
            tqdm(icu_dataset, desc="FIDDLE: formatting ICU stays", total=len(stay_ids))
        )
    )

    # Unpack results
    ids = []
    ts = []
    variable_names = []
    variable_values = []

    for result in tqdm(results, desc="Unpacking results"):
        ids.extend(result[0])
        ts.extend(result[1])
        variable_names.extend(result[2])
        variable_values.extend(result[3])

    fiddle_df["ID"] = np.array(ids).astype(int)
    fiddle_df["t"] = np.array(ts)
    fiddle_df["variable_name"] = np.array(variable_names).astype(str)
    fiddle_df["variable_value"] = variable_values

    var_dict = {
        **{str(i): name for i, name in enumerate(sorted(get_feature_set("all")))},
        **{name: name for name in icu_dataset.statics_set},
    }
    fiddle_df.variable_name = fiddle_df.variable_name.map(var_dict)
    pop_df = pd.DataFrame(data={"ID": fiddle_df.ID.unique()})

    # Remove time series features with not enough unique values
    unique_values = fiddle_df.groupby("variable_name")["variable_value"].nunique()
    unique_values = unique_values[unique_values > 2]
    fiddle_df = fiddle_df[
        (fiddle_df.variable_name.isin(unique_values.index))
        | (fiddle_df.variable_name.isin(icu_dataset.statics_set))
    ]

    return fiddle_df, pop_df


def run_fiddle(
    icu_dataset,
    T=None,
    dt=1,
    theta_1=0.001,
    theta_2=0.001,
    theta_freq=1.0,
    stats_functions=["min", "max", "mean"],
    X_discretization_bins=None,
    S_discretization_bins=None,
    N=None,
    Ds=None,
    prefilter=False,
    postfilter=False,
    verbose=False,
    save_output=False,
    batch_size=None,
):
    """Run FIDDLE on the given data.

    Args:
        icu_dataset (ICUDataset): ICUDataset object.
        T (int): Time horizon. Defaults to the maximum observed in the dataset.
        dt (int, optional): Time step. Defaults to 1.
        output_dir (str): Output directory.
        config_fname (str): Configuration file name.
        theta_1 (float, optional): Threshold for the pre-filter step. Defaults to 0.001.
        theta_2 (float, optional): Threshold for the post-filter step. Defaults to 0.001.
        theta_freq (float, optional): Threshold specifying the threshold, in terms of the average number of measurements per time window, at which we deem a variable “frequent” (for which summary statistics will be calculated). Defaults to 1.0.
        stats_functions (list, optional): List of summary statistics take for time varying features. Defaults to ["min", "max", "median"].
        X_discretization_bins (dict, optional): Dictionary containing the discretization bins for time dependent features. Defaults to None (computed on the selected data).
        S_discretization_bins (dict, optional): Dictionary containing the discretization bins for time invariant features. Defaults to None (computed on the selected data).
        N (int, optional): Number of samples. Defaults to None.
        Ds (dict, optional): Dictionary containing the discretization bins. Defaults to None.
        prefilter (bool, optional): Whether to pre-filter the data. Defaults to False.
        postfilter (bool, optional): Whether to post-filter the data. Defaults to False.
        verbose (bool, optional): Whether to print the output. Defaults to False.
        save_output (bool, optional): Whether to save the output. Defaults to False.
        batch_size (int, optional): Batch size. Defaults to None (loaded from the config file).

    """
    config = FIDDLE_config.load_config(None)

    # Get required formatted data frames
    df_data, df_population = get_fiddle_df(icu_dataset)

    # Settings
    hierarchical_sep = config.get("hierarchical_sep", ":")
    hierarchical_levels = config.get("hierarchical_levels", [])
    value_type_override = config.get("value_types", {})

    discretize = config.get("discretize", True)
    use_ordinal_encoding = config.get("use_ordinal_encoding", False)

    parallel = config.get("parallel", False)
    n_jobs = config.get("n_jobs", 1)
    if batch_size is None:
        batch_size = config.get("batch_size", 100)

    N = N or len(df_population)
    df_population = df_population.iloc[:N]
    if T is None:
        T = icu_dataset.lengths.max().item()
    L = int(np.floor(T / dt))

    if verbose:
        print("Running FIDDLE preprocessing...")
        print()
        print("Input arguments:")
        print("    {:<6} = {}".format("T", T))
        print("    {:<6} = {}".format("dt", dt))
        print("    {:<6} = {}".format("\u03b8\u2081", theta_1))
        print("    {:<6} = {}".format("\u03b8\u2082", theta_2))
        print("    {:<6} = {}".format("\u03b8_freq", theta_freq))
        print("    {:<6} = {} {}".format("k", len(stats_functions), stats_functions))
        print()
        print("{} = {}".format("discretize", {False: "no", True: "yes"}[discretize]))
        if discretize:
            print(
                "    S discretization bins:",
                S_discretization_bins or "to be computed from data",
            )
            print(
                "    X discretization bins:",
                X_discretization_bins or "to be computed from data",
            )
        print()
        print("N = {}".format(N))
        print("L = {}".format(L))
        print("", flush=True)

    # Main
    args = AttrDict(
        {
            "T": T,
            "df_data": df_data,
            "df_population": df_population,
            "dt": dt,
            "theta_1": theta_1,
            "theta_2": theta_2,
            "theta_freq": theta_freq,
            "stats_functions": stats_functions,
            "N": N,
            "Ds": Ds,
            "L": L,
            "discretize": discretize,
            "use_ordinal_encoding": use_ordinal_encoding,
            "X_discretization_bins": X_discretization_bins,
            "S_discretization_bins": S_discretization_bins,
            "hierarchical_sep": hierarchical_sep,
            "hierarchical_levels": hierarchical_levels,
            "value_type_override": value_type_override,
            "parallel": parallel,
            "n_jobs": n_jobs,
            "output_dir": (save_output if save_output else None),
            "prefilter": prefilter,
            "postfilter": postfilter,
            "verbose": verbose,
            "batch_size": batch_size,
        }
    )

    if verbose:
        print("N jobs:", n_jobs)

    if prefilter:
        if verbose:
            FIDDLE_steps.print_header("1) Pre-filter")
        df_data = FIDDLE_steps.pre_filter(df_data, theta_1, df_population, args)

    if verbose:
        FIDDLE_steps.print_header("2) Transform; 3) Post-filter")
    df_data, df_types = FIDDLE_steps.parse_variable_data_type(df_data, args)
    df_time_invariant, df_time_series = FIDDLE_steps.split_by_timestamp_type(df_data)

    # Process time-invariant data
    (
        S,
        S_feature_names,
        S_feature_aliases,
        S_discretization_bins,
    ) = FIDDLE_steps.process_time_invariant(df_time_invariant, args)

    # Process time-dependent data
    (
        X,
        X_feature_names,
        X_feature_aliases,
        X_discretization_bins,
    ) = FIDDLE_steps.process_time_dependent(df_time_series, df_types, args)
    X, S = X.todense(), S.todense()

    # Assert that expected output shapes are correct
    assert X.shape[0] == S.shape[0]
    assert X.shape[2] % 5 == 0
    assert S.shape[1] == 5 * len(
        [static for static in icu_dataset.statics_set if "male" not in static]
    ) + 2 * (len([static for static in icu_dataset.statics_set if "male" in static]))

    if X.shape[2] != 5 * len(icu_dataset.features):
        warnings.warn(
            "Number of features in the output does not match the expected number of variables. Adding zero-padded missing features."
        )
        # Add missing features in the right order matching the original dataset
        missing_features = set(icu_dataset.features) - set(
            df_data.variable_name.unique()
        )
        X_full = np.zeros((X.shape[0], X.shape[1], 5 * len(icu_dataset.features)))
        for i, feature in enumerate(icu_dataset.features):
            if feature in missing_features:
                continue
            idx = [
                var for var in icu_dataset.features if var not in missing_features
            ].index(feature)
            X_full[:, :, 5 * i : 5 * (i + 1)] = X[:, :, 5 * idx : 5 * (idx + 1)]
        X = X_full

        assert X.shape[2] == 5 * len(icu_dataset.features)

    return (
        X,
        S,
        X_feature_names,
        S_feature_names,
        X_discretization_bins,
        S_discretization_bins,
    )
