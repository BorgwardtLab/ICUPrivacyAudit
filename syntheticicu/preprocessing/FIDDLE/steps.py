"""
FIDDLE Preprocessing steps
1. Pre-filter
2. Transform
3. Post-filter
"""

import json
import multiprocessing
import time
from itertools import chain

import joblib
from joblib import Parallel, delayed

from syntheticicu.preprocessing.FIDDLE.helpers import *


def pre_filter(df, threshold, df_population, args):
    T = int(args.T)
    theta_1 = args.theta_1
    df_population = args.df_population

    # Remove rows not in population
    if args.verbose:
        print("Remove rows not in population")
    df = df[df["ID"].isin(df_population.index)]

    # Remove rows with t outside of [0, T)
    if args.verbose:
        print("Remove rows with t outside of [0, {}]".format(T))
    df = df[pd.isnull(df[t_col]) | ((0 <= df[t_col]) & (df[t_col] < T))]

    # Data table should not contain duplicate rows with any numerical values
    # Check for inconsistencies
    var_names = [
        v
        for v, ty in args.value_type_override.items()
        if "hierarchical" in ty.lower() or "categorical" in ty.lower()
    ]
    df_tmp = df[~df[var_col].isin(var_names)]
    dups = df_tmp.duplicated(subset=[ID_col, t_col, var_col], keep=False)
    df_dups = df_tmp[dups]
    if any(dups) and any(is_numeric(v) for v in df_dups[val_col] if not pd.isnull(v)):
        print(df_dups.head())
        raise Exception("Inconsistent numerical values recorded")

    # Remove variables that occur too rarely as defined by the threshold
    if args.verbose:
        print("Remove rare variables (<= {})".format(threshold))

    ## Calculate overall occurrence rate of each variable based on IDs
    df_count = calculate_variable_counts(
        df, df_population
    )  # (N x |var|) table of counts
    df_bool = df_count.astype(bool)  # convert counts to boolean

    ## Keep variables that are recorded for more than threshold fraction of IDs
    variables_keep = df_bool.columns[df_bool.mean(axis=0) > threshold]
    df_out = df[df[var_col].isin(variables_keep)]
    assert set(variables_keep) == set(df_out[var_col].unique())

    variables = sorted(df_bool.columns)
    variables_remove = sorted(set(variables) - set(variables_keep))

    if args.verbose:
        print("Total variables     :", len(variables))
        print("Rare variables      :", len(variables_remove))
        print("Remaining variables :", len(variables_keep))
        print("# rows (original)   :", len(df))
        print("# rows (filtered)   :", len(df_out))

    return df_out


def parse_variable_data_type(df_data, args):
    # 1. parse hierarchical values (e.g. ICD codes) into strings
    # 2. automatically detect value types, respecting user override, and set dtypes in DataFrames
    # 3. pre-map duplicated non-numerical values into multiple categorical variables
    output_dir = args.output_dir
    df = df_data
    assert val_col in df.columns
    if args.verbose:
        print_header("*) Detecting and parsing value types", char="-")

    ## 1. Hierarchical values
    var_names = [
        v for v, ty in args.value_type_override.items() if "hierarchical" in ty.lower()
    ]
    if len(var_names) == 0:  # No hierarchical values
        pass
    else:
        if args.verbose:
            print("Parsing hierarchical values")
        for var_name in var_names:
            var_type = args.value_type_override[var_name]
            df_var = df.loc[df[var_col] == var_name, val_col]
            if var_type.lower() == "hierarchical_icd":
                # need to figure out ICD version
                raise NotImplementedError
            elif var_type.lower() == "hierarchical_icd9":
                df_var = df_var.apply(lambda s: map_icd_hierarchy(s, version=9))
            elif var_type.lower() == "hierarchical_icd10":
                df_var = df_var.apply(lambda s: map_icd_hierarchy(s, version=10))
            else:
                df_var = df_var.apply(lambda s: s.split(hierarchical_sep))

            # Assign mapped values back to original df
            df.loc[df[var_col] == var_name, val_col] = df_var

        # Only encode selected levels
        df_nonhier = df[~df[var_col].isin(var_names)]
        df_hier = df[df[var_col].isin(var_names)]
        df_hier_levels = []
        for hier_level in args.hierarchical_levels:
            # encode level if available
            df_hier_level = df_hier.copy()
            df_hier_level[val_col] = df_hier_level[val_col].apply(
                lambda h: h[min(hier_level, len(h))]
            )
            df_hier_levels.append(df_hier_level)
        df_hier_levels = pd.concat(df_hier_levels).drop_duplicates()

        # Combine hierarchical and non-hierarchical data
        df = pd.concat([df_nonhier, df_hier_levels])

    ## 2. Detect value types
    data_types = []

    # Collect the unique values of each variable
    # values_by_variable: dict(variable_name -> [value1, value2, ...])
    d = df[[var_col, val_col]].drop_duplicates().sort_values(by=[var_col, val_col])
    values_by_variable = defaultdict(list)
    for n, v in zip(d[var_col], d[val_col]):
        values_by_variable[n].append(v)

    # Determine type of each variable
    for variable, values in sorted(values_by_variable.items()):
        # Manual override type in config
        if variable in args.value_type_override:
            data_types.append((variable, args.value_type_override[variable]))
            # Force categorical values to be a string
            if args.value_type_override[variable] == "Categorical" and any(
                is_numeric(v) for v in values if not pd.isnull(v)
            ):
                m_var = df[var_col] == variable
                df.loc[m_var, val_col] = df.loc[m_var, val_col].apply(
                    lambda s: "_" + str(s)
                )
        else:
            if len(values) == 1 and pd.isnull(values[0]):
                data_types.append((variable, "None"))
            elif all(is_numeric(v) for v in values if not pd.isnull(v)):
                data_types.append((variable, "Numeric"))
            elif any(is_numeric(v) for v in values if not pd.isnull(v)):
                data_types.append((variable, "Numeric + Categorical"))
            else:
                data_types.append((variable, "Categorical"))

    df_types = pd.DataFrame(data_types, columns=["variable_name", "value_type"])
    df_types[var_col] = df_types[var_col].astype(str)
    df_types = df_types.set_index(var_col)

    if output_dir:
        fpath = output_dir + "value_types.csv"
        df_types.to_csv(fpath, quoting=1)
        if args.verbose:
            print("Saved as:", fpath)

    ## 3. Pre-map duplicated non-numerical values to separate variables
    var_names = [
        v
        for v, ty in data_types
        if "numeric" not in ty.lower() and "none" not in ty.lower()
    ]
    df_non_num = df[df[var_col].isin(var_names)].copy()
    dup_ = df_non_num.duplicated(subset=[ID_col, t_col, var_col], keep=False)
    df_non_num_dup = df_non_num[dup_].copy()
    dup_var_names = df_non_num_dup[var_col].unique()
    df_non_num_dup[var_col] = (
        df_non_num_dup[var_col].astype(str) + ":" + df_non_num_dup[val_col].astype(str)
    )
    df_non_num_dup[val_col] = 1
    df_non_num.loc[dup_, :] = df_non_num_dup
    df.loc[df[var_col].isin(var_names), :] = df_non_num

    return df, df_types["value_type"]


def split_by_timestamp_type(df):
    variables_inv = df[pd.isnull(df[t_col])][
        var_col
    ].unique()  # Invariant variables have t = NULL
    df_time_invariant = df[df[var_col].isin(variables_inv)]
    df_time_series = df[~df[var_col].isin(variables_inv)]

    return df_time_invariant, df_time_series


def process_time_invariant(df_data_time_invariant, args):
    if len(df_data_time_invariant) == 0:
        return None, None, None

    output_dir = args.output_dir
    df_population = args.df_population
    theta_2 = args.theta_2

    ##############
    if args.verbose:
        print_header("2-A) Transform time-invariant data", char="-")
    start_time = time.time()

    ## Create Nxd^ table
    df_time_invariant = transform_time_invariant_table(
        df_data_time_invariant, df_population
    )

    ## Discretize
    S_all, S_all_feature_names, S_discretization_bins = map_time_invariant_features(
        df_time_invariant, args
    )

    if output_dir:
        dir_path = output_dir + "/"
        df_time_invariant[[]].to_csv(dir_path + "S.ID.csv")
        sparse.save_npz(dir_path + "S_all.npz", S_all)
        json.dump(
            list(S_all_feature_names),
            open(dir_path + "S_all.feature_names.json", "w"),
            sort_keys=True,
        )
        json.dump(
            S_discretization_bins, open(dir_path + "S_all.discretization.json", "w")
        )

    if args.verbose:
        print("Time elapsed: %f seconds" % (time.time() - start_time))

    if args.postfilter:
        ##############
        if args.verbose:
            print_header("3-A) Post-filter time-invariant data", char="-")

        ## Filter
        S, S_feature_names, S_feature_aliases = post_filter_time_invariant(
            S_all, S_all_feature_names, theta_2
        )
        if args.verbose:
            print("Time elapsed: %f seconds" % (time.time() - start_time))

        ## Save output
        if args.output_dir:
            if args.verbose:
                print()
                print("Output")
                print("S: shape={}, density={:.3f}".format(S.shape, S.density))

            sparse.save_npz(dir_path + "S.npz", S)

            with open(dir_path + "S.feature_names.json", "w") as f:
                json.dump(list(S_feature_names), f, sort_keys=True)
            with open(dir_path + "S.feature_aliases.json", "w") as f:
                json.dump(S_feature_aliases, f, sort_keys=True)

            if args.verbose:
                print("Total time: %f seconds" % (time.time() - start_time))
                print("", flush=True)

        return S, S_feature_names, S_feature_aliases, S_discretization_bins

    else:
        return S_all, S_all_feature_names, None, S_discretization_bins


def process_time_dependent(df_data_time_series, df_types, args):
    if len(df_data_time_series) == 0:
        return None, None, None

    output_dir = args.output_dir
    theta_2 = args.theta_2

    ##############
    if args.verbose:
        print_header("2-B) Transform time-dependent data", char="-")
    start_time = time.time()

    ## Create NxLxD^ table
    df_time_series, dtypes_time_series = transform_time_series_table(
        df_data_time_series, df_types, args
    )
    if args.verbose:
        print("Time elapsed: %f seconds" % (time.time() - start_time))

    ## Map variables to features
    X_all, X_all_feature_names, X_discretization_bins = map_time_series_features(
        df_time_series, dtypes_time_series, args
    )

    ## Save output
    if output_dir:
        dir_path = output_dir + "/"
        joblib.dump(df_time_series, output_dir + "df_time_series.joblib")
        joblib.dump(dtypes_time_series, output_dir + "dtypes_time_series.joblib")
        df_time_series[[]].to_csv(dir_path + "X.ID,t_range.csv")
        sparse.save_npz(dir_path + "X_all.npz", X_all)
        json.dump(
            list(X_all_feature_names),
            open(dir_path + "X_all.feature_names.json", "w"),
            sort_keys=True,
        )
        json.dump(
            X_discretization_bins, open(dir_path + "X_all.discretization.json", "w")
        )

    if args.verbose:
        print("Time elapsed: %f seconds" % (time.time() - start_time))

    if args.postfilter:
        ##############
        if args.verbose:
            print_header("3-B) Post-filter time-dependent data", char="-")

        if args.verbose:
            print(X_all.shape, X_all.density)

        ## Filter features
        X, X_feature_names, X_feature_aliases = post_filter_time_series(
            X_all, X_all_feature_names, theta_2, args
        )
        if args.verbose:
            print(X.shape, X.density)
            print("Time elapsed: %f seconds" % (time.time() - start_time))

        ## Save output
        if args.output_dir:
            sparse.save_npz(dir_path + "X.npz", X)
            with open(dir_path + "X.feature_names.json", "w") as f:
                json.dump(list(X_feature_names), f, sort_keys=True)
            with open(dir_path + "X.feature_aliases.json", "w") as f:
                json.dump(X_feature_aliases, f, sort_keys=True)

            if args.verbose:
                print()
                print("Output")
                print("X: shape={}, density={:.3f}".format(X.shape, X.density))

        if args.verbose:
            print("Total time: %f seconds" % (time.time() - start_time))
            print("", flush=True)

        return X, X_feature_names, X_feature_aliases, X_discretization_bins
    else:
        return X_all, X_all_feature_names, None, X_discretization_bins


######
# Time-invariant routines
######
def transform_time_invariant_table(df_in, df_population):
    df_in = df_in.copy()

    # Recorded Value (np.nan if not recorded)
    df_value = pd.pivot_table(df_in, val_col, ID_col, var_col, "last", np.nan)
    df_value = df_value.reindex(index=df_population.index, fill_value=np.nan)
    df_value.columns = [str(col) + "_value" for col in df_value.columns]

    return df_value


def map_time_invariant_features(df, args):
    # Categorical -> binary features
    # Numeric -> binary/float-valued features
    discretization_bins = None
    if args.discretize:
        discretization_bins = args.S_discretization_bins
        if discretization_bins is None:
            discretization_bins = [
                compute_bin_edges(df[col], q=5) for col in df.columns
            ]
            discretization_bins = dict(discretization_bins)

        out = [
            smart_qcut_dummify(
                df[col],
                discretization_bins[col],
                use_ordinal_encoding=args.use_ordinal_encoding,
            )
            for col in df.columns
        ]
        time_invariant_features = pd.concat(out, axis=1)
        feature_names_all = time_invariant_features.columns.values
        sdf = time_invariant_features.astype(pd.SparseDtype(int, fill_value=0))
        S_ = sparse.COO(sdf.sparse.to_coo())
    else:
        # Split a mixed column into numeric and string columns
        for col in df.columns:
            col_data = df[col]
            col_is_numeric = [is_numeric(v) for v in col_data if not pd.isnull(v)]
            if not all(col_is_numeric) and any(
                col_is_numeric
            ):  # have mixed type values
                numeric_mask = col_data.apply(is_numeric)
                df[col + "_str"] = df[col].copy()
                df.loc[~numeric_mask, col] = np.nan
                df.loc[numeric_mask, col + "_str"] = np.nan

        out = [smart_dummify_impute(df[col]) for col in df.columns]
        time_invariant_features = pd.concat(out, axis=1)
        feature_names_all = time_invariant_features.columns.values
        sdf = time_invariant_features.astype(pd.SparseDtype(float, fill_value=0))
        S_ = sparse.COO(sdf.sparse.to_coo())

    if args.verbose:
        print()
        print("Output")
        print("S_all, binary features    :\t", S_.shape)

    return S_, feature_names_all, discretization_bins


def post_filter_time_invariant(S_, S_feature_names_all, threshold):
    # Filter features (optional)
    assert S_.shape[1] == len(S_feature_names_all)
    feature_names_0 = S_feature_names_all
    S0 = S_.to_scipy_sparse()

    ## Remove nearly-constant features (with low variance)
    ## a binary feature is removed if =0 (or =1) for >th fraction of examples
    ## i.e., variance <= (th * (1 - th))
    sel_rare = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    S1 = sel_rare.fit_transform(S0)
    feature_names_1 = feature_names_0[sel_rare.get_support()]

    ## Keep only first of pairwise perfectly correlated features
    sel_corr = CorrelationSelector()
    S2 = sel_corr.fit_transform(S1)
    feature_names_2 = feature_names_1[sel_corr.get_support()]
    feature_aliases = sel_corr.get_feature_aliases(feature_names_1)

    S = sparse.COO(S2)
    feature_names = feature_names_2
    assert S.shape[1] == len(feature_names)

    return S, feature_names, feature_aliases


######
# Time-series routines
######
def func_encode_single_time_series(
    i, g, variables, variables_num_freq, T, dt, stats_functions, impute=True
):
    try:
        assert g.index.nunique() == 1
        assert g.index.unique()[0] == i
        # non-frequent
        variables_non = sorted(set(variables) - set(variables_num_freq))

        if len(variables_non) > 0:
            variables_non = sorted(set(variables) - set(variables_num_freq))
            df_j = pivot_event_table(g).reindex(columns=variables_non).sort_index()
            df_values_j = most_recent_values(df_j, variables, T, dt)
            df_out = df_values_j

        if len(variables_num_freq) > 0:
            # frequent
            # we're only producing mask, ffill, and statistics if the data is measured frequently enough
            df_i = pivot_event_table(g).reindex(columns=variables_num_freq).sort_index()
            mask_i = presence_mask(df_i, variables_num_freq, T, dt)
            delta_t_i = get_delta_time(mask_i)
            df_i = impute_ffill(df_i, variables_num_freq, T, dt, mask_i)
            df_stats_i = summary_statistics(
                df_i, variables_num_freq, stats_functions, T, dt
            )
            df_values_i = most_recent_values(df_i, variables, T, dt)
            if impute:
                check_imputed_output(df_values_i)
                check_imputed_output(df_stats_i)

            df_out = df_out.join([mask_i, delta_t_i, df_values_i, df_stats_i])
    except:
        print(i)
        raise Exception(i)
    return i, df_out


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def form_batches_of_examples(df_in, args, batch_size):
    grouped = df_in.set_index(ID_col)
    IDs = list(grouped.index.unique())
    batches_IDs = list(divide_chunks(IDs, batch_size))
    batches = [grouped.loc[chunk] for chunk in batches_IDs]
    return batches, batches_IDs


def process_batch_time_series(batch, batch_IDs, args):
    # batch, batch_IDs, args = first_arg
    variables, variables_num_freq = args.variables, args.variables_num_freq
    out = dict(
        func_encode_single_time_series(
            i,
            batch.loc[i:i],
            variables,
            variables_num_freq,
            args.T,
            args.dt,
            args.stats_functions,
        )
        for i in batch_IDs
    )
    return out


def transform_time_series_table(df_in, df_types, args):
    output_dir = args.output_dir
    theta_freq = args.theta_freq
    stats_functions = args.stats_functions
    N, L = args.N, args.L
    df_population = args.df_population
    parallel = args.parallel

    # Determine all unique variable names
    variables = get_unique_variables(df_in)
    assert df_in[var_col].nunique() == len(variables)

    if args.verbose:
        print("Total variables    :", len(variables))

    # Determine frequent variables -> we'll calculate statistics, mask, and delta_time only on these
    variables_num_freq = get_frequent_numeric_variables(
        df_in, df_types, variables, theta_freq, args
    )

    if args.verbose:
        print("Frequent variables :", list(variables_num_freq))
        print("{} = {}".format("M\u2081", len(variables_num_freq)))
        print("{} = {}".format("M\u2082", len(variables) - len(variables_num_freq)))
        print("{} = {} {}".format("k ", len(stats_functions), stats_functions))

        print()
        print("FIDDLE: transforming each example...")
    args.variables = variables
    args.variables_num_freq = variables_num_freq

    # Encode time series table for each patient
    if args.parallel:
        batches, batches_IDs = form_batches_of_examples(
            df_in, args, batch_size=args.batch_size
        )
        if args.verbose:
            print("Batches of size {}: ".format(args.batch_size), len(batches))
        out = list(
            tqdm(
                Parallel(n_jobs=-1)(
                    delayed(process_batch_time_series)(batch, batch_id, args)
                    for batch, batch_id, args in zip(
                        batches, batches_IDs, [args] * len(batches)
                    )
                ),
                total=len(batches),
                desc="FIDDLE: transforming time series by batch",
            )
        )

        out = dict((key, d[key]) for d in out for key in d)
        if args.verbose:
            print()
            print("Parallel processing done", flush=True)

    else:
        grouped = list(df_in.groupby(ID_col))
        out = dict(
            func_encode_single_time_series(
                i,
                g.set_index(ID_col),
                variables,
                variables_num_freq,
                args.T,
                args.dt,
                args.stats_functions,
            )
            for i, g in tqdm(
                grouped[:N], desc="FIDDLE: transforming time series encoding"
            )
        )

    df_original = out[0]
    df_copy = pd.DataFrame().reindex_like(df_original)
    # function call
    for i, j in df_original.dtypes.items():
        if i.endswith("_mask"):
            assert j == bool
            df_copy[i] = False
            df_copy[i] = df_copy[i].astype(bool)
        if i.endswith("_delta_time"):
            df_copy[i] = 0
            df_copy[i] = df_copy[i].astype(int)
        if j == "object":
            df_copy[i] = df_copy[i].astype("object")

    for ID in tqdm(df_population.index.values[:N]):
        out.setdefault(ID, df_copy.copy())

    # out = {ID: out[ID] for ID in df_population.index.values[:N]}
    assert len(out) == N
    D_timeseries = out

    # Check each example have identical LxD table structure
    ID0 = sorted(D_timeseries.keys())[0]
    df0 = D_timeseries[ID0]
    for ID, df_i in D_timeseries.items():
        pd.testing.assert_index_equal(df_i.index, df0.index)
        pd.testing.assert_index_equal(df_i.columns, df0.columns)

    D_timeseries = out
    D_ = len(list(D_timeseries.values())[0].columns)

    # Function to generate index pairs for a single item
    def generate_index(ID, df_):
        return [(ID, t_) for t_ in list(df_.index)]

    sorted_keys = sorted(D_timeseries.keys())
    # Parallelize the process
    index = list(
        chain.from_iterable(
            Parallel(n_jobs=1)(
                delayed(generate_index)(ID, D_timeseries[ID])
                for ID in tqdm(sorted_keys, desc="Generating index pairs")
            )
        )
    )

    # index = pd.Index(index, names=['ID', 't_range'])
    index = pd.MultiIndex.from_tuples(index, names=["ID", "t_range"])
    assert len(index) == N * L

    ## Assume all dataframes have the same columns, used after concatenation
    columns = list(sorted(D_timeseries.items())[0][1].columns)
    columns = np.array(columns)

    dtypes = sorted(D_timeseries.items())[0][1].dtypes
    # dtypes = D_timeseries[0].dtypes

    # parallelized version of the above just to speed up, and also removed saving of ID since we never need it
    def process_entry(df_):
        return df_.to_numpy()

    # Parallelize the process
    feature_values = Parallel(n_jobs=-1, backend="threading")(
        delayed(process_entry)(D_timeseries[ID])
        for ID in tqdm(sorted_keys, desc="Concatenating all dfs")
    )
    time_series = np.array(
        list(chain(*feature_values))
    )  # dont need to convert to numpy array once concatenated since we are transforming this into a time series anyways

    assert time_series.shape == (len(index), len(columns))

    df_time_series = pd.DataFrame(data=time_series, index=index, columns=columns)

    # Print metadata
    if args.verbose:
        print("DONE: Transforming each example...")
    ## Freq: Count missing entries using mask
    ts_mask = df_time_series[[col for col in df_time_series if col.endswith("_mask")]]
    ts_mask.columns = [col.replace("_mask", "") for col in ts_mask.columns]
    if args.verbose:
        print(
            "(freq) number of missing entries :\t",
            "{} out of {}={} total".format(
                (1 - ts_mask).astype(int).sum().sum(),
                "\u00d7".join(str(i) for i in [N, L, ts_mask.shape[1]]),
                ts_mask.size,
            ),
        )

    ## Freq: Count imputed entries using mask and dt
    ts_delta_time = df_time_series[
        [col for col in df_time_series if col.endswith("_delta_time")]
    ]
    ts_delta_time.columns = [
        col.replace("_delta_time", "") for col in ts_delta_time.columns
    ]

    if (
        args.output_dir
    ):  # really only need to do these calculations when theres an output dir set
        imputed = (1 - ts_mask).astype(bool) & (ts_delta_time > 0)

        if args.verbose:
            print(
                "(freq) number of imputed entries :\t",
                "{}".format(imputed.sum().sum(), ts_delta_time.size),
            )

        imputed.sum().rename("count").to_csv(output_dir + "/" + "freq_imputed.csv")

        not_imputed = (1 - ts_mask).astype(bool) & (ts_delta_time == 0)
        if args.verbose:
            print(
                "(freq) number of not imputed entries :\t",
                "{}".format(not_imputed.sum().sum(), ts_delta_time.size),
            )
        not_imputed.sum().rename("count").to_csv(
            output_dir + "/" + "freq_not_imputed.csv"
        )

    if args.verbose:
        ## Non-Freq: Count missing entries
        non_freq_cols = sorted(
            [c + "_value" for c in set(variables) - set(variables_num_freq)]
        )
        non_freqs = df_time_series[non_freq_cols]
        print(
            "(non-freq) number of missing entries :\t",
            "{} out of {}={} total".format(
                non_freqs.isna().sum().sum(),
                "\u00d7".join(str(i) for i in [N, L, non_freqs.shape[1]]),
                non_freqs.size,
            ),
        )

        print()
        print("(N \u00d7 L \u00d7 ^D) table :\t", (N, L, len(columns)))
    return df_time_series, dtypes


def map_time_series_features(df_time_series, dtypes, args):
    N, L = args.N, args.L

    df_time_series = df_time_series.dropna(axis="columns", how="all").sort_index()

    if args.verbose:
        print("Discretizing features...")
    ts_mask = select_dtype(df_time_series, "mask", dtypes)
    ts_mixed = select_dtype(df_time_series, "~mask", dtypes)
    assert len(ts_mixed.columns) + len(ts_mask.columns) == len(df_time_series.columns)
    ts_feature_mask = ts_mask.astype(int)
    ts_mixed_cols = [ts_mixed[col] for col in ts_mixed.columns]

    if args.verbose:
        print()
    discretization_bins = None
    if args.discretize:
        dtype = int
        if args.verbose:
            print("Processing", len(ts_mixed_cols), "non-boolean variable columns...")

        discretization_bins = args.X_discretization_bins
        if discretization_bins is None:
            if args.verbose:
                print("    Computing bin edges for numeric variables...")

            # Parallel computation using joblib
            discretization_bins = Parallel(n_jobs=-1, backend="threading")(
                delayed(compute_bin_edges)(col_data, q=5)
                for col_data in tqdm(
                    ts_mixed_cols, desc="FIDDLE: mapping features to bin edges"
                )
            )
            discretization_bins = dict(discretization_bins)

        else:
            if args.verbose:
                print("    Usng predetermined bin edges for numeric variables...")

        if args.verbose:
            print("    Discretizing variables to binary features")

        if args.parallel:
            with multiprocessing.Pool(args.n_jobs) as pool:
                out = list(
                    tqdm(
                        pool.imap(
                            smart_qcut_dummify_parallel,
                            [
                                (
                                    col_data,
                                    discretization_bins[col_data.name],
                                    args.use_ordinal_encoding,
                                )
                                for col_data in ts_mixed_cols
                            ],
                        ),
                        total=len(ts_mixed_cols),
                        desc="FIDDLE: mapping features qcut",
                    )
                )
        else:
            out = [
                smart_qcut_dummify(
                    col_data,
                    discretization_bins[col_data.name],
                    use_ordinal_encoding=args.use_ordinal_encoding,
                )
                for col_data in tqdm(ts_mixed_cols, desc="FIDDLE: mapping features")
            ]

    else:
        dtype = float
        df = ts_mixed.copy()

        # Split a mixed column into numeric and string columns
        for col in df.columns:
            col_data = df[col]
            col_is_numeric = [is_numeric(v) for v in col_data if not pd.isnull(v)]
            if not all(col_is_numeric) and any(
                col_is_numeric
            ):  # have mixed type values
                numeric_mask = col_data.apply(is_numeric)
                df[col + "_str"] = df[col].copy()
                df.loc[~numeric_mask, col] = np.nan
                df.loc[numeric_mask, col + "_str"] = np.nan

        ts_mixed_cols = [df[col] for col in df.columns]

        if args.verbose:
            print("Discretizing categorical features...")
        if args.parallel:
            with multiprocessing.Pool(args.n_jobs) as pool:
                out = list(
                    tqdm(
                        pool.imap(
                            smart_dummify_impute,
                            [(col_data) for col_data in ts_mixed_cols],
                        ),
                        total=len(ts_mixed_cols),
                        desc="FIDDLE: discretizing categorical features",
                    )
                )
        else:
            out = [
                smart_dummify_impute(col_data)
                for col_data in tqdm(
                    ts_mixed_cols, desc="FIDDLE: discretizing categorical features"
                )
            ]

    out = [ts_feature_mask, *out]
    D_all = sum(len(df_i.columns) for df_i in out)
    X_all_feature_names = np.asarray(sum([list(df_i.columns) for df_i in out], []))
    X_dense = np.concatenate([df_i.values for df_i in out], axis=1).astype(dtype)
    X_all = sparse.COO(X_dense)

    if args.verbose:
        print("Finished discretizing features")
    assert X_all.shape[0] == N * L
    X_all = X_all.reshape((N, L, D_all))

    if args.verbose:
        print()
        print("Output")
        print("X_all: shape={}, density={:.3f}".format(X_all.shape, X_all.density))
    return X_all, X_all_feature_names, discretization_bins


def post_filter_time_series(X_all, feature_names_all, threshold, args):
    N, L = args.N, args.L
    assert X_all.shape[0] == N
    assert X_all.shape[1] == L
    #     assert X_all.dtype == int
    start_time = time.time()

    X0 = X_all
    feature_names_0 = feature_names_all
    if args.verbose:
        print("Original :", len(feature_names_0))

    ## Remove nearly-constant features (with low variance)
    sel_const = FrequencyThreshold_temporal(threshold=threshold, L=L)
    sel_const.fit(X0.reshape((N * L, -1)))
    m_ts_const = sel_const.get_support()
    assert len(m_ts_const) == X0.shape[-1]
    X1 = X0[:, :, m_ts_const]
    feature_names_1 = feature_names_0[m_ts_const]
    if args.verbose:
        print("Nearly-constant:", len(feature_names_0) - len(feature_names_1))
        print("*** time: ", time.time() - start_time)

    ## Keep only first of pairwise perfectly correlated features
    sel_ts_corr = CorrelationSelector()
    sel_ts_corr.fit(X1.reshape((N * L, -1)))
    m_ts_corr = sel_ts_corr.get_support()
    assert len(m_ts_corr) == X1.shape[-1]
    X2 = X1[:, :, m_ts_corr]
    feature_names_2 = feature_names_1[m_ts_corr]
    feature_aliases = sel_ts_corr.get_feature_aliases(feature_names_1)
    if args.verbose:
        print("Correlated     :", len(feature_names_1) - len(feature_names_2))
        print("*** time: ", time.time() - start_time)

    X = sparse.COO(X2)
    feature_names = feature_names_2
    assert X.shape == (N, L, len(feature_names))

    ## Save output
    if args.verbose:
        print()
        print("Output")
        print("X: shape={}, density={:.3f}".format(X.shape, X.density))

    return X, feature_names, feature_aliases
