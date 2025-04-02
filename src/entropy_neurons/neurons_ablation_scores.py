import pandas as pd
import numpy as np
import math
from scipy.stats import sem, percentileofscore

def compute_entropy_delta_pre_to_post_ablation(
    pre_ablation_data: pd.DataFrame, post_ablation_data: pd.DataFrame
) -> pd.DataFrame:
    pre_ablation_data_ = pre_ablation_data.copy()
    post_ablation_data_ = post_ablation_data.copy()

    pre_ablation_data_.columns = "pre_ablation_" + pre_ablation_data_.columns.values
    post_ablation_data_.columns = "post_ablation_" + post_ablation_data_.columns.values

    pre_post_ablation_data = pd.merge(
        pre_ablation_data_, post_ablation_data_, left_index=True, right_index=True, how="outer"
    )

    # Calculate the entropy difference
    pre_post_ablation_data["entropy_difference"] = (
        pre_post_ablation_data["pre_ablation_entropy"] - pre_post_ablation_data["post_ablation_entropy"]
    )

    # Compute the mean entropy difference for each transition
    mean_entropy_difference = (
        pre_post_ablation_data.groupby(["pre_ablation_knowledge_source", "post_ablation_knowledge_source"])[
            "entropy_difference"
        ]
        .mean()
        .unstack(fill_value=0)
    )

    # Ensure the DataFrame has all three knowledge sources in both index and columns
    knowledge_sources = ["CK", "ND", "PK"]
    mean_entropy_difference = mean_entropy_difference.reindex(
        index=knowledge_sources, columns=knowledge_sources, fill_value=0
    )

    return mean_entropy_difference


def compute_pre_to_post_ablation_convertions(
    pre_ablation_data: pd.DataFrame, post_ablation_data: pd.DataFrame
) -> pd.DataFrame:
    pre_ablation_data_ = pre_ablation_data.copy()
    post_ablation_data_ = post_ablation_data.copy()
    pre_ablation_data_.columns = "pre_ablation_" + pre_ablation_data_.columns.values
    post_ablation_data_.columns = "post_ablation_" + post_ablation_data_.columns.values

    pre_post_ablation_data = pd.merge(
        pre_ablation_data_, post_ablation_data_, left_index=True, right_index=True, how="outer"
    )

    # Count transitions
    knowledge_source_transitions_count = (
        pre_post_ablation_data.groupby(["pre_ablation_knowledge_source", "post_ablation_knowledge_source"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure the DataFrame has all three knowledge sources in both index and columns
    knowledge_sources = ["CK", "ND", "PK"]
    knowledge_source_transitions_count = knowledge_source_transitions_count.reindex(
        index=knowledge_sources, columns=knowledge_sources, fill_value=0
    )

    return knowledge_source_transitions_count


def invariant_knowledge_sources(transition_table_count):
    invariant_count = 0
    for idx in ["PK", "ND", "CK"]:
        if idx in transition_table_count.columns:
            invariant_count += transition_table_count.loc[idx, idx]
    return {
        "invariant_count": invariant_count,
        "invariant_proportion": invariant_count / transition_table_count.to_numpy().sum(),
    }


def compute_mean_and_ci_and_quantile(transition_tables: list[pd.DataFrame], mean_ablation_transition_count):
    """This function takes a list of N 3x3 knowledge source transition tables (dataframes)
    (counts, proportions, or percentages) after ablating activations of randomly
    selected neurons. It also takes a knowledge source transition table corresponding
    to the ablation of entropy neurons. It returns a dict with the following statistics:
        - random_ablation_transition_tables_mean (pd.DataFrame): a 3x3 mean transition table
            of the list of transition tables
        - random_ablation_transition_tables_ci (pd.DataFrame): a 3x3 table with 95% confidence
            intervals corresponding to the retunred mean transition table (this is only returned
            if the transition_tables are proportions as we can assume that the distribution is
            binomial as opposed to counts where we can't be sure about the type of distribution
            and prefer to use quantiles to assess the significance of results.)
        - quantiles (pd.DataFrame): the quantiles of each transition statistics (count or proportion)
            in the random transition distribution.
        -
    """
    # Stack transition_tables into 3D NumPy array
    data_array = np.array([df.values for df in transition_tables])

    # Compute mean across all transition_tables (axis=0)
    mean_values = np.mean(data_array, axis=0)

    # Compute standard error (SE)
    se_values = sem(data_array, axis=0, ddof=1)
    se_df = pd.DataFrame(se_values, index=transition_tables[0].index, columns=transition_tables[0].columns)

    flat_data = data_array.flatten()
    is_data_proportions = np.all(flat_data >= 0) and np.all(flat_data <= 1)
    if is_data_proportions:
        # Compute 95% Confidence Interval (CI = 1.96 * SE) for a binomial distribution (transition from
        # a knowledge source or not)

        binomial_ci_values = 1.96 * np.sqrt(mean_values * (1 - mean_values) / len(data_array))
        binomial_ci_df = pd.DataFrame(
            binomial_ci_values, index=transition_tables[0].index, columns=transition_tables[0].columns
        )

    mean_df = pd.DataFrame(mean_values, index=transition_tables[0].index, columns=transition_tables[0].columns)

    # Compute quantiles
    quantiles = np.zeros_like(mean_ablation_transition_count, dtype=float)
    for i in range(mean_ablation_transition_count.shape[0]):
        for j in range(mean_ablation_transition_count.shape[1]):
            quantiles[i, j] = percentileofscore(data_array[:, i, j], mean_ablation_transition_count[i, j])
    # Convert quantiles to DataFrame
    quantiles_df = pd.DataFrame(quantiles, index=transition_tables[0].index, columns=transition_tables[0].columns)

    return {
        "random_ablation_transition_tables_mean": mean_df,
        "random_ablation_transition_tables_ci": binomial_ci_df if is_data_proportions else None,
        "EN_transition_table_quantiles_in_random_distro": quantiles_df,
        "random_ablation_standard_error_table": se_df,
    }


def binomial_std(n, p):
    """
    Compute the standard deviation of a binomial distribution.

    Parameters:
    n (int): Number of trials.
    p (float): Probability of success on each trial.

    Returns:
    float: Standard deviation of the binomial distribution.
    """
    return math.sqrt(n * p * (1 - p))


def compute_ckcr(df):
    if "CK" in df.columns:
        PK2CK = df.loc["PK", "CK"]
        ND2CK = df.loc["ND", "CK"]
        ND2ND = df.loc["ND", "ND"]
        PK2PK = df.loc["PK", "PK"]
        ND2PK = df.loc["ND", "PK"]
        PK2ND = df.loc["PK", "ND"]
        return 100 * (PK2CK + ND2CK) / (PK2CK + ND2CK + ND2ND + PK2PK + ND2PK + PK2ND)
    else:
        return 0


def compute_pkcr(df):
    if "PK" in df.columns:
        ND2PK = df.loc["ND", "PK"]
        CK2PK = df.loc["CK", "PK"]
        ND2ND = df.loc["ND", "ND"]
        CK2CK = df.loc["CK", "CK"]
        ND2CK = df.loc["ND", "CK"]
        CK2ND = df.loc["CK", "ND"]
        return 100 * (ND2PK + CK2PK) / (ND2PK + CK2PK + ND2ND + CK2CK + ND2CK + CK2ND)
    else:
        return 0


def compute_ndcr(df):
    if "PK" in df.columns:
        CK2ND = df.loc["CK", "ND"]
        PK2ND = df.loc["PK", "ND"]
        PK2PK = df.loc["PK", "PK"]
        CK2CK = df.loc["CK", "CK"]
        CK2PK = df.loc["CK", "PK"]
        PK2CK = df.loc["PK", "CK"]
        return 100 * (CK2ND + PK2ND) / (CK2ND + PK2ND + PK2PK + CK2CK + CK2PK + PK2CK)
    else:
        return 0


def compute_transition_proportions(transition_table_counts: pd.DataFrame) -> pd.DataFrame:
    """Returns a proportion of convertions for each class (source) to another (target)
    relative to the source class.

    transition_table_counts (pd.DataFrame): a 3x3 shape dataframe with the count of elements
        that went from one class (source -- row index) to another (target -- column name).
    """
    row_sums = transition_table_counts.sum(axis=1)
    proportion_table = transition_table_counts.div(row_sums, axis=0)
    return proportion_table


def compute_random_ablation_entropy_delta(
    pre_ablation_df: pd.DataFrame,
    ablation_data: pd.DataFrame,
    ablation_type: pd.DataFrame,
    nb_random_samples_for_ablation: int,
) -> pd.DataFrame:
    random_entropy_delta = []

    for i in range(nb_random_samples_for_ablation):
        random_entropy_delta.append(
            compute_entropy_delta_pre_to_post_ablation(
                pre_ablation_data=pre_ablation_df,
                post_ablation_data=pd.DataFrame(ablation_data[f"random_neurons_{i}_{ablation_type}"]),
            )
        )

    return random_entropy_delta


def compute_random_ablation_scores(pre_ablation_df, ablation_data, ablation_type, nb_random_samples_for_ablation):
    """Computes the CKCR, PKCR, and transition tables (proportions and counts) for the random ablations"""
    random_ablation_knowledge_source_transitions_counts = []
    random_ckcr_scores = []
    random_pkcr_scores = []
    random_ndcr_scores = []
    random_transition_proportions = []

    for i in range(nb_random_samples_for_ablation):
        random_ablation_knowledge_source_transitions_count = compute_pre_to_post_ablation_convertions(
            pre_ablation_data=pre_ablation_df.copy(),
            post_ablation_data=pd.DataFrame(ablation_data[f"random_neurons_{i}_{ablation_type}"]),
        )
        random_ablation_knowledge_source_transitions_count.index = ["CK", "ND", "PK"]
        random_ckcr_scores.append(compute_ckcr(random_ablation_knowledge_source_transitions_count))
        random_pkcr_scores.append(compute_pkcr(random_ablation_knowledge_source_transitions_count))
        random_ndcr_scores.append(compute_ndcr(random_ablation_knowledge_source_transitions_count))
        random_transition_proportion = compute_transition_proportions(
            random_ablation_knowledge_source_transitions_count
        )
        random_transition_proportions.append(random_transition_proportion)
        random_ablation_knowledge_source_transitions_counts.append(random_ablation_knowledge_source_transitions_count)

    return (
        random_ckcr_scores,
        random_pkcr_scores,
        random_ndcr_scores,
        random_transition_proportions,
        random_ablation_knowledge_source_transitions_counts,
    )


def compute_conversion_scores_data(
    random_ckcr_scores, random_pkcr_scores, random_ndcr_scores, ckcr_EN_score, pkcr_EN_score, ndcr_EN_score
):
    ckcr_random_mean = np.mean(random_ckcr_scores)
    pkcr_random_mean = np.mean(random_pkcr_scores)
    ndcr_random_mean = np.mean(random_ndcr_scores)

    ckcr_random_std = np.std(random_ckcr_scores)
    pkcr_random_std = np.std(random_pkcr_scores)
    ndcr_random_std = np.std(random_ndcr_scores)

    ckcr_EN_quantile_in_random_distro = percentileofscore(random_ckcr_scores, ckcr_EN_score)
    pkcr_EN_quantile_in_random_distro = percentileofscore(random_pkcr_scores, pkcr_EN_score)
    ndcr_EN_quantile_in_random_distro = percentileofscore(random_ndcr_scores, ndcr_EN_score)

    convertion_ratios_data = {
        "ckcr_random_distro": random_ckcr_scores,
        "pkcr_random_distro": random_pkcr_scores,
        "ndcr_random_distro": random_ndcr_scores,
        "ckcr_random_mean": ckcr_random_mean,
        "pkcr_random_mean": pkcr_random_mean,
        "ndcr_random_mean": ndcr_random_mean,
        "ckcr_EN_quantile_in_random_distro": ckcr_EN_quantile_in_random_distro,
        "pkcr_EN_quantile_in_random_distro": pkcr_EN_quantile_in_random_distro,
        "ndcr_EN_quantile_in_random_distro": ndcr_EN_quantile_in_random_distro,
        "ckcr_random_std": ckcr_random_std,
        "pkcr_random_std": pkcr_random_std,
        "ndcr_random_std": ndcr_random_std,
        "ckcr_EN_score": ckcr_EN_score,
        "pkcr_EN_score": pkcr_EN_score,
        "ndcr_EN_score": ndcr_EN_score,
    }

    return convertion_ratios_data
