import pandas as pd
from pathlib import Path
import json
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from src.entropy_neurons.neurons_visualizations import plot_transition_distribution
from src.entropy_neurons.neurons_ablations import AblationType
from src.entropy_neurons.constants import PATH, MODEL_NAMES

if __name__ == "__main__":
    model_names = MODEL_NAMES

    parser = argparse.ArgumentParser(description="Plotting post ablation figures", add_help=True)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[name for name in model_names],
    )
    parser.add_argument("--datetime", type=str)
    parser.add_argument("--device", choices=["cuda", "cpu"], type=str)
    parser.add_argument("--ablation_value", type=str, choices=[ablation_type.value for ablation_type in AblationType])

    args = parser.parse_args()
    model_name = args.model_name
    
    device = args.device if args.device is not None else "cuda"

    ablation_values = [ablation_type.value for ablation_type in AblationType]
    if args.ablation_value is not None:
        ablation_values = [args.ablation_value]

    if model_name is not None:
        model_names = [model_name]

    control_experiment_scores = {}
    en_only_ablation_transition_counts = {}
    en_only_ablation_transition_proportions = {}
    convertion_ratios_data = {}
    invariants = {}

    for model_name in model_names:
        print(f"Loading scores for {model_name}")
        control_experiment_scores[model_name] = {}
        en_only_ablation_transition_proportions[model_name] = {}
        en_only_ablation_transition_counts[model_name] = {}
        convertion_ratios_data[model_name] = {}
        invariants[model_name] = {}

        datetime = args.datetime

        for ablation_type in ablation_values:
            control_experiment_scores[model_name][ablation_type] = {}
            en_only_ablation_transition_proportions[model_name][ablation_type] = {}
            en_only_ablation_transition_counts[model_name][ablation_type] = {}
            convertion_ratios_data[model_name][ablation_type] = {}

            print(f"Ablation type: {ablation_type}")

            with open(
                Path(PATH)
                / datetime
                / "ablation_experiments"
                / model_name
                / f"{model_name}_{ablation_type}_invariants.pkl",
                "rb",
            ) as fo:
                invariants[model_name][ablation_type] = pickle.load(fo)

            for data_type in ["proportions", "counts"]:
                with open(
                    Path(PATH)
                    / datetime
                    / "ablation_experiments"
                    / model_name
                    / f"{model_name}_{ablation_type}_stats_mean_ci_quantile_{data_type}.pkl",
                    "rb",
                ) as fo:
                    control_experiment_scores[model_name][ablation_type][data_type] = pickle.load(fo)

                en_only_ablation_transition_counts[model_name][ablation_type][data_type] = pd.read_csv(
                    Path(PATH)
                    / datetime
                    / "ablation_experiments"
                    / model_name
                    / f"{model_name}_{ablation_type}_EN_only_transition_counts.csv"
                )
                en_only_ablation_transition_proportions[model_name][ablation_type][data_type] = pd.read_csv(
                    Path(PATH)
                    / datetime
                    / "ablation_experiments"
                    / model_name
                    / f"{model_name}_{ablation_type}_EN_only_transition_proportions.csv"
                )

                en_only_ablation_transition_proportions[model_name][ablation_type][data_type].index = [
                    "CK",
                    "ND",
                    "PK",
                ]
                en_only_ablation_transition_counts[model_name][ablation_type][data_type].index = [
                    "CK",
                    "ND",
                    "PK",
                ]

                with open(
                    Path(PATH)
                    / datetime
                    / "ablation_experiments"
                    / model_name
                    / f"{model_name}_{ablation_type}_transition_ratios.json",
                ) as fo:
                    convertion_ratios_data[model_name][ablation_type][data_type] = json.load(fo)

            print(model_name)
            data = convertion_ratios_data[model_name][ablation_type][data_type]

            ckcr_error_bar = 3 * np.sqrt(
                data["ckcr_EN_score"]
                * (1 - data["ckcr_EN_score"])
                / en_only_ablation_transition_counts[model_name][ablation_type][data_type]
                .loc[["ND", "PK"]]
                .sum(axis=1)
                .sum()
            )

            pkcr_error_bar = 3 * np.sqrt(
                data["pkcr_EN_score"]
                * (1 - data["pkcr_EN_score"])
                / en_only_ablation_transition_counts[model_name][ablation_type][data_type]
                .loc[["ND", "CK"]]
                .sum(axis=1)
                .sum()
            )

            ndcr_error_bar = 3 * np.sqrt(
                data["ndcr_EN_score"]
                * (1 - data["ndcr_EN_score"])
                / en_only_ablation_transition_counts[model_name][ablation_type][data_type]
                .loc[["CK", "PK"]]
                .sum(axis=1)
                .sum()
            )

            df = pd.DataFrame(
                {
                    "Knowledge Source": ["CK", "PK", "ND"],
                    "Conversion Ratio": [data["ckcr_EN_score"], data["pkcr_EN_score"], data["ndcr_EN_score"]],
                    "Std Conversion Ratio": [ckcr_error_bar, pkcr_error_bar, ndcr_error_bar],
                    "Random Mean": [data["ckcr_random_mean"], data["pkcr_random_mean"], data["ndcr_random_mean"]],
                    "Random Std": [data["ckcr_random_std"], data["pkcr_random_std"], data["ndcr_random_std"]],
                }
            )

            sns.set_theme(context="paper")

            fig, ax = plt.subplots(figsize=(8, 6))

            barplot = sns.barplot(
                x="Knowledge Source",
                y="Conversion Ratio",
                data=df,
                palette="Set2",
                ax=ax,
                edgecolor="black",
                linewidth=1.5,
                width=0.4,
                hatch="\\\\",
            )

            for i, row in df.iterrows():
                ax.errorbar(
                    x=i,
                    y=row["Conversion Ratio"],
                    yerr=row["Std Conversion Ratio"],
                    fmt="o",
                    color="black",
                    capsize=5,
                    capthick=1.5,
                )

            bar_width = 0.2
            for i, row in df.iterrows():
                ax.bar(
                    i + 1.5 * bar_width,
                    row["Random Mean"],
                    width=bar_width,
                    color="lightgray",
                    edgecolor="black",
                    linewidth=1.5,
                    hatch="/",
                )
                ax.errorbar(
                    x=i + 1.5 * bar_width,
                    y=row["Random Mean"],
                    yerr=row["Random Std"],
                    fmt="o",
                    color="black",
                    capsize=5,
                    capthick=1.5,
                )

            ax.set_xlabel("Knowledge Source", fontsize=16)
            ax.set_ylabel("Conversion Ratio (%)", fontsize=16)

            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

            legend_labels = ["Random Neurons"]
            legend_handles = [
                plt.Rectangle((0, 0), 4, 4, color="lightgray", edgecolor="black", linewidth=1.5, hatch="/")
            ]
            ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=16)

            plt.tight_layout()

            plt.savefig(Path(PATH) / datetime / "entropy_neurons_figures" / model_name / f"{model_name}_{ablation_type}-conversion_ratios.pdf")

            plt.show()

            nd_to_ck = (
                100
                * en_only_ablation_transition_proportions[model_name][ablation_type][
                    "proportions"
                ].loc["ND", "CK"]
            )
            nd_to_ck_count = (
                100
                * en_only_ablation_transition_counts[model_name][ablation_type]["counts"].loc[
                    "ND", "CK"
                ]
            )

            pk_to_ck = (
                100
                * en_only_ablation_transition_proportions[model_name][ablation_type][
                    "proportions"
                ].loc["PK", "CK"]
            )
            pk_to_ck_count = (
                100
                * en_only_ablation_transition_counts[model_name][ablation_type]["counts"].loc[
                    "PK", "CK"
                ]
            )

            nd_to_ck_mean = (
                100
                * control_experiment_scores[model_name][ablation_type]["proportions"][
                    "random_ablation_transition_tables_mean"
                ].loc["ND", "CK"]
            )
            pk_to_ck_mean = (
                100
                * control_experiment_scores[model_name][ablation_type]["proportions"][
                    "random_ablation_transition_tables_mean"
                ].loc["PK", "CK"]
            )

            nd_to_ck_std_err = (
                100
                * control_experiment_scores[model_name][ablation_type]["proportions"][
                    "random_ablation_standard_error_table"
                ].loc["ND", "CK"]
            )
            pk_to_ck_std_err = (
                100
                * control_experiment_scores[model_name][ablation_type]["proportions"][
                    "random_ablation_standard_error_table"
                ].loc["PK", "CK"]
            )

            data = pd.DataFrame(
                {
                    "Transition": [r"ND $\rightarrow$ CK", r"PK $\rightarrow$ CK"],
                    "Proportion": [nd_to_ck, pk_to_ck],
                    "EN Std error": [
                        3 * np.sqrt(nd_to_ck * (1 - nd_to_ck) / nd_to_ck_count),
                        3 * np.sqrt(pk_to_ck * (1 - pk_to_ck) / pk_to_ck_count),
                    ],
                    "Random Mean": [nd_to_ck_mean, pk_to_ck_mean],
                    "Random Standard Error": [nd_to_ck_std_err, pk_to_ck_std_err],
                }
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            barplot = sns.barplot(
                x="Transition",
                y="Proportion",
                data=data,
                palette="Set2",
                ax=ax,
                edgecolor="black",
                linewidth=1.5,
                hatch="\\\\",
                width=0.4,
            )

            for i, row in data.iterrows():
                ax.errorbar(
                    x=i,
                    y=row["Proportion"],
                    yerr=row["Random Standard Error"],
                    fmt="o",
                    color="black",
                    capsize=5,
                    capthick=1.5,
                )

            bar_width = 0.2
            for i, row in data.iterrows():
                ax.bar(
                    i + 1.5 * bar_width,
                    row["Random Mean"],
                    width=bar_width,
                    color="lightgray",
                    edgecolor="black",
                    linewidth=1.5,
                    hatch="/",
                )
                ax.errorbar(
                    x=i + 1.5 * bar_width,
                    y=row["Random Mean"],
                    yerr=row["Random Standard Error"],
                    fmt="o",
                    color="black",
                    capsize=5,
                    capthick=1.5,
                )

            ax.set_xlabel("Transition", fontsize=16)
            ax.set_ylabel("Conversion Ratio (%)", fontsize=16)

            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

            legend_labels = ["Random Neurons"]
            legend_handles = [
                plt.Rectangle((0, 0), 4, 4, color="lightgray", edgecolor="black", linewidth=1.5, hatch="/")
            ]
            ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=16)

            plt.tight_layout()

            plt.savefig(
                Path(PATH) / datetime / "entropy_neurons_figures" / model_name / f"{model_name}_{ablation_type}-ndck_pkck_conversion_ratios.pdf"
            )
            plt.show()

            plot_transition_distribution(
                model_name=model_name,
                ablation_type=ablation_type,
                model_data=invariants[model_name],
                savepath=Path(PATH) / datetime,
            )
