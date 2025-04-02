import pandas as pd
from pathlib import Path
import pickle
import argparse

from src.entropy_neurons.neurons_latex import (
    generate_latex_synthetic_table_counts,
    generate_latex_synthetic_table_proportions,
    generate_latex_table_invariance_scores_ablation_value_wise,
)
from src.entropy_neurons.neurons_ablations import AblationType
from src.entropy_neurons.constants import PATH, MODEL_NAMES

if __name__ == "__main__":
    model_names = MODEL_NAMES

    parser = argparse.ArgumentParser(description="Computing ablation scores", add_help=True)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[name for name in model_names],
        help="Model name to build the knowledge datasets.",
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
    invariants = {}

    for model_name in model_names:
        print(f"Loading scores for {model_name}")
        control_experiment_scores[model_name] = {}
        en_only_ablation_transition_proportions[model_name] = {}
        en_only_ablation_transition_counts[model_name] = {}
        invariants[model_name] = {}

        datetime = args.datetime

        for ablation_type in ablation_values:

            control_experiment_scores[model_name][ablation_type] = {}
            en_only_ablation_transition_proportions[model_name][ablation_type] = {}
            en_only_ablation_transition_counts[model_name][ablation_type] = {}

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

    ablation_value_wise_transition_scores_table = generate_latex_table_invariance_scores_ablation_value_wise(
        ablation_types=AblationType, data=invariants, model_names=model_names
    )

    with open(
        Path(PATH) / datetime / "latex_tables" / "ablation_value_wise_transition_scores_table.txt", "w"
    ) as fo:
        fo.write(ablation_value_wise_transition_scores_table)

    datetime = args.datetime

    # generate synthetic table
    for ablation_type in ablation_values:

        latex_synthetic_table_counts = generate_latex_synthetic_table_counts(
            control_experiment_scores=control_experiment_scores,
            en_only_ablation_transition_counts=en_only_ablation_transition_counts,
            en_only_ablation_transition_proportions=en_only_ablation_transition_proportions,
            ablation_type=ablation_type,
            model_names=model_names,
        )
        with open(
            Path(PATH)
            / datetime
            / "latex_tables"
            / f"{ablation_type}-transition_scores_counts.txt",
            "w",
        ) as fo:
            fo.write(latex_synthetic_table_counts)

        latex_synthetic_table_proportions = generate_latex_synthetic_table_proportions(
            control_experiment_scores=control_experiment_scores,
            en_only_ablation_transition_counts=en_only_ablation_transition_counts,
            en_only_ablation_transition_proportions=en_only_ablation_transition_proportions,
            ablation_type=ablation_type,
            model_names=model_names,
        )
        with open(
            Path(PATH)
            / datetime
            / "latex_tables"
            / f"{ablation_type}-transition_scores.txt",
            "w",
        ) as fo:
            fo.write(latex_synthetic_table_proportions)
