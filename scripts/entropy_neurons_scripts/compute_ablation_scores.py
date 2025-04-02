import pandas as pd
from pathlib import Path
import argparse
import json
import pickle
from scipy.stats import percentileofscore

from src.entropy_neurons.neurons_ablation_scores import (
    compute_pre_to_post_ablation_convertions,
    compute_mean_and_ci_and_quantile,
    compute_ckcr,
    compute_pkcr,
    compute_transition_proportions,
    compute_random_ablation_scores,
    compute_conversion_scores_data,
    invariant_knowledge_sources,
    compute_random_ablation_entropy_delta,
    compute_entropy_delta_pre_to_post_ablation,
)
from src.entropy_neurons.neurons_ablations import AblationType
from src.utils import setup_directories
from src.entropy_neurons.constants import MODEL_NAMES, PATH, NB_RANDOM_SAMPLES_FOR_ABLATION

if __name__ == "__main__":
    model_names = MODEL_NAMES

    parser = argparse.ArgumentParser(description="Computing ablation scores", add_help=True)
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

    if model_name is not None:
        model_names = [model_name]

    ablation_values = [ablation_type.value for ablation_type in AblationType]
    if args.ablation_value is not None:
        ablation_values = [args.ablation_value]

    for model_name in model_names:
        print(f"Computing ablation scores for {model_name}")

        datetime = args.datetime

        run_path = Path(PATH) / datetime

        setup_directories(permanent_path=run_path, model_name=model_name)

        entropy_neurons_idxs = pd.read_csv(
            Path("input_data")
            / "selected_entropy_neurons"
            / f"{model_name}_selected_entropy_neurons.csv"
        ).neuron_idx.tolist()

        with open(
            Path(run_path)
            / "ablation_experiments"
            / model_name
            / f"activation_ablation_scores_with_random_ablation-nb_random_ablations={NB_RANDOM_SAMPLES_FOR_ABLATION}.json"
        ) as fo:
            ablation_data = json.load(fo)

        pre_ablation_df = pd.DataFrame(ablation_data["pre_ablation"])

        for ablation_type in ablation_values:
            # compute the scores for entropy neurons ablation
            print(f"ablation_type: {ablation_type}")
            entropy_delta = compute_entropy_delta_pre_to_post_ablation(
                pre_ablation_data=pre_ablation_df.copy(),
                post_ablation_data=pd.DataFrame(ablation_data[ablation_type]),
            )
            knowledge_source_transitions_count = compute_pre_to_post_ablation_convertions(
                pre_ablation_data=pre_ablation_df.copy(),
                post_ablation_data=pd.DataFrame(ablation_data[ablation_type]),
            )
            en_only_ablation_invariant = invariant_knowledge_sources(knowledge_source_transitions_count)

            knowedge_source_transition_proportion = compute_transition_proportions(
                knowledge_source_transitions_count
            )
            ckcr_EN_score = compute_ckcr(knowledge_source_transitions_count)
            pkcr_EN_score = compute_pkcr(knowledge_source_transitions_count)
            ndcr_EN_score = compute_pkcr(knowledge_source_transitions_count)

            (
                random_ckcr_scores,
                random_pkcr_scores,
                random_ndcr_scores,
                random_transition_proportions,
                random_ablation_knowledge_source_transitions_counts,
            ) = compute_random_ablation_scores(
                pre_ablation_df=pre_ablation_df,
                ablation_data=ablation_data,
                ablation_type=ablation_type,
                nb_random_samples_for_ablation=NB_RANDOM_SAMPLES_FOR_ABLATION,
            )

            random_ablation_knowledge_source_entropy_delta_tables = compute_random_ablation_entropy_delta(
                pre_ablation_df=pre_ablation_df,
                ablation_data=ablation_data,
                ablation_type=ablation_type,
                nb_random_samples_for_ablation=NB_RANDOM_SAMPLES_FOR_ABLATION,
            )

            random_ablation_invariant = [
                invariant_knowledge_sources(random_transition_counts)
                for random_transition_counts in random_ablation_knowledge_source_transitions_counts
            ]

            # compute the quantile of EN only ablation invariant in the random ablation invariants
            invariants = {
                "en_only_ablation": en_only_ablation_invariant,
                "random_ablation_distro": random_ablation_invariant,
            }

            random_ablation_invariant_proportions = [
                random_invariant["invariant_proportion"] for random_invariant in random_ablation_invariant
            ]
            invariants["invariant_quantile_proportion"] = percentileofscore(
                random_ablation_invariant_proportions, en_only_ablation_invariant["invariant_proportion"]
            )

            convertion_ratios_data = compute_conversion_scores_data(
                random_ckcr_scores=random_ckcr_scores,
                random_pkcr_scores=random_pkcr_scores,
                random_ndcr_scores=random_ndcr_scores,
                ckcr_EN_score=ckcr_EN_score,
                pkcr_EN_score=pkcr_EN_score,
                ndcr_EN_score=ndcr_EN_score,
            )

            control_stats_entropy_delta = compute_mean_and_ci_and_quantile(
                transition_tables=random_ablation_knowledge_source_entropy_delta_tables,
                mean_ablation_transition_count=entropy_delta.to_numpy(),
            )

            mean_ci_quantile_ablation_transitions = {}

            mean_ci_quantile_ablation_transitions["proportions"] = compute_mean_and_ci_and_quantile(
                transition_tables=random_transition_proportions,
                mean_ablation_transition_count=knowedge_source_transition_proportion.to_numpy(),
            )

            mean_ci_quantile_ablation_transitions["counts"] = compute_mean_and_ci_and_quantile(
                transition_tables=random_ablation_knowledge_source_transitions_counts,
                mean_ablation_transition_count=knowledge_source_transitions_count.to_numpy(),
            )

            entropy_scores = {
                "entropy_delta": entropy_delta,
                "random_entropy_delta_tables": random_ablation_knowledge_source_entropy_delta_tables,
                "control_stats_entropy_delta": control_stats_entropy_delta,
            }

            for data_type in ["proportions", "counts"]:
                with open(
                    Path(run_path)
                    / "ablation_experiments"
                    / model_name
                    / f"{model_name}_{ablation_type}_stats_mean_ci_quantile_{data_type}.pkl",
                    "wb+",
                ) as fo:
                    pickle.dump(mean_ci_quantile_ablation_transitions[data_type], fo)

            with open(
                Path(run_path)
                / "ablation_experiments"
                / model_name
                / f"{model_name}_{ablation_type}_invariants.pkl",
                "wb+",
            ) as fo:
                pickle.dump(invariants, fo)

            # EN only ablation data transition counts
            knowledge_source_transitions_count.to_csv(
                Path(run_path)
                / "ablation_experiments"
                / model_name
                / f"{model_name}_{ablation_type}_EN_only_transition_counts.csv",
                index=False,
            )

            knowedge_source_transition_proportion.to_csv(
                Path(run_path)
                / "ablation_experiments"
                / model_name
                / f"{model_name}_{ablation_type}_EN_only_transition_proportions.csv",
                index=False,
            )

            # pkcr, ckcr, etc.
            with open(
                Path(run_path)
                / "ablation_experiments"
                / model_name
                / f"{model_name}_{ablation_type}_transition_ratios.json",
                "w+",
            ) as fo:
                json.dump(convertion_ratios_data, fo)

            with open(
                Path(run_path)
                / "ablation_experiments"
                / model_name
                / f"{model_name}_{ablation_type}_entropy_scores.pkl",
                "wb+",
            ) as fo:
                pickle.dump(entropy_scores, fo)
