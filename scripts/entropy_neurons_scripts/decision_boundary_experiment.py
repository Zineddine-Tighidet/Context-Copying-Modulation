import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

from src.entropy_neurons.constants import PATH, MODEL_NAMES, NB_RANDOM_SAMPLES_FOR_ABLATION
from src.entropy_neurons.neurons_ablations import AblationType

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

    ablation_values = [ablation_type.value for ablation_type in AblationType]
    if args.ablation_value is not None:
        ablation_values = [args.ablation_value]

    if model_name is not None:
        model_names = [model_name]

    for model_name in model_names:
        print(model_name)

        with open(Path(PATH) / args.datetime / "ablation_experiments" / model_name / f"activation_ablation_scores_with_random_ablation-nb_random_ablations={NB_RANDOM_SAMPLES_FOR_ABLATION}.json", "r") as fo:
            log_probas_pre_post_ablation = json.load(fo)

        log_probas_pre_ablation_df = pd.DataFrame(log_probas_pre_post_ablation["pre_ablation"])

        for ablation_type in ablation_values:
            log_probas_mean_ablation_df = pd.DataFrame(log_probas_pre_post_ablation[ablation_type])

            ck_log_prob = log_probas_pre_ablation_df.conflicting_counter_knowledge_object_log_proba.values
            pk_log_prob = log_probas_pre_ablation_df.conflicting_parametric_knowledge_object_log_proba.values
            nd_log_prob = log_probas_pre_ablation_df.conflicting_undefined_knowledge_object_log_proba.values

            # log(P_CK / P_PK)
            x1_pre = ck_log_prob - pk_log_prob
            # log(P_ND / P_CK)
            x2_pre = nd_log_prob - ck_log_prob
            # log(P_ND / P_PK)
            x3_pre = nd_log_prob - pk_log_prob

            x_d = 0  # Decision boundary

            ck_log_prob = log_probas_mean_ablation_df.conflicting_counter_knowledge_object_log_proba.values
            pk_log_prob = log_probas_mean_ablation_df.conflicting_parametric_knowledge_object_log_proba.values
            nd_log_prob = log_probas_mean_ablation_df.conflicting_undefined_knowledge_object_log_proba.values

            # log(P_CK / P_PK)
            x1_post = ck_log_prob - pk_log_prob
            # log(P_ND / P_CK)
            x2_post = nd_log_prob - ck_log_prob
            # log(P_ND / P_PK)
            x3_post = nd_log_prob - pk_log_prob

            epsilon_1 = x1_pre - x1_post
            epsilon_2 = x2_pre - x2_post
            epsilon_3 = x3_pre - x3_post

            epsilons = np.concatenate([epsilon_1, epsilon_2, epsilon_3])
            noise_categ = np.concatenate([[r"$\epsilon_1 = \Delta\log \frac{P_{CK}}{P_{PK}}$" for _ in range(len(epsilon_1))], [r"$\epsilon_2 = \Delta\log \frac{P_{ND}}{P_{CK}}$" for _ in range(len(epsilon_2))], [r"$\epsilon_3 = \Delta\log \frac{P_{ND}}{P_{PK}}$" for _ in range(len(epsilon_3))]])

            font_size = 16
            plt.figure()
            plt.rcParams.update({'font.size': font_size})
            palette = sns.color_palette("Dark2", len(log_probas_pre_ablation_df.knowledge_source.unique()))

            sns.kdeplot(
                x=epsilons,
                hue=noise_categ,
                fill=True,
                common_norm=False,
                palette=palette,
            )

            plt.xlim(-1.5, 1.5)

            plt.ylabel('Density', fontsize=font_size+8)

            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)

            plt.tight_layout()

            plt.savefig(Path(PATH) / args.datetime / "entropy_neurons_figures" / model_name /  f"{model_name}_{ablation_type}_ablation_empirical_noises.pdf")
            plt.show()


            ck_mask = log_probas_pre_ablation_df.knowledge_source == 'CK'
            pk_mask = log_probas_pre_ablation_df.knowledge_source == 'PK'
            nd_mask = log_probas_pre_ablation_df.knowledge_source == 'ND'

            plt.figure()
            plt.rcParams.update({'font.size': font_size})

            color_map = {
                "CK": "#98df98",  # Lighter green
                "ND": "#6baed6",  # Lighter blue
                "PK": "#ff9848"   # Lighter orange
            }

            sns.kdeplot(
                x=x1_pre[ck_mask],
                label='CK',
                color=color_map["CK"],
                clip=(0, None),
                fill=True
            )

            sns.kdeplot(
                x=x1_pre[pk_mask],
                label='PK',
                color=color_map["PK"],
                clip=(None, 0),
                fill=True
            )

            sns.kdeplot(
                x=x1_pre[nd_mask],
                label='ND',
                color=color_map["ND"],
                clip=(None, None),
                fill=True
            )

            plt.ylabel('Density', fontsize=font_size+8)

            plt.axvline(0, linewidth=3, linestyle="--", color="r")

            plt.legend(title='Knowledge Source')

            plt.tight_layout()

            plt.savefig(Path(PATH) / args.datetime / "entropy_neurons_figures" / model_name /  f"{model_name}_{ablation_type}_ablation_log_ck_pk.pdf")

            plt.show()

            plt.figure()
            plt.rcParams.update({'font.size': font_size})

            sns.kdeplot(
                x=x2_pre[ck_mask],
                label='CK',
                color=color_map["CK"],
                clip=(None, 0),
                fill=True
            )

            sns.kdeplot(
                x=x2_pre[pk_mask],
                label='PK',
                color=color_map["PK"],
                clip=(None, None),
                fill=True
            )

            sns.kdeplot(
                x=x2_pre[nd_mask],
                label='ND',
                color=color_map["ND"],
                clip=(0, None),
                fill=True
            )

            plt.ylabel('Density', fontsize=font_size+8)

            plt.axvline(0, linewidth=3, linestyle="--", color="r")

            plt.legend(title='Knowledge Source')

            plt.tight_layout()

            plt.savefig(Path(PATH) / args.datetime / "entropy_neurons_figures" / model_name /  f"{model_name}_{ablation_type}_ablation_log_nd_ck.pdf")

            plt.show()

            plt.figure()
            plt.rcParams.update({'font.size': font_size})
            sns.kdeplot(
                x=x3_pre[ck_mask],
                label='CK',
                color=color_map["CK"],
                clip=(None, None),
                fill=True
            )

            sns.kdeplot(
                x=x3_pre[pk_mask],
                label='PK',
                color=color_map["PK"],
                clip=(None, 0),
                fill=True
            )
            sns.kdeplot(
                x=x3_pre[nd_mask],
                label='ND',
                color=color_map["ND"],
                clip=(0, None),
                fill=True
            )

            plt.ylabel('Density', fontsize=font_size+8)

            plt.axvline(0, linewidth=3, linestyle="--", color="r")

            plt.legend(title='Knowledge Source')

            plt.tight_layout()

            plt.savefig(Path(PATH) / args.datetime / "entropy_neurons_figures" / model_name / f"{model_name}_{ablation_type}_ablation_log_nd_pk.pdf")

            plt.show()