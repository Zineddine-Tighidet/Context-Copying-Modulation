import pandas as pd
import argparse
import transformer_lens
from transformer_lens import HookedTransformer
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

from src.utils import setup_directories
from src.entropy_neurons.neurons_ablations import (
    generate_with_activation_ablation,
    generate_without_ablation,
    generate_random_neurons_idxs,
)
from src.model import MODEL_PATHS
from src.entropy_neurons.neurons_ablations import AblationType
from src.entropy_neurons.constants import PATH, MODEL_NAMES, ABLATION_ALPHA, NB_RANDOM_SAMPLES_FOR_ABLATION, SPLIT_REGEXP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computing neurons mean activations")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", choices=["cuda", "cpu"], type=str)
    parser.add_argument("--datetime", type=str)
    parser.add_argument("--neurons_ablation", action="store_true")
    parser.add_argument("--ablation_value", type=str, choices=[ablation_type.value for ablation_type in AblationType])

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda"

    print(f"Device: {device}")

    model_names = MODEL_NAMES

    if args.model_name is not None:
        model_names = [args.model_name]
    
    ablation_values = [ablation_type.value for ablation_type in AblationType]
    if args.ablation_value is not None:
        ablation_values = [args.ablation_value]

    for model_name in model_names:
        print("=" * 100)
        print(f"Processing ablations on {model_name}")
        print("=" * 100)

        datetime = args.datetime
        run_path = Path(PATH) / datetime

        setup_directories(permanent_path=run_path, model_name=model_name)

        entropy_neurons_idxs = pd.read_csv(
            Path("input_data")
            / "selected_entropy_neurons"
            / f"{model_name}_selected_entropy_neurons.csv"
        ).neuron_idx.tolist()

        model_path = MODEL_PATHS[model_name]

        transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(model_path)
        model = HookedTransformer.from_pretrained(model_path, device=device)

        activations_stats = {}
        for statistic in ["mean", "std", "median", "min", "max", "mode"]:
            with open(
                Path("input_data")
                / "activations_stats"
                / model_name
                / f"neurons_{statistic}_activations_last_layer.npy",
                "rb",
            ) as fo:
                activations_stats[f"{statistic}_activations"] = np.load(fo)

        probing_data = pd.read_csv(
            Path(run_path)
            / "ablation_experiments"
            / model_name
            / "probing_prompts.csv"
        )
        probing_prompts = probing_data.knowledge_probing_prompt.tolist()
        pk_first_token_id = probing_data.parametric_knowledge_first_token_id.tolist()
        ck_first_token_id = probing_data.counter_knowledge_object_first_token_id.tolist()

        pre_ablation_model_outputs = generate_without_ablation(
            max_new_tokens=10, probing_data=probing_data, model=model, batch_size=256, split_regexp=SPLIT_REGEXP
        )

        ablation_data_results = {
            "pre_ablation": {
                "probing_prompts": probing_prompts,
                "entropy": pre_ablation_model_outputs["entropies"],
                "rel_lemma": probing_data.rel_lemma.tolist(),
                "relation_group_id": probing_data.relation_group_id.tolist(),
                "output_txt": pre_ablation_model_outputs["output_texts"],
                "output_object": pre_ablation_model_outputs["output_objects"],
                "knowledge_source": pre_ablation_model_outputs["knowledge_sources"],
                "parametric_object": probing_data.parametric_knowledge_object.tolist(),
                "context_object": probing_data.counter_knowledge_object.tolist(),
                "conflicting_counter_knowledge_object_log_proba": pre_ablation_model_outputs["ck_log_prob"],
                "conflicting_parametric_knowledge_object_log_proba": pre_ablation_model_outputs["pk_log_prob"],
                "conflicting_undefined_knowledge_object_log_proba": pre_ablation_model_outputs["nd_log_prob"],
            }
        }

        if args.neurons_ablation:
            for ablation_type in ablation_values:
                post_mean_model_outputs = generate_with_activation_ablation(
                    neurons_indices=entropy_neurons_idxs if args.neurons_ablation else None,
                    batch_size=256,
                    max_new_tokens=10,
                    mean_activations=activations_stats["mean_activations"] if args.neurons_ablation else None,
                    std_activations=activations_stats["std_activations"] if args.neurons_ablation else None,
                    max_activations=activations_stats["max_activations"] if args.neurons_ablation else None,
                    min_activations=activations_stats["min_activations"] if args.neurons_ablation else None,
                    mode_activations=activations_stats["mode_activations"] if args.neurons_ablation else None,
                    median_activations=activations_stats["median_activations"] if args.neurons_ablation else None,
                    probing_data=probing_data,
                    split_regexp=SPLIT_REGEXP,
                    model=model,
                    ablation_type=ablation_type,
                    alpha=ABLATION_ALPHA,
                )

                ablation_data_results[ablation_type] = {
                    "output_txt": post_mean_model_outputs["output_texts"],
                    "output_object": post_mean_model_outputs["output_objects"],
                    "knowledge_source": post_mean_model_outputs["knowledge_sources"],
                    "entropy": post_mean_model_outputs["entropies"],
                    "conflicting_counter_knowledge_object_log_proba": post_mean_model_outputs["ck_log_prob"],
                    "conflicting_parametric_knowledge_object_log_proba": post_mean_model_outputs["pk_log_prob"],
                    "conflicting_undefined_knowledge_object_log_proba": post_mean_model_outputs["nd_log_prob"],
                }

            # random ablation for control
            np.random.seed(123)
            previous_random_neurons_idxs = []
            for i in tqdm(range(NB_RANDOM_SAMPLES_FOR_ABLATION), desc="Random ablation"):
                random_neurons_idxs = generate_random_neurons_idxs(
                    entropy_neurons_idxs=entropy_neurons_idxs,
                    min_value=0,
                    max_value=model.cfg.d_mlp - 1,
                    previous_random_neurons_idxs=previous_random_neurons_idxs,
                )
                previous_random_neurons_idxs.append(random_neurons_idxs)

                for ablation_type in ablation_values:
                    post_random_neurons_ablation_model_outputs = generate_with_activation_ablation(
                        neurons_indices=random_neurons_idxs if args.neurons_ablation else None,
                        batch_size=256,
                        max_new_tokens=10,
                        probing_data=probing_data,
                        split_regexp=SPLIT_REGEXP,
                        mean_activations=activations_stats["mean_activations"] if args.neurons_ablation else None,
                        std_activations=activations_stats["std_activations"] if args.neurons_ablation else None,
                        max_activations=activations_stats["max_activations"] if args.neurons_ablation else None,
                        min_activations=activations_stats["min_activations"] if args.neurons_ablation else None,
                        mode_activations=activations_stats["mode_activations"] if args.neurons_ablation else None,
                        median_activations=activations_stats["median_activations"] if args.neurons_ablation else None,
                        model=model,
                        ablation_type=ablation_type,
                        alpha=ABLATION_ALPHA,
                    )

                    ablation_data_results[f"random_neurons_{i}_{ablation_type}"] = {
                        "output_txt": post_random_neurons_ablation_model_outputs["output_texts"],
                        "output_object": post_random_neurons_ablation_model_outputs["output_objects"],
                        "knowledge_source": post_random_neurons_ablation_model_outputs["knowledge_sources"],
                        "entropy": post_random_neurons_ablation_model_outputs["entropies"],
                        "conflicting_counter_knowledge_object_log_proba": post_random_neurons_ablation_model_outputs[
                            "ck_log_prob"
                        ],
                        "conflicting_parametric_knowledge_object_log_proba": post_random_neurons_ablation_model_outputs[
                            "pk_log_prob"
                        ],
                        "conflicting_undefined_knowledge_object_log_proba": post_random_neurons_ablation_model_outputs[
                            "nd_log_prob"
                        ],
                    }

        with open(
            Path(run_path)
            / "ablation_experiments"
            / model_name
            / f"activation_ablation_scores_with_random_ablation-nb_random_ablations={NB_RANDOM_SAMPLES_FOR_ABLATION}.json",
            "w+",
        ) as fo:
            json.dump(ablation_data_results, fo)
