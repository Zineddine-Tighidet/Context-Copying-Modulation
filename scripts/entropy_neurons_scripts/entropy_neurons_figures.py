from pathlib import Path
import argparse
import pandas as pd

from src.model import MODEL_PATHS
from src.entropy_neurons.neurons_visualizations import (
    visualize_logitvar_and_norm,
    visualize_logitvar_and_rho,
    visualize_norm_distribution,
    visualize_rho_layer_variation,
)
from src.utils import setup_directories
from src.entropy_neurons.constants import PATH, MODEL_NAMES

MODEL_NAME_TO_NB_LAYERS = {
    "gpt2-small": 12,
    "Phi-1_5": 24,
    "Meta-Llama-3-8B": 32,
    "Mistral-7B-v0.1": 32,
    "EleutherAI_pythia-1.4b": 24,
}

parser = argparse.ArgumentParser(description="Computing and plotting logitvars and norms", add_help=True)
parser.add_argument(
    "--model_name",
    type=str,
    choices=[name for name in MODEL_PATHS.keys()],
    help="Model name to build the knowledge datasets.",
)
parser.add_argument("--plot_everything", action="store_true")
parser.add_argument("--plot_logitvar_and_rho", action="store_true")
parser.add_argument("--plot_norm_distro", action="store_true")
parser.add_argument("--plot_logitvar_and_norm", action="store_true")
parser.add_argument(
    "--last_layer_only",
    action="store_true",
    help="Visualize on last layer only.",
)
parser.add_argument(
    "--plot_rho_layer_variation", action="store_true", help="Visualize the rho distributions accross layers."
)
parser.add_argument("--datetime", type=str, required=True)

args = parser.parse_args()
assert (
    args.plot_logitvar_and_norm
    or args.plot_logitvar_and_rho
    or args.plot_norm_distro
    or args.plot_everything
    or args.plot_rho_layer_variation
), "You must specify at least one visualization to plot."

run_path = Path(PATH) / args.datetime

model_name = args.model_name
plot_logitvar_and_rho = args.plot_logitvar_and_rho
plot_norm_distro = args.plot_norm_distro
plot_logitvar_and_norm = args.plot_logitvar_and_norm
plot_rho_layer_variation = args.plot_rho_layer_variation
last_layer_only = args.last_layer_only
plot_everything = args.plot_everything
model_names = MODEL_NAMES

if plot_everything:
    print("Producing all plots.")
    plot_logitvar_and_norm = True
    plot_logitvar_and_rho = True
    plot_norm_distro = True
    plot_rho_layer_variation = True

if args.model_name is not None:
    model_names = [model_name]

for model_name in model_names:
    print(f"Visulizations for {model_name}")
    setup_directories(permanent_path=run_path, model_name=model_name)

    entropy_neurons_idxs = pd.read_csv(
        Path("input_data")
        / "selected_entropy_neurons"
        / f"{model_name}_selected_entropy_neurons.csv"
    ).neuron_idx.tolist()

    if plot_logitvar_and_norm:
        logitvars_and_norms_df = pd.read_csv(
            Path(run_path) / "entropy_neurons_measures" / model_name / f"{model_name}_logitvar_and_norm.csv"
        )
        if last_layer_only:
            # without entropy neurons
            visualize_logitvar_and_norm(
                logitvar_and_norm_df=logitvars_and_norms_df,
                model_name=model_name,
                entropy_neurons_idxs=entropy_neurons_idxs,
                all_layers=False,
                save_path=Path(run_path) / "entropy_neurons_figures" / model_name,
            )
        else:
            # All layers (grid)
            # with entropy neurons
            visualize_logitvar_and_norm(
                logitvar_and_norm_df=logitvars_and_norms_df,
                model_name=model_name,
                entropy_neurons_idxs=entropy_neurons_idxs,
                all_layers=True,
                save_path=Path(run_path) / "entropy_neurons_figures" / model_name,
            )

    if plot_logitvar_and_rho:
        rho_df = pd.read_csv(Path(run_path) / "entropy_neurons_measures" / model_name / f"{model_name}_rho.csv")
        logitvars_and_norms_df = pd.read_csv(
            Path(run_path) / "entropy_neurons_measures" / model_name / f"{model_name}_logitvar_and_norm.csv"
        )

        visualize_logitvar_and_rho(
            model_name=model_name,
            logitvars_df=logitvars_and_norms_df,
            rho_df=rho_df,
            entropy_neurons=entropy_neurons_idxs,
            save_path=Path(run_path) / "entropy_neurons_figures" / model_name,
        )

    if plot_norm_distro:
        logitvars_and_norms_df = pd.read_csv(
            Path(run_path) / "entropy_neurons_measures" / model_name / f"{model_name}_logitvar_and_norm.csv"
        )
        visualize_norm_distribution(
            model_name=model_name,
            logitvars_and_norms_df=logitvars_and_norms_df,
            entropy_neurons=entropy_neurons_idxs,
            save_path=Path(run_path) / "entropy_neurons_figures" / model_name,
        )

    if plot_rho_layer_variation:
        rho_by_layer = []
        for layer in range(MODEL_NAME_TO_NB_LAYERS[model_name]):
            rho_df = pd.read_csv(
                Path(run_path) / "entropy_neurons_measures" / model_name / f"{model_name}_rho_layer={layer}.csv"
            )
            rho_df["layer"] = layer + 1
            rho_by_layer.append(rho_df)
        rho_by_layer_df = pd.concat(rho_by_layer)
        visualize_rho_layer_variation(
            model_name=model_name,
            rho_by_layer_df=rho_by_layer_df,
            save_path=Path(run_path) / "entropy_neurons_figures",
        )
