import argparse
import transformer_lens
from transformer_lens import HookedTransformer
from pathlib import Path

from src.model import MODEL_PATHS
from src.entropy_neurons.neurons_measures import (
    compute_logitvar_and_norm,
    compute_rho,
    identify_entropy_neurons,
    LOGITVAR_PROPORTION,
    RHO_PROPORTION,
)
from src.utils import setup_directories
from src.entropy_neurons.constants import PATH, MODEL_NAMES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computing neuron measures", add_help=True)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[name for name in MODEL_PATHS.keys()],
        help="Model name to build the knowledge datasets (if not specified, the measure for all the models are computed.)",
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="The device name on which to load the models."
    )
    parser.add_argument("--compute_rho_by_layer", action="store_true")
    parser.add_argument("--datetime", type=str, required=True)
    parser.add_argument("--nb_entropy_neurons_to_select", type=int)
    args = parser.parse_args()

    PERMANENT_PATH = Path(PATH) / args.datetime
    nb_entropy_neurons_to_select = args.nb_entropy_neurons_to_select

    model_names = MODEL_NAMES

    if args.model_name is not None:
        model_names = [args.model_name]

    for model_name in model_names:
        setup_directories(permanent_path=PERMANENT_PATH, model_name=model_name)

        model_path = MODEL_PATHS[model_name]
        transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(model_path)
        model = HookedTransformer.from_pretrained(
            model_name=model_path, device="cuda" if args.device is None else args.device
        )

        logitvars_df = compute_logitvar_and_norm(
            model_name=model_name,
            compute_on_all_layers=False,
            d=model.cfg.d_model,
            L=model.cfg.n_layers,
            layer=model.cfg.n_layers - 1,
            N=model.cfg.d_mlp,
            savepath=Path(PERMANENT_PATH) / "entropy_neurons_measures" / model_name,
            V=model.cfg.d_vocab,
            W_out=model.W_out,
            W_u=model.W_U,
        )

        if args.compute_rho_by_layer:
            for layer in range(model.cfg.n_layers):
                rho_df = compute_rho(
                    W_u=model.W_U,
                    W_out=model.W_out,
                    layer=layer,
                    save_by_layer=True,
                    model_name=model_name,
                    N=model.cfg.d_mlp,
                    rho_savepath=Path(PERMANENT_PATH) / "entropy_neurons_measures" / model_name,
                    sing_values_figures_savepath=Path(PERMANENT_PATH) / "entropy_neurons_figures" / model_name,
                )
        rho_df = compute_rho(
            W_u=model.W_U,
            W_out=model.W_out,
            layer=model.cfg.n_layers - 1,
            model_name=model_name,
            N=model.cfg.d_mlp,
            rho_savepath=Path(PERMANENT_PATH) / "entropy_neurons_measures" / model_name,
            sing_values_figures_savepath=Path(PERMANENT_PATH) / "entropy_neurons_figures" / model_name,
        )

        if args.nb_entropy_neurons_to_select is not None:
            identify_entropy_neurons(
                logitvars_df=logitvars_df,
                rho_df=rho_df,
                logitvar_proportion=LOGITVAR_PROPORTION,
                rho_proportion=RHO_PROPORTION,
                nb_neurons_to_select=nb_entropy_neurons_to_select,
                model_name=model_name,
                savepath=Path(PERMANENT_PATH) / "entropy_neurons_measures" / model_name,
            )
