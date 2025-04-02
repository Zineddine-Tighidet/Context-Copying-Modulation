from tqdm import tqdm
from pathlib import Path
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
from statistics import mean, stdev
import numpy as np
import torch

MODEL_NAME_TO_NULL_SPACE_START_INDEX = {
    "gpt2-small": -40,
    "Meta-Llama-3-8B": -96,
    "Phi-1_5": -48,
    "Mistral-7B-v0.1": -96,
    "EleutherAI_pythia-1.4b": -48,
}

LOGITVAR_PROPORTION = 1
RHO_PROPORTION = 0.4

def compute_logitvar_and_norm(
    model_name: str,
    W_out: torch.Tensor,
    W_u: torch.Tensor,
    d: int,
    N: int,
    V: int,
    L: int,
    compute_on_all_layers: bool,
    layer: int,
    savepath: str,
) -> dict:
    """Compute the LogitVar and the norm for all neurons output weights w_out:

    For efficiency reasons, I compute the matrix multiplication between W_u and W_out, the neurons' output weights
    and then use the columns of the resulting matrix to compute the LogitVars rather than recomputing the LogitVar
    for each neuron separately.

    Here is the formula to compute the LogitVar for a specific neuron:

    LogitVar(w_out) = Var(W_u * w_out/Norm(W_u)_d=1 * Norm(w_out))

    where Norm(W_u)_d=1 is the column-wise norm of W_u

    Reference paper (Confidence Regulation Neurons in Language Models -- NeurIPS 2024): https://arxiv.org/abs/2406.16254

    Args:
        W_out (torch.Tensor): the output weights of all neurons -- dimension LxNxd where N is the number of neurons and L the number of layers
        W_u (torch.Tensor): the unembedding matrix of dimension dx|V|
        d: the model's hidden dimension
        N: the mlp hidden layer (number of neurons)
        V: the vocab size
        compute_on_all_layers: whether to compute the LogitVar and norm on the neurons of all layers
        L: the number of layers
        layer: the layer on which to select entropy neurons (ignored if compute_on_all_layers is True)

    Returns:
        dict: returns a dict with the LogitVars and output weight norms.
    """

    assert W_u.shape == (d, V), f"W_u must be of shape ({d}, {V}) but the provided shape is {W_u.shape}"
    assert W_out.shape == (L, N, d), f"W_out must be of shape ({L}, {N}, {d}) but the provided shape is {W_out.shape}"

    W_out_ = W_out.clone()  # make a deep copy to avoid changing the model weights
    if compute_on_all_layers:
        print("Computing the LogitVars and Weight Norms the neurons of all layers")
        W_out_ = W_out_.reshape(L * N, d)
    else:
        print(f"Computing the LogitVars and Weight Norms for layer {layer}")
        W_out_ = W_out_[layer]

    logitvar_and_out_weight_norms_neurons = []
    norm_W_u = W_u.norm(dim=0)
    norm_W_out = W_out_.norm(dim=-1)
    W_out_W_u = W_out_ @ W_u
    for neuron_idx in tqdm(
        range(W_out_.shape[0]), desc="Computing LogitVar and Norm for each neuron", total=W_out_.shape[0]
    ):
        norm_w_out_neuron = norm_W_out[neuron_idx]
        logitvar_and_out_weight_norms_neurons.append(
            {
                "neuron_idx": neuron_idx,
                "logitvar": float((W_out_W_u[neuron_idx] / (norm_W_u * norm_w_out_neuron)).var().cpu()),
                "norm_out_weight": float(norm_w_out_neuron.cpu()),
            }
        )

    logitvar_and_out_weight_norms_neurons_df = pd.DataFrame(logitvar_and_out_weight_norms_neurons)

    logitvar_and_out_weight_norms_neurons_df.to_csv(Path(savepath) / f"{model_name}_logitvar_and_norm.csv")

    return logitvar_and_out_weight_norms_neurons_df


def compute_logitvar_and_norm_layer_statistics(logitvars_and_norms: dict) -> dict:
    logitvars_and_norm_out_weights = {}
    for layer in logitvars_and_norms:
        norm_distro = [neuron["norm_out_weight"] for neuron in logitvars_and_norms[layer]]

        mean_norm_distro = mean(norm_distro)
        sample_std_norm_distro = stdev(norm_distro)

        logitvar_distro = [neuron["logitvar"] for neuron in logitvars_and_norms[layer]]
        mean_logitvar_distro = mean(logitvar_distro)
        sample_std_logitvar_distro = stdev(logitvar_distro)

        logitvars_and_norm_out_weights[layer] = {
            "norm_distro": norm_distro,
            "mean_norm_distro": mean_norm_distro,
            "sample_std_norm_distro": sample_std_norm_distro,
            "logitvar_distro": logitvar_distro,
            "mean_logitvar_distro": mean_logitvar_distro,
            "sample_std_logitvar_distro": sample_std_logitvar_distro,
        }

    logitvar_and_norm_out_weights_pd = pd.DataFrame(logitvars_and_norm_out_weights).T

    return logitvar_and_norm_out_weights_pd


def compute_rho(
    W_u,
    W_out,
    N,
    model_name,
    sing_values_figures_savepath,
    rho_savepath,
    model_name_to_null_space_start_index=MODEL_NAME_TO_NULL_SPACE_START_INDEX,
    layer=-1,
    save_by_layer=False,
    null_space_end_index=0,
):
    # take svd of W_U
    U, S, V = torch.linalg.svd(W_u, full_matrices=False)

    null_space_start_index = model_name_to_null_space_start_index[model_name]

    # plot the singular values as a function of the singular vector indexes (used to identify the
    # drop of the null space)
    sns.set_theme(context="paper")
    sns.lineplot(x=np.arange(S.shape[0]), y=S.detach().cpu(), linewidth=2)
    sns.lineplot(
        x=np.arange(S.shape[0])[null_space_start_index:],
        y=S.detach().cpu()[null_space_start_index:],
        color="r",
        label="Null Space",
        linewidth=4,
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.xlabel("Singular Vector Index", fontsize=15)
    plt.ylabel("Singular Value", fontsize=15)
    if save_by_layer:
        plt.savefig(
            Path(sing_values_figures_savepath)
            / f"singular_values_and_null_space_nb_dims={abs(null_space_start_index)}_layer={layer}.pdf"
        )
    else:
        plt.savefig(
            Path(sing_values_figures_savepath)
            / f"singular_values_and_null_space_nb_dims={abs(null_space_start_index)}.pdf"
        )
    plt.close()

    # make scatter plot of W_out[-1] @ U_entropy and W_out[-1].norm()
    U_entropy = range(null_space_start_index, null_space_end_index)
    norm = W_out[layer].norm(dim=-1)
    norm_fraction_on_U_entropy = (W_out[layer] @ U)[:, U_entropy].norm(dim=-1) / norm

    # make dataframe
    df = pd.DataFrame(
        {
            "rho": norm_fraction_on_U_entropy.detach().cpu(),
            "norm": norm.detach().cpu(),
            "neuron_index": np.arange(N),
        }
    )

    if save_by_layer:
        df.to_csv(Path(rho_savepath) / f"{model_name}_rho_layer={layer}.csv")
    else:
        df.to_csv(Path(rho_savepath) / f"{model_name}_rho.csv")
    return df


def identify_entropy_neurons(
    logitvars_df: pd.DataFrame,
    rho_df: pd.DataFrame,
    model_name: str,
    savepath: str,
    nb_neurons_to_select: int,
    logitvar_proportion: float = 0.025,
    rho_proportion: float = 0.8,
) -> dict:
    """this function returns a list of entropy neurons indices

    Args:
        rho_proportion (float): the minimal proportion of fraction of norm in the null space
            to use to select entropy neurons
        logitvar_proportion (float): the maximal proportion of logitvar to select the entropy
            neurons

    Returns:
        dict: a list of entropy neurons indices
    """

    logitvars_and_rho = pd.merge(logitvars_df, rho_df, left_on="neuron_idx", right_on="neuron_index")

    entropy_neurons = logitvars_and_rho[logitvars_and_rho["rho"] >= rho_proportion]
    entropy_neurons = entropy_neurons.sort_values("logitvar", ascending=True)
    entropy_neurons = entropy_neurons.head(n=nb_neurons_to_select)

    entropy_neurons_idxs = entropy_neurons.neuron_idx.tolist()

    with open(
        Path(savepath)
        / f"{model_name}_entropy_neurons-rho_prop={rho_proportion}-logitvar_prop={logitvar_proportion}-nb_neurons={nb_neurons_to_select}.json",
        "w+",
    ) as fo:
        json.dump(entropy_neurons.to_dict(orient="index"), fo)

    return entropy_neurons_idxs
