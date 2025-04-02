from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from adjustText import adjust_text

MODEL_NAME_TO_BEAUTIFUL_MODELNAME = {
    "Phi-1_5": "Phi-1.5",
    "Mistral-7B-v0.1": "Mistral-7B-v0.1",
    "gpt2-small": "GPT2-small",
    "Meta-Llama-3-8B": "Llama3-8B",
    "EleutherAI_pythia-1.4b": "Pythia-1.4B",
}


def myFormatter(tick, _):
    if tick == 1e-05:
        return "0.00001"
    elif tick == 1e-06:
        return "0.000001"
    else:
        return f"{tick:.15g}"


def plot_transition_distribution(model_data, model_name, ablation_type, savepath, proportion_measures=True):
    """
    Creates a histogram plot for a specific model and ablation type.

    Args:
        model_data (dict): Data for specific model
        model_name (str): Name of the model
        ablation_type (str): Type of ablation (mean, mode, median, mean_minus_sigma, or mean_plus_sigma)
    """
    sns.set_theme(context="paper")

    plt.figure(figsize=(8, 6))

    measure_type = "invariant_proportion" if proportion_measures else "invariant_count"

    if measure_type == "invariant_proportion":
        transition_scores = [
            (1 - random_ablation_score[measure_type]) * 100
            for random_ablation_score in model_data[ablation_type]["random_ablation_distro"]
        ]

    sns.histplot(
        transition_scores,
        color="cornflowerblue",
        alpha=0.7,
        kde=True,
        bins="auto",
        line_kws={"color": "darkblue", "lw": 5},
    )

    if measure_type == "invariant_proportion":
        en_transition_score = (1 - model_data[ablation_type]["en_only_ablation"][measure_type]) * 100

    plt.axvline(
        x=en_transition_score,
        color="red",
        linestyle="--",
        linewidth=4,
        label=f"Entropy neurons (Q-value = {100-model_data[ablation_type]['invariant_quantile_proportion']})",
    )

    plt.xlabel("Transition Score (%)", fontsize=24)
    plt.ylabel("Count", fontsize=24)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=19)
    plt.tight_layout()

    plt.savefig(
        Path(savepath)
        / "entropy_neurons_figures"
        / model_name
        / f"{model_name}_{ablation_type}_transition_score.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def visualize_norm_distribution(
    model_name: str,
    logitvars_and_norms_df: pd.DataFrame,
    entropy_neurons: list,
    save_path: str,
):
    if entropy_neurons is not None:
        logitvars_and_norms_df["neuron_type"] = logitvars_and_norms_df.neuron_idx.apply(
            lambda x: "Entropy Neurons" if x in entropy_neurons else "Other Neurons"
        )

    sns.set_theme(context="paper")

    custom_palette = ["cornflowerblue", "orangered"]

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]}, figsize=(8, 5))

    sns.histplot(
        data=logitvars_and_norms_df,
        x="norm_out_weight",
        kde=True,
        color="cornflowerblue",
        line_kws={"color": "crimson", "lw": 1},
        ax=axs[0],
    )
    axs[0].scatter(
        logitvars_and_norms_df["norm_out_weight"],
        [6] * len(logitvars_and_norms_df),
        color="cornflowerblue",
        s=20,
        alpha=0.5,
        label="Other Neurons",
    )

    sns.boxplot(
        data=logitvars_and_norms_df,
        x="norm_out_weight",
        hue="neuron_type",
        ax=axs[1],
        palette=custom_palette,
        width=0.5,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "cornflowerblue",
            "markeredgecolor": "cornflowerblue",
            "linestyle": "none",
        },
        linewidth=0.8,
        linecolor="cornflowerblue",
        showfliers=False,
    )

    if entropy_neurons is not None:
        logitvars_and_norms_entropy_neurons_df = logitvars_and_norms_df[
            logitvars_and_norms_df.neuron_type == "Entropy Neurons"
        ]

        axs[0].scatter(
            logitvars_and_norms_entropy_neurons_df["norm_out_weight"],
            [6] * len(logitvars_and_norms_entropy_neurons_df["norm_out_weight"]),
            color="red",
            s=30,
            alpha=0.7,
            label=f"Entropy Neurons\n (k = {len(logitvars_and_norms_entropy_neurons_df)} ~ p = {float(len(logitvars_and_norms_entropy_neurons_df)/logitvars_and_norms_df.shape[0]*100):.2f}% | N = {logitvars_and_norms_df.shape[0]})",
        )

        logitvars_and_norms_entropy_neurons_df["neuron_labels"] = logitvars_and_norms_entropy_neurons_df.apply(
            lambda neuron: str(int(neuron.neuron_idx)) if neuron.neuron_idx in entropy_neurons else "",
            axis=1,
        )

        axs[0].set_xlabel(r"$||w_{out}^{(i)}||$", size=20)
        axs[0].set_ylabel("Count", size=18)

        axs[1].set_xlabel(r"$||w_{out}^{(i)}||$", size=20)
        axs[1].legend(loc="best", fontsize=14)
        axs[0].legend(loc="best", fontsize=14)

    if entropy_neurons is not None:
        plt.tight_layout()
        plt.savefig(
            Path(save_path) / f"{model_name}_weight_norm_distro-nb_neurons_with_low_logitvar={len(entropy_neurons)}.png"
        )
        plt.show()
        plt.close()
    else:
        plt.tight_layout()
        plt.savefig(Path(save_path) / f"{model_name}_weight_norm_distro.png")
        plt.show()
        plt.close()


def visualize_logitvar_and_rho(
    model_name: str,
    logitvars_df: pd.DataFrame,
    rho_df: pd.DataFrame,
    entropy_neurons: list,
    save_path: str,
    formatter: FuncFormatter = myFormatter,
):
    logitvar_and_rho = pd.merge(logitvars_df, rho_df, left_on="neuron_idx", right_on="neuron_index")

    logitvar_and_rho["neuron_type"] = logitvar_and_rho.neuron_idx.apply(
        lambda x: "Entropy Neurons" if x in entropy_neurons else "Other Neurons"
    )

    sns.set_theme(context="paper")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(4, 4)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    sns.scatterplot(
        x="rho",
        y="logitvar",
        data=logitvar_and_rho,
        s=30,
        alpha=0.6,
        ax=ax_main,
        color="cornflowerblue",
        edgecolors="blue",
    )

    ax_main.set_yscale("log")

    sns.kdeplot(
        data=logitvar_and_rho,
        ax=ax_main,
        x="rho",
        y="logitvar",
        fill=False,
        color="mediumblue",
    )

    if entropy_neurons is not None:
        entropy_data = logitvar_and_rho[logitvar_and_rho.neuron_idx.isin(entropy_neurons)]
        sns.scatterplot(
            x="rho",
            y="logitvar",
            data=entropy_data,
            s=45,
            alpha=0.6,
            ax=ax_main,
            color="red",
            edgecolors="red",
            label=f"Entropy Neurons\n (k = {len(entropy_data)} ~ p = {float(100*len(entropy_data)/logitvar_and_rho.shape[0]):.2f}% | N = {logitvar_and_rho.shape[0]})",
        )

    ax_main.tick_params(axis="x", which="both", labelsize=20)
    ax_main.tick_params(axis="y", which="both", labelsize=20)
    ax_main.yaxis.set_major_formatter(formatter)
    ax_main.set_xlabel(r"$\rho_i = \frac{||V_0^Tw_{out}^{(i)}||}{||w_{out}^{(i)}||}$", size=35)
    ax_main.set_ylabel(r"LogitVar($w_{out}^{(i)}$)", size=35)

    if entropy_neurons is not None:
        legend = ax_main.legend(loc="upper right", fontsize=18)
        legend.get_frame().set_alpha(0.8)

    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    sns.kdeplot(logitvar_and_rho["rho"], ax=ax_top, color="cornflowerblue", fill=True, alpha=0.5).set(
        xlabel="", ylabel=""
    )
    ax_top.set_facecolor(sns.axes_style()["axes.facecolor"])
    ax_top.grid(True, axis="x", linestyle="--", alpha=0.7)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.yaxis.set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_top.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    sns.kdeplot(
        y=logitvar_and_rho["logitvar"],
        ax=ax_right,
        color="cornflowerblue",
        fill=True,
        alpha=0.5,
        linewidth=1,
        bw_adjust=0.5,
    ).set(xlabel="", ylabel="")

    ax_right.set_facecolor(sns.axes_style()["axes.facecolor"])
    ax_right.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_right.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    ax_main.yaxis.set_major_formatter(formatter)

    plt.subplots_adjust(top=0.95, bottom=0.05)
    ax = plt.gca()
    plt.legend(fontsize=18, loc="best")
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    print(f"save_path: {save_path}")
    if entropy_neurons is not None:
        plt.savefig(
            Path(save_path) / f"{model_name}_logitvar_and_rho-nb_neurons_with_low_logitvar={len(entropy_data)}.png"
        )
    else:
        plt.savefig(Path(save_path) / f"{model_name}_logitvar_and_rho.png")
    plt.show()
    plt.close()


def visualize_logitvar_and_norm(
    logitvar_and_norm_df: pd.DataFrame,
    model_name: str,
    save_path: str,
    entropy_neurons_idxs: dict,
    all_layers: bool = False,
    custom_formatter=myFormatter,
) -> None:
    """
    Visualize logit variance and norm, with the option to plot for all layers or a specific layer.

    Args:
        logitvar_and_norm (dict): Nested dictionary containing logitvars and norms for each layer.
        model_name (str): Name of the model.
        save_path (str): Path to save the plots.
        all_layers (bool): If True, plot for all layers in the data. Default is False.
        entropy_neurons_idxs (dict): a dict with keys in the form layer{layer_idx} and values containing the index of the selected entropy neurons

    Returns:
        None
    """

    formatter = FuncFormatter(custom_formatter)

    sns.set_theme(context="paper")

    if all_layers:
        unique_layers = sorted(logitvar_and_norm_df["layer"].unique())
        n_layers = len(unique_layers)

        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False)

        max_ylim = (float("inf"), float("-inf"))

        for idx, current_layer in enumerate(unique_layers):
            row, col = divmod(idx, n_cols)

            data = logitvar_and_norm_df[logitvar_and_norm_df["layer"] == current_layer]

            if entropy_neurons_idxs is not None:
                entropy_neurons = entropy_neurons_idxs.get(f"layer{current_layer}", [])

            ax_main = axes[row][col]
            sns.scatterplot(
                x="norm_out_weight",
                y="logitvar",
                data=data,
                s=30,
                alpha=0.6,
                ax=ax_main,
                color="cornflowerblue",
                edgecolors="blue",
            )
            ax_main.set_yscale("log")
            sns.kdeplot(data=data, ax=ax_main, x="norm_out_weight", y="logitvar", fill=False, color="mediumblue")
            # Highlight entropy neurons in red
            if entropy_neurons_idxs is not None:
                entropy_data = data.iloc[entropy_neurons]
                sns.scatterplot(
                    x="norm_out_weight",
                    y="logitvar",
                    data=entropy_data,
                    s=30,
                    alpha=0.6,
                    ax=ax_main,
                    color="red",
                    edgecolors="red",
                    label=f"Entropy Neurons\n (k = {len(entropy_data)} ~ p = {float(len(entropy_data)/data.shape[0]*100):.2f}% | N = {data.shape[0]})",
                )

            current_ylim = ax_main.get_ylim()
            max_ylim = (min(max_ylim[0], current_ylim[0]), max(max_ylim[1], current_ylim[1]))
            ax_main.set_title(f"Layer {current_layer + 1}", fontsize=15)
            ax_main.set_ylabel(r"$\text{LogitVar(}w_{out}\text{)}$", fontsize=35)
            ax_main.set_xlabel(r"$||w_{out}||$", fontsize=35)
            if entropy_neurons_idxs is not None:
                if idx == len(unique_layers) - 1:
                    legend = ax_main.legend(loc="best", fontsize=16)
                    legend.get_frame().set_alpha(0.8)

            ax_top = ax_main.inset_axes([0, 1.05, 1, 0.3], sharex=ax_main)
            sns.kdeplot(data["norm_out_weight"], ax=ax_top, color="cornflowerblue", fill=True, alpha=0.5).set(
                xlabel="", ylabel=""
            )
            ax_top.set_facecolor(sns.axes_style()["axes.facecolor"])
            ax_top.grid(True, axis="x", linestyle="--", alpha=0.7)
            ax_top.spines["left"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            ax_top.spines["bottom"].set_visible(False)
            ax_top.yaxis.set_visible(False)
            ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax_top.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

            ax_right = ax_main.inset_axes([1.05, 0, 0.3, 1], sharey=ax_main)
            sns.boxplot(
                y=data["logitvar"],
                ax=ax_right,
                color="cornflowerblue",
                width=0.5,
                flierprops={
                    "marker": "o",
                    "markersize": 3,
                    "markerfacecolor": "cornflowerblue",
                    "markeredgecolor": "cornflowerblue",
                    "linestyle": "none",
                },
                linewidth=0.8,
                linecolor="cornflowerblue",
            ).set(xlabel="", ylabel="")

            ax_right.set_facecolor(sns.axes_style()["axes.facecolor"])
            ax_right.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax_right.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
            ax_main.yaxis.set_major_formatter(formatter)

        for idx in range(len(unique_layers), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].axis("off")

        for row_axes in axes:
            for ax in row_axes:
                if ax.has_data():
                    ax.set_ylim(max_ylim)

        plt.tight_layout()
        plt.savefig(
            Path(save_path)
            / f"{'entropy_neurons_' if entropy_neurons_idxs is not None else ''}scatterplot_logitvar_norm_{model_name}_all_layers_grid.pdf"
        )
        plt.close()

    else:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(4, 4)

        ax_main = fig.add_subplot(gs[1:4, 0:3])
        sns.scatterplot(
            x="norm_out_weight",
            y="logitvar",
            data=logitvar_and_norm_df,
            s=30,
            alpha=0.6,
            ax=ax_main,
            color="cornflowerblue",
            edgecolors="blue",
        )

        ax_main.set_yscale("log")

        sns.kdeplot(
            data=logitvar_and_norm_df, ax=ax_main, x="norm_out_weight", y="logitvar", fill=False, color="mediumblue"
        )

        # Highlight entropy neurons in red
        if entropy_neurons_idxs is not None:
            entropy_data = logitvar_and_norm_df.iloc[entropy_neurons_idxs]
            sns.scatterplot(
                x="norm_out_weight",
                y="logitvar",
                data=entropy_data,
                s=45,
                alpha=0.6,
                ax=ax_main,
                color="red",
                edgecolors="red",
                label=f"Entropy Neurons\n (k = {len(entropy_data)} ~ p = {float(100*len(entropy_data)/logitvar_and_norm_df.shape[0]):.2f}% | N = {logitvar_and_norm_df.shape[0]})",
            )

            texts = []
            for i in range(len(entropy_data)):
                if model_name == "gpt2-small":
                    texts.append(
                        ax_main.text(
                            entropy_data["norm_out_weight"].iloc[i],
                            entropy_data["logitvar"].iloc[i] + entropy_data["logitvar"].iloc[i] * 0.04,
                            str(entropy_data["neuron_idx"].iloc[i]),
                            fontsize=8,
                            fontweight="bold",
                            color="red",
                            ha="center",
                            va="bottom",
                            bbox=dict(facecolor="white", alpha=0, edgecolor="none", boxstyle="round"),
                        )
                    )
                else:
                    texts.append(
                        ax_main.text(
                            entropy_data["norm_out_weight"].iloc[i],
                            entropy_data["logitvar"].iloc[i] + entropy_data["logitvar"].iloc[i] * 0.04,
                            str(entropy_data["neuron_idx"].iloc[i]),
                            fontsize=14,
                            fontweight="bold",
                            color="red",
                            ha="center",
                            va="bottom",
                            bbox=dict(facecolor="white", alpha=0, edgecolor="none", boxstyle="round"),
                        )
                    )

            adjust_text(
                texts,
                ax=ax_main,
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5, connectionstyle="arc3,rad=0.2"),
                expand_text=(3, 3),
                force_text=(0.5, 0.5),
                expand_points=(3, 3),
            )

        ax_main.tick_params(axis="x", which="both", labelsize=20)
        ax_main.tick_params(axis="y", which="both", labelsize=20)

        ax_main.yaxis.set_major_formatter(formatter)

        ax_main.set_ylabel(r"$\text{LogitVar(}w_{out}\text{)}$", fontsize=35)
        ax_main.set_xlabel(r"$||w_{out}||$", fontsize=35)
        if entropy_neurons_idxs is not None:
            legend = ax_main.legend(loc="upper right", fontsize=18)
            legend.get_frame().set_alpha(0.5)

        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        sns.kdeplot(
            logitvar_and_norm_df["norm_out_weight"], ax=ax_top, color="cornflowerblue", fill=True, alpha=0.5
        ).set(xlabel="", ylabel="")
        ax_top.set_facecolor(sns.axes_style()["axes.facecolor"])
        ax_top.grid(True, axis="x", linestyle="--", alpha=0.7)
        ax_top.spines["left"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.yaxis.set_visible(False)
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_top.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        sns.boxplot(
            y=logitvar_and_norm_df["logitvar"],
            ax=ax_right,
            color="cornflowerblue",
            width=0.5,
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "cornflowerblue",
                "markeredgecolor": "cornflowerblue",
                "linestyle": "none",
            },
            linewidth=0.8,
            linecolor="cornflowerblue",
        ).set(xlabel="", ylabel="")

        ax_right.set_facecolor(sns.axes_style()["axes.facecolor"])
        ax_right.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_right.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax_main.yaxis.set_major_formatter(formatter)

        plt.subplots_adjust(top=0.95, bottom=0.05)

        plt.tight_layout()
        print(f"save_path: {save_path}")
        plt.savefig(
            Path(save_path)
            / f"{model_name}_{'entropy_neurons_' if entropy_neurons_idxs is not None else ''}scatterplot_logitvar_norm.pdf"
        )
        plt.close()


def visualize_norm_and_logitvar_layer_variation(
    logitvar_and_norm_out_weights_pd: pd.DataFrame, model_name: str, save_path: str, custom_formatter=myFormatter
):
    formatter = FuncFormatter(custom_formatter)

    for fig_type, fig_ylabel in {"norm": r"$||w_{out}||$", "logitvar": r"$\text{LogitVar(}w_{out}\text{)}$"}.items():
        mean_values = logitvar_and_norm_out_weights_pd[f"mean_{fig_type}_distro"].values.astype(float)
        std_values = logitvar_and_norm_out_weights_pd[f"sample_std_{fig_type}_distro"].values.astype(float)

        flattened_norm_distro = np.concatenate(logitvar_and_norm_out_weights_pd[f"{fig_type}_distro"].values)
        layer_labels = np.repeat(
            np.arange(1, len(mean_values) + 1), logitvar_and_norm_out_weights_pd[f"{fig_type}_distro"].apply(len)
        )

        boxplot_data = pd.DataFrame({"layer": layer_labels, "value": flattened_norm_distro})

        sns.set_theme(context="paper")

        x_values = np.arange(0, len(mean_values))

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(
            x_values,
            mean_values,
            marker="o",
            linestyle="-",
            color="cornflowerblue",
            label="Mean",
            linewidth=3,
            markersize=8,
        )

        ax.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            color="cornflowerblue",
            alpha=0.2,
            label="Standard Deviation",
        )

        sns.boxplot(
            x="layer",
            y="value",
            data=boxplot_data,
            color="cornflowerblue",
            linecolor="cornflowerblue",
            width=0.35,
            fliersize=1.5,
            ax=ax,
        )

        ax.set_xlabel("Layer", fontsize=22)
        ax.set_ylabel(fig_ylabel, fontsize=22)

        ax.set_xticks(x_values)
        ax.set_xticklabels([i + 1 for i in x_values], fontsize=18)

        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticks(), fontsize=18)
        plt.yscale("log")
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(axis="y", linestyle="--", alpha=0.7)

        ax.legend(fontsize=20)

        plt.tight_layout()

        plt.savefig(Path(save_path) / f"{fig_type}_layer_variation.pdf")
        plt.close()


def visualize_rho_layer_variation(rho_by_layer_df: list, save_path: str, model_name: str):
    sns.set_theme(context="paper")

    nb_layers = len(rho_by_layer_df.layer.unique())
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=rho_by_layer_df,
        x="layer",
        y="rho",
        color="cornflowerblue",
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "cornflowerblue",
            "markeredgecolor": "cornflowerblue",
            "linestyle": "none",
        },
        linewidth=0.8,
        linecolor="cornflowerblue",
    )
    sns.boxplot(
        data=rho_by_layer_df[rho_by_layer_df.layer == nb_layers],
        x=nb_layers,
        y="rho",
        color="orangered",
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "orangered",
            "markeredgecolor": "orangered",
            "linestyle": "none",
        },
        linewidth=0.8,
        linecolor="orangered",
    )

    plt.ylabel(r"$\rho_i = \frac{||V_0^Tw_{out}^{(i)}||}{||w_{out}^{(i)}||}$", fontsize=25)
    plt.xlabel("Layer", fontsize=25)

    plt.savefig(Path(save_path) / f"{model_name}_rho_layer_variation.png")

    plt.show()
