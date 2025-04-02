from src.entropy_neurons.neurons_visualizations import MODEL_NAME_TO_BEAUTIFUL_MODELNAME

NECESSARY_LATEX_PACKAGE_FOR_TABLES = """
\\documentclass{article}
\\usepackage{float}
\\usepackage{colortbl}
\\usepackage{makecell}
\\usepackage{amsmath}
\\usepackage{caption}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{multirow}

\\definecolor{lightgray}{gray}{0.95}
"""


def generate_latex_synthetic_table_proportions(
    control_experiment_scores,
    en_only_ablation_transition_counts,
    en_only_ablation_transition_proportions,
    ablation_type,
    model_names,
    latex_packages=NECESSARY_LATEX_PACKAGE_FOR_TABLES,
):
    latex_code = latex_packages

    latex_code += "\\begin{document}\n"

    latex_code += """\\begin{table}[H]
    \\centering
    \\footnotesize
    \\resizebox{\\textwidth}{!}{
    \\begin{tabular}{l | ccc | ccc | ccc}
    \\hline
    \\rowcolor[gray]{0.95}
    \\textbf{Model Name} & \\multicolumn{3}{c|}{\\textbf{From CK}} & \\multicolumn{3}{c|}{\\textbf{From ND}} & \\multicolumn{3}{c}{\\textbf{From PK}} \\\\
    \\rowcolor[gray]{0.95} & \\textbf{To CK} & \\textbf{To ND} & \\textbf{To PK} & \\textbf{To CK} & \\textbf{To ND} & \\textbf{To PK} & \\textbf{To CK} & \\textbf{To ND} & \\textbf{To PK} \\\\
    \\hline
    """

    # Add rows for each model
    for model_name in model_names:
        # Extract necessary data
        en_counts = en_only_ablation_transition_counts[model_name][ablation_type]["counts"]
        en_counts.index = ["CK", "ND", "PK"]
        en_props = en_only_ablation_transition_proportions[model_name][ablation_type]["proportions"]
        en_props.index = ["CK", "ND", "PK"]

        mean_proportion = control_experiment_scores[model_name][ablation_type]["proportions"][
            "random_ablation_transition_tables_mean"
        ]
        se_table_prop = control_experiment_scores[model_name][ablation_type]["proportions"][
            "random_ablation_standard_error_table"
        ]

        latex_code += "\\renewcommand{\\arraystretch}{2.5}\n"  # Increase row height *before* the data row
        latex_code += f"\\makecell{{{MODEL_NAME_TO_BEAUTIFUL_MODELNAME[model_name]}}} "
        for from_state in ["CK", "ND", "PK"]:
            for to_state in ["CK", "ND", "PK"]:
                # EN Ablation
                prop = en_props.loc[from_state, to_state] * 100
                mean_percentage = mean_proportion.loc[from_state, to_state] * 100
                se = 3 * se_table_prop.loc[from_state, to_state] * 100
                cell_color = ""
                latex_code += (
                    f"& {cell_color}\\makecell{{${prop:.1f}$\\\\ \\scriptsize (${mean_percentage:.1f}\\pm {se:.1f}$) }}"
                )
        latex_code += "\\\\ \\hline\n"

    latex_code += f"""
    \\end{{tabular}}
    }}
    \\caption{{Transition counts for \\texttt{{{ablation_type}}} ablation across models.}}
    \\label{{table:{ablation_type}_ablation_transition_table}}
    \\end{{table}}
    """

    latex_code += "\\end{document}\n"

    return latex_code


def generate_latex_synthetic_table_counts(
    control_experiment_scores,
    en_only_ablation_transition_counts,
    en_only_ablation_transition_proportions,
    ablation_type,
    model_names,
    latex_packages=NECESSARY_LATEX_PACKAGE_FOR_TABLES,
):
    latex_code = latex_packages

    latex_code += "\\begin{document}\n"

    latex_code += """\\begin{table}[H]
    \\centering
    \\footnotesize
    \\resizebox{\\textwidth}{!}{
    \\begin{tabular}{l | ccc | ccc | ccc}
    \\hline
    \\rowcolor[gray]{0.95}
    \\textbf{Model Name} & \\multicolumn{3}{c|}{\\textbf{From CK}} & \\multicolumn{3}{c|}{\\textbf{From ND}} & \\multicolumn{3}{c}{\\textbf{From PK}} \\\\
    \\rowcolor[gray]{0.95} & \\textbf{To CK} & \\textbf{To ND} & \\textbf{To PK} & \\textbf{To CK} & \\textbf{To ND} & \\textbf{To PK} & \\textbf{To CK} & \\textbf{To ND} & \\textbf{To PK} \\\\
    \\hline
    """

    # Add rows for each model
    for model_name in model_names:
        # Extract necessary data
        en_counts = en_only_ablation_transition_counts[model_name][ablation_type]["counts"]
        en_counts.index = ["CK", "ND", "PK"]
        en_props = en_only_ablation_transition_proportions[model_name][ablation_type]["proportions"]
        en_props.index = ["CK", "ND", "PK"]

        mean_counts = control_experiment_scores[model_name][ablation_type]["counts"][
            "random_ablation_transition_tables_mean"
        ]
        se_table_count = control_experiment_scores[model_name][ablation_type]["counts"][
            "random_ablation_standard_error_table"
        ]
        latex_code += "\\renewcommand{\\arraystretch}{2.5}\n"  # Increase row height *before* the data row
        latex_code += f"\\makecell{{{MODEL_NAME_TO_BEAUTIFUL_MODELNAME[model_name]}}} "
        for from_state in ["CK", "ND", "PK"]:
            for to_state in ["CK", "ND", "PK"]:
                # EN Ablation
                count = en_counts.loc[from_state, to_state]
                mean_count = mean_counts.loc[from_state, to_state]
                se = 3 * se_table_count.loc[from_state, to_state]
                cell_color = ""
                latex_code += (
                    f"& {cell_color}\\makecell{{${count:.1f}$\\\\ \\scriptsize (${mean_count:.1f}\\pm {se:.1f}$) }}"
                )
        latex_code += "\\\\ \\hline\n"

    latex_code += f"""
    \\end{{tabular}}
    }}
    \\caption{{Transition counts for \\texttt{{{ablation_type}}} ablation across models.}}
    \\label{{table:{ablation_type}_ablation_transition_table}}
    \\end{{table}}
    """

    latex_code += "\\end{document}\n"

    return latex_code


def generate_latex_table_invariance_scores_ablation_value_wise(
    data, ablation_types, model_names, latex_packages=NECESSARY_LATEX_PACKAGE_FOR_TABLES
):
    """
    Creates a printable LaTeX table string for multiple ablation types.
    """

    latex_str = latex_packages

    latex_str += "\\begin{document}\n"

    latex_str += r"""
\begin{table}[ht]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{|l|c|c|c|c|}
    \toprule
    \textbf{Ablation Value} & \textbf{Model} & \textbf{EN Transition Score (\%)} & \textbf{Q-val} & \textbf{Random Transition Score (\%)} \\
    \midrule"""

    for ablation_type in ablation_types:
        latex_str += f"""
    \multirow{{{len(model_names)}}}{{*}}{{{ablation_type.render_table_label()}}}"""
        for model in model_names:
            if model in data and ablation_type.value in data[model]:
                model_data = data[model][ablation_type.value]
                # Calculate random invariance mean
                random_ablation_proportions_distro = [
                    invariant_scores["invariant_proportion"]
                    for invariant_scores in model_data["random_ablation_distro"]
                ]
                random_inv = sum(random_ablation_proportions_distro) / len(random_ablation_proportions_distro)
                latex_str += f"""
            & {MODEL_NAME_TO_BEAUTIFUL_MODELNAME[model]} & {(1 - model_data['en_only_ablation']["invariant_proportion"])*100:.1f} & {100-model_data['invariant_quantile_proportion']:.1f} & {(1 - random_inv)*100:.1f} \\\\"""

        latex_str += "\n\\hline \n"

    latex_str += r"""
    \end{tabular}%
    }
    \caption{Ablation value-wise Transition Scores (\%) for entropy neurons ablation and averaged 100 random ablations.}
    \label{tab:ablation-invariance-scores}
\end{table}
"""

    latex_str += "\\end{document}\n"

    return latex_str
