<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2408.08656-b31b1b.svg)](https://arxiv.org/abs/2509.10663)
[![Website](https://img.shields.io/badge/InductionModulation-%F0%9F%8C%90Website-purple?style=flat)](https://zineddine-tighidet.github.io/projects/project-emnlp2025.html)

<h1>Context Copying Modulation:
<br>
The Role of Entropy Neurons in Managing Parametric and Contextual Knowledge Conflicts</h1>
<div>
    <a href='https://zineddine-tighidet.github.io/' target='_blank'>Zineddine Tighidet</a><sup>1,2</sup>&emsp;
    <a>Andrea Mogini</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=IFLcfvUAAAAJ&hl=fr', target='_blank'>Hedi Ben-younes</a><sup>1</sup>&emsp;
    <a href='https://www.jialimei.me', target='_blank'>Jiali Mei</a><sup>1</sup>&emsp;
    <br>
    <a href='https://pages.isir.upmc.fr/gallinari/' target='_blank'>Patrick Gallinari</a><sup>2,3</sup>&emsp;
    <a href='https://www.piwowarski.fr' target='_blank'>Benjamin Piwowarski</a><sup>2</sup>&emsp;
</div>
<br>
<div>
    <sup>1</sup>BNP Paribas, Paris, France&emsp;<br>
    <sup>2</sup>Sorbonne Université, CNRS, ISIR, F-75005 Paris, France&emsp;<br>
    <sup>3</sup>Criteo AI Lab, Paris, France&emsp;
</div>
<br>

Correspondence to: *zineddine.tighidet*@{*bnpparibas.com*, *sorbonne-universite.fr*}

<br>

<img src="input_data/entropy_neurons_schema.png" width="70%"/>
<img src="input_data/entropy_neurons_mechanism_on_induction.gif" width="70%"/>

</div>

# Abstract

The behavior of Large Language Models (LLMs) when facing contextual information that conflicts with their internal parametric knowledge is inconsistent, with no generally accepted explanation for the expected outcome distribution. Recent work has identified in autoregressive transformer models a class of neurons -- called *entropy neurons* -- that produce a significant effect on the model while having an overall moderate impact on the ranking of the predicted tokens. In this paper, we investigate the preliminary claim that these neurons are involved in inhibiting context copying behavior in transformers by looking at their role in resolving conflicts between contextual and parametric information. We show that *entropy neurons* are responsible for suppressing context copying across a range of LLMs, and that ablating them leads to a substantial change in the generation process. These results enhance our understanding of the internal dynamics of LLMs when handling conflicting information.

# Supported language models

Experiments can be executed on the following language models:

- Phi-1.5
- GPT-2-small
- Llama-3-8B
- Mistral-7B-v0.1
- Pythia-1.4B

# Steps to Reproduce the Results

This repository requires Python version 3.9.17.

## 1. Install framework package:

First create a conda environment:
```sh
conda create --name context_copying_modulation python=3.9.17
conda activate context_copying_modulation
```

then install the requirements:
```sh
pip install -r project-requirements.txt
```

**Important:** set up the `HUGGINGFACE_TOKEN` constant in the `src/model.py` module with your huggingface access token to load the LLMs from the huggingface hub.

Finally install the package with `pip`:
```sh
pip install .
```

## 2. Reproduce the results in the paper:

**Important:** before running the `pipeline.sh` script, make sure to add execution permission:
```sh
chmod +x ./scripts/entropy_neurons_scripts/pipeline.sh
```

### For all the models:
```sh
./scripts/entropy_neurons_scripts/pipeline.sh --ablation_value <ablation_value> --device <device>
```

### For a specific model:
```sh
./scripts/entropy_neurons_scripts/pipeline.sh --model_name <model_name> --ablation_value <ablation_value> --device <device>
```

### Arguments

- `<model_name>`: the name of the model. Options are `gpt2-small`, `EleutherAI_pythia-1.4b`, `Phi-1_5`, `Mistral-7B-v0.1`, `Meta-Llama-3-8B`.
- `<device>`: the device on which to load the LLMs (`cuda` or `cpu`)
- `<ablation_value>`: the value to use for ablation. Options are `mean_ablation` (used in the main paper), `mode_ablation`, `median_ablation`, `mean_minus_sigma_ablation`, `mean_plus_sigma_ablation`

## 3. Paper Results:

After running the `pipeline.sh` script, you can find the paper figures and tables in the `<ablation_value>-<model_name>-datetime` or `all_ablation_values-all_models-datetime` folder depending if the scripts was performed on one model or all the models/all the ablation values (see above).

### Figures:

The figure can be found in the `entropy_neurons_figures` sub-folder:

- `entropy_neurons_figures/<model_name>/<model_name>_logitvar_and_rho-nb_neurons_with_low_logitvar=<nb_selected_entropy_neurons>.png`: the selected entropy neurons for `<model_name>` displayed on the $LogitVar$ and $\rho$ axis.
- `entropy_neurons_figures/<model_name>/singular_values_and_null_space_nb_dims=<null_space_dimension>.pdf`: figure with the singular values and the selected null space (in red) for the unembedding matrix $W_U$.
- `entropy_neurons_figures/<model_name>/<model_name>_weight_norm_distro-nb_neurons_with_low_logitvar=<nb_selected_entropy_neurons>.png`: the weight norm distribution for entropy neurons (red) and other neurons (blue).
- `entropy_neurons_figures/<model_name>/<model_name>_<ablation_value>-conversion_ratios.pdf`: the Conversion Ratio figure for all the knowledge categories (CK, PK, and ND).
- `entropy_neurons_figures/<model_name>/<model_name>_<ablation_value>_transition_score.png`: the Global Transition Score histogram for random neurons (blue bars) and entropy neurons (red dashed line).

### Hardware:

The experiments in the paper where performed on NVIDIA A100 and H100 GPUs each equiped with 80GB of VRAM.

## Citation:
```
@misc{tighidet2025contextcopyingmodulationrole,
      title={Context Copying Modulation: The Role of Entropy Neurons in Managing Parametric and Contextual Knowledge Conflicts}, 
      author={Zineddine Tighidet and Andrea Mogini and Hedi Ben-younes and Jiali Mei and Patrick Gallinari and Benjamin Piwowarski},
      year={2025},
      eprint={2509.10663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.10663}, 
}
```
