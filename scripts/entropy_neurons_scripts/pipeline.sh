#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [--model_name <model_name>] [--datetime <datetime>] [--ablation_value <ablation_value>] [--device <device>]"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --datetime) datetime="$2"; shift ;;
        --ablation_value) ablation_value="$2"; shift ;;
        --device) device="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Set current_datetime to the provided datetime or the current datetime if not provided
current_datetime=${datetime:-$(date +"%Y-%m-%d_%H:%M:%S")}

# Set the prefix for current_datetime based on whether ablation_value is provided
if [[ -n "$ablation_value" ]]; then
    current_datetime="${ablation_value}-${current_datetime}"
else
    current_datetime="all_ablation_values-${current_datetime}"
fi

# Set the prefix for current_datetime based on whether model_name is provided
if [[ -n "$model_name" ]]; then
    current_datetime="${model_name}-${current_datetime}"
else
    current_datetime="all_models-${current_datetime}"
fi

# Define the model_name argument if provided
model_name_arg=""
if [[ -n "$model_name" ]]; then
    model_name_arg="--model_name $model_name"
fi

python scripts/build_probing_prompts.py --datetime $current_datetime $model_name_arg --device $device
python scripts/entropy_neurons_scripts/compute_and_save_neurons_measures.py --datetime $current_datetime $model_name_arg --device $device
python scripts/entropy_neurons_scripts/entropy_neurons_figures.py --last_layer_only --plot_norm_distro --datetime $current_datetime --plot_logitvar_and_rho $model_name_arg
python scripts/entropy_neurons_scripts/entropy_neurons_ablation.py $model_name_arg --datetime $current_datetime --neurons_ablation --ablation_value $ablation_value --device $device
python scripts/entropy_neurons_scripts/compute_ablation_scores.py --datetime $current_datetime $model_name_arg --ablation_value $ablation_value --device $device
python scripts/entropy_neurons_scripts/post_ablation_figures.py --datetime $current_datetime $model_name_arg --ablation_value $ablation_value --device $device