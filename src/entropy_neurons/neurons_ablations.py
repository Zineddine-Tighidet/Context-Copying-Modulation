from tqdm import tqdm
import torch
import transformer_lens
from transformer_lens import HookedTransformer
import numpy as np
from enum import Enum
import os
import re
import pandas as pd
from typing import Optional

from src.model import MODEL_PATHS
from src.prompt_model import identify_knowledge_source
from src.utils import get_ram, get_vram


class AblationType(Enum):
    MEAN_ABLATION = "mean_ablation"
    MEAN_MINUS_SIGMA_ABLATION = "mean_minus_sigma_ablation"
    MEAN_PLUS_SIGMA_ABLATION = "mean_plus_sigma_ablation"
    MEDIAN_ABLATION = "median_ablation"
    MODE_ABLATION = "mode_ablation"

    def render_caption_text(self):
        render_map = {
            AblationType.MEAN_ABLATION: r"Activation ablation with $\mu$",
            AblationType.MEAN_MINUS_SIGMA_ABLATION: r"Activation ablation with $\mu_{n_i} - 3\sigma_{n_i}$",
            AblationType.MEAN_PLUS_SIGMA_ABLATION: r"Activation ablation with $\mu_{n_i} + 3\sigma_{n_i}$",
            AblationType.MEDIAN_ABLATION: "Activation ablation with median",
            AblationType.MODE_ABLATION: "Activation ablation with mode",
        }
        return render_map[self]

    def render_table_label(self):
        render_map = {
            AblationType.MEAN_ABLATION: r"$\mu$",
            AblationType.MEAN_MINUS_SIGMA_ABLATION: r"max($\mu_{n_i} - 3\sigma_{n_i}$, min_{n_i})",
            AblationType.MEAN_PLUS_SIGMA_ABLATION: r"min($\mu_{n_i} + 3\sigma_{n_i}$, max_{n_i})",
            AblationType.MEDIAN_ABLATION: "Median",
            AblationType.MODE_ABLATION: "Mode",
        }
        return render_map[self]


def generate_without_ablation(
    model: HookedTransformer,
    probing_data: pd.DataFrame,
    split_regexp: str,
    max_new_tokens: int = 10,
    batch_size: int = 64,
):
    output_texts = []
    entropies = []
    knowledge_sources = []
    output_objects = []
    all_ck_log_prob = []
    all_pk_log_prob = []
    all_nd_log_prob = []
    all_nd_token_id = []

    for i in tqdm(range(0, len(probing_data), batch_size), desc="Processing batches without ablation"):
        batch_input_texts = probing_data.iloc[i : i + batch_size].knowledge_probing_prompt.tolist()
        batch_counter_knowledge_objects = probing_data.iloc[i : i + batch_size].counter_knowledge_object.tolist()
        batch_parametric_knowledge_objects = probing_data.iloc[i : i + batch_size].parametric_knowledge_object.tolist()
        batch_parametric_knowledge_first_token_ids = probing_data.iloc[
            i : i + batch_size
        ].parametric_knowledge_first_token_id.tolist()
        batch_counter_knowledge_object_first_token_ids = probing_data.iloc[
            i : i + batch_size
        ].counter_knowledge_object_first_token_id.tolist()

        input_encodings = model.to_tokens(batch_input_texts, padding_side="left")

        generated_batch = model.generate(
            input_encodings, max_new_tokens=max_new_tokens, do_sample=False, padding_side="left", verbose=False
        )

        with torch.no_grad():
            # Get the logits for the entire batch
            logits = model(input_encodings).detach()

        # Compute log probabilities using log_softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Compute entropy for each token in the batch
        entropies_batch = -(torch.exp(log_probs) * log_probs).sum(dim=-1)

        for j in range(len(batch_input_texts)):
            input_ids = input_encodings[j]
            output_ids = generated_batch[j].tolist()
            counter_knowledge_object = batch_counter_knowledge_objects[j]
            parametric_knowledge_object = batch_parametric_knowledge_objects[j]
            parametric_knowledge_first_token_id = batch_parametric_knowledge_first_token_ids[j]
            counter_knowledge_object_first_token_id = batch_counter_knowledge_object_first_token_ids[j]

            # Extract the entropies for the generated part
            generated_entropies = entropies_batch[j, -1]

            # Compute the average entropy for the generated tokens
            output_entropy = generated_entropies.item()
            entropies.append(output_entropy)

            output_txt = model.tokenizer.decode(output_ids[len(input_ids) :], skip_special_tokens=True)
            output_texts.append(output_txt)

            # Extract the log probabilities for the first token
            last_token_index = (input_ids != model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][-1]
            first_token_log_prob = log_probs[j, last_token_index]

            # identify the knowledge source here
            output_object = re.split(split_regexp, output_txt)[0].lower().strip()
            output_objects.append(output_object)

            knowledge_source = identify_knowledge_source(
                counter_knowledge_object=counter_knowledge_object,
                parametric_knowledge_object=parametric_knowledge_object,
                output_object=output_object,
            )
            knowledge_sources.append(knowledge_source)

            # identify P(CK), P(PK), and P(ND) here
            ck_log_prob, pk_log_prob, nd_log_prob, nd_token_id = identify_ck_pk_nd_log_probabilities(
                ck_first_token_id=counter_knowledge_object_first_token_id,
                pk_first_token_id=parametric_knowledge_first_token_id,
                knowledge_source=knowledge_source,
                log_probs_distro=first_token_log_prob.detach().to("cpu").tolist(),
                model=model,
            )

            all_ck_log_prob.append(ck_log_prob)
            all_pk_log_prob.append(pk_log_prob)
            all_nd_log_prob.append(nd_log_prob)
            all_nd_token_id.append(int(nd_token_id))

    return {
        "output_texts": output_texts,
        "entropies": entropies,
        "output_objects": output_objects,
        "knowledge_sources": knowledge_sources,
        "ck_log_prob": all_ck_log_prob,
        "pk_log_prob": all_pk_log_prob,
        "nd_log_prob": all_nd_log_prob,
        "nd_token_id": nd_token_id,
    }


def generate_with_activation_ablation(
    model: HookedTransformer,
    probing_data: pd.DataFrame,
    split_regexp: str,
    neurons_indices: list[int],
    mean_activations: np.ndarray,
    std_activations: np.ndarray,
    mode_activations: np.ndarray,
    median_activations: np.ndarray,
    min_activations: np.ndarray,
    max_activations: np.ndarray,
    ablation_type: str = AblationType.MEAN_ABLATION.value,
    alpha: float = 1.0,
    batch_size: int = 64,
    max_new_tokens: int = 10,
):
    """
    Generate text with ablation applied to specified neurons and/or force specified attention heads
    to attend only to the BOS token.

    Args:
        model (HookedTransformer): The transformer model to use for generation.
        pre_ablation_model_outputs (pd.DataFrame): DataFrame with pre-ablation outputs.
        probing_data (pd.DataFrame): DataFrame with probing prompts and knowledge objects.
        split_regexp (str): Regular expression to split generated text for object extraction.
        neurons_indices (list, optional): Indices of neurons to ablate.
        mean_activations (np.ndarray, optional): Mean activations for neuron ablation.
        std_activations (np.ndarray, optional): Standard deviations for neuron ablation.
        mode_activations (np.ndarray, optional): Mode activations for neuron ablation.
        median_activations (np.ndarray, optional): Median activations for neuron ablation.
        min_activations (np.ndarray, optional): Minimum activations for neuron ablation.
        max_activations (np.ndarray, optional): Maximum activations for neuron ablation.
        ablation_type (str): Type of neuron ablation to apply.
        alpha (float): Scaling factor for standard deviation in ablation.
        batch_size (int): Number of samples to process per batch.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        dict: Dictionary containing output texts, entropies, knowledge sources, output objects,
              and log probabilities.
    """

    # Neuron ablation hook
    def ablate_entropy_neurons(outputs, hook):
        activations = outputs.detach().clone()
        if neurons_indices is None:
            return activations
        neurons_indices_tensor = torch.LongTensor(neurons_indices).to(activations.device)

        if ablation_type == AblationType.MEAN_ABLATION.value:
            mean_activations_tensor = torch.Tensor(mean_activations).to(activations.device)
            activations[:, :, neurons_indices_tensor] = mean_activations_tensor[neurons_indices_tensor]
        elif ablation_type == AblationType.MEAN_MINUS_SIGMA_ABLATION.value:
            mean_activations_tensor = torch.Tensor(mean_activations).to(activations.device)
            std_activations_tensor = torch.Tensor(std_activations).to(activations.device)
            min_activations_tensor = torch.Tensor(min_activations).to(activations.device)
            activations[:, :, neurons_indices_tensor] = torch.max(
                min_activations_tensor[neurons_indices_tensor],
                mean_activations_tensor[neurons_indices_tensor]
                - alpha * std_activations_tensor[neurons_indices_tensor],
            )
        elif ablation_type == AblationType.MEAN_PLUS_SIGMA_ABLATION.value:
            mean_activations_tensor = torch.Tensor(mean_activations).to(activations.device)
            std_activations_tensor = torch.Tensor(std_activations).to(activations.device)
            max_activations_tensor = torch.Tensor(max_activations).to(activations.device)
            activations[:, :, neurons_indices_tensor] = torch.min(
                max_activations_tensor[neurons_indices_tensor],
                mean_activations_tensor[neurons_indices_tensor]
                + alpha * std_activations_tensor[neurons_indices_tensor],
            )
        elif ablation_type == AblationType.MODE_ABLATION.value:
            mode_activations_tensor = torch.Tensor(mode_activations).to(activations.device)
            activations[:, :, neurons_indices_tensor] = mode_activations_tensor[neurons_indices_tensor]
        elif ablation_type == AblationType.MEDIAN_ABLATION.value:
            median_activations_tensor = torch.Tensor(median_activations).to(activations.device)
            activations[:, :, neurons_indices_tensor] = median_activations_tensor[neurons_indices_tensor]
        return activations

    # Prepare hooks
    hooks = []
    if (neurons_indices is not None) and (
        mean_activations is not None
    ):  # Apply neuron ablation if indices are provided
        hooks.append((f"blocks.{model.cfg.n_layers-1}.mlp.hook_post", ablate_entropy_neurons))

    # Text generation
    output_texts = []
    entropies = []
    knowledge_sources = []
    output_objects = []
    all_ck_log_prob = []
    all_pk_log_prob = []
    all_nd_log_prob = []

    for i in tqdm(
        range(0, len(probing_data), batch_size), desc=f"Processing batches with {ablation_type} ablation"
    ):
        batch_input_texts = probing_data.iloc[i : i + batch_size].knowledge_probing_prompt.tolist()
        batch_counter_knowledge_objects = probing_data.iloc[i : i + batch_size].counter_knowledge_object.tolist()
        batch_parametric_knowledge_objects = probing_data.iloc[i : i + batch_size].parametric_knowledge_object.tolist()
        batch_parametric_knowledge_first_token_ids = probing_data.iloc[
            i : i + batch_size
        ].parametric_knowledge_first_token_id.tolist()
        batch_counter_knowledge_object_first_token_ids = probing_data.iloc[
            i : i + batch_size
        ].counter_knowledge_object_first_token_id.tolist()

        input_encodings = model.to_tokens(batch_input_texts, padding_side="left")

        with model.hooks(fwd_hooks=hooks):
            generated_batch = model.generate(
                input_encodings, max_new_tokens=max_new_tokens, do_sample=False, padding_side="left", verbose=False
            )

        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                logits = model(generated_batch).detach()

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropies_batch = -(torch.exp(log_probs) * log_probs).sum(dim=-1)

        for j in range(len(batch_input_texts)):
            input_ids = input_encodings[j]
            output_ids = generated_batch[j].tolist()
            counter_knowledge_object = batch_counter_knowledge_objects[j]
            parametric_knowledge_object = batch_parametric_knowledge_objects[j]
            parametric_knowledge_first_token_id = batch_parametric_knowledge_first_token_ids[j]
            counter_knowledge_object_first_token_id = batch_counter_knowledge_object_first_token_ids[j]

            generated_entropies = entropies_batch[j, len(input_ids)-1]
            output_entropy = generated_entropies.item()
            entropies.append(output_entropy)

            output_txt = model.tokenizer.decode(output_ids[len(input_ids) :], skip_special_tokens=True)
            output_texts.append(output_txt)

            # Extract the log probabilities for the first token
            # last_token_index = (input_ids != model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][-1]
            first_token_log_prob = log_probs[j, len(input_ids)-1]

            output_object = re.split(split_regexp, output_txt)[0].lower().strip()
            output_objects.append(output_object)

            knowledge_source = identify_knowledge_source(
                counter_knowledge_object=counter_knowledge_object,
                parametric_knowledge_object=parametric_knowledge_object,
                output_object=output_object,
            )
            knowledge_sources.append(knowledge_source)

            ck_log_prob, pk_log_prob, nd_log_prob, _ = identify_ck_pk_nd_log_probabilities(
                ck_first_token_id=counter_knowledge_object_first_token_id,
                # nd_token_id=nd_token_id,
                pk_first_token_id=parametric_knowledge_first_token_id,
                knowledge_source=knowledge_source,
                log_probs_distro=first_token_log_prob.detach().to("cpu").tolist(),
                model=model,
            )

            all_ck_log_prob.append(ck_log_prob)
            all_pk_log_prob.append(pk_log_prob)
            all_nd_log_prob.append(nd_log_prob)

    return {
        "output_texts": output_texts,
        "entropies": entropies,
        "knowledge_sources": knowledge_sources,
        "output_objects": output_objects,
        "ck_log_prob": all_ck_log_prob,
        "pk_log_prob": all_pk_log_prob,
        "nd_log_prob": all_nd_log_prob,
    }


def get_nd_token_id(log_probs, ck_or_pk_token_id):
    # Get the indices of the probabilities sorted in descending order
    sorted_indices = np.argsort(log_probs)[::-1]

    # Check if ck_or_pk_token_id is the second most probable
    if sorted_indices[1] == ck_or_pk_token_id:
        # Return the third most probable index
        return sorted_indices[2]
    else:
        # Return the second most probable index
        return sorted_indices[1]


def identify_ck_pk_nd_log_probabilities(
    log_probs_distro: list,
    knowledge_source: str,
    pk_first_token_id: int,
    ck_first_token_id: int,
    model: HookedTransformer,
    nd_token_id: Optional[int] = None,
) -> dict:
    """This function returns a dict with the probabilites of CK, PK, and ND.

    - log_probs_distro (torch.Tensor): the log probability distribution over the vocabulary
    - knowledge_source (str): the knowledge source (CK, PK, or ND)

    returns a tuple with CK, PK, and ND log probability.
    """
    assert len(log_probs_distro) == model.cfg.d_vocab

    if knowledge_source == "CK":
        # context
        conflicting_counter_knowledge_object_log_proba = max(log_probs_distro)

        # parametric
        parametric_knowledge_first_token = model.tokenizer.decode(pk_first_token_id)
        parametric_knowledge_for_probing_prompt_first_token_id = model.tokenizer.encode(
            f" {parametric_knowledge_first_token}"
        )[0]

        conflicting_parametric_knowledge_object_log_proba = log_probs_distro[
            parametric_knowledge_for_probing_prompt_first_token_id
        ]

        # undefined
        if nd_token_id is None:
            undefined_knowledge_for_probing_prompt_first_token_id = get_nd_token_id(
                log_probs=log_probs_distro, ck_or_pk_token_id=parametric_knowledge_for_probing_prompt_first_token_id
            )
        else:
            undefined_knowledge_for_probing_prompt_first_token_id = nd_token_id
        conflicting_undefined_knowledge_object_log_proba = log_probs_distro[
            undefined_knowledge_for_probing_prompt_first_token_id
        ]
    elif knowledge_source == "PK":
        # parametric
        conflicting_parametric_knowledge_object_log_proba = max(log_probs_distro)

        # context
        counter_knowledge_first_token = model.tokenizer.decode(ck_first_token_id)
        counter_knowledge_for_probing_prompt_first_token_id = model.tokenizer.encode(
            f" {counter_knowledge_first_token}"
        )[0]

        conflicting_counter_knowledge_object_log_proba = log_probs_distro[
            counter_knowledge_for_probing_prompt_first_token_id
        ]

        # undefined
        if nd_token_id is None:
            undefined_knowledge_for_probing_prompt_first_token_id = get_nd_token_id(
                log_probs=log_probs_distro, ck_or_pk_token_id=counter_knowledge_for_probing_prompt_first_token_id
            )
        else:
            undefined_knowledge_for_probing_prompt_first_token_id = nd_token_id

        conflicting_undefined_knowledge_object_log_proba = log_probs_distro[
            undefined_knowledge_for_probing_prompt_first_token_id
        ]
    else:  # ND
        # undefined
        conflicting_undefined_knowledge_object_log_proba = max(log_probs_distro)
        undefined_knowledge_for_probing_prompt_first_token_id = np.argmax(log_probs_distro)

        # parametric
        parametric_knowledge_first_token = model.tokenizer.decode(pk_first_token_id)
        parametric_knowledge_for_probing_prompt_first_token_id = model.tokenizer.encode(
            f" {parametric_knowledge_first_token}"
        )[0]

        conflicting_parametric_knowledge_object_log_proba = log_probs_distro[
            parametric_knowledge_for_probing_prompt_first_token_id
        ]

        # context
        counter_knowledge_first_token = model.tokenizer.decode(ck_first_token_id)
        counter_knowledge_for_probing_prompt_first_token_id = model.tokenizer.encode(
            f" {counter_knowledge_first_token}"
        )[0]

        conflicting_counter_knowledge_object_log_proba = log_probs_distro[
            counter_knowledge_for_probing_prompt_first_token_id
        ]

    return (
        conflicting_counter_knowledge_object_log_proba,
        conflicting_parametric_knowledge_object_log_proba,
        conflicting_undefined_knowledge_object_log_proba,
        undefined_knowledge_for_probing_prompt_first_token_id,
    )


def compute_mean_std_max_min_median_mlp_l1_activations(
    prompts: list,
    device: str = "cuda",
    model_name: str = "gpt2-small",
    initial_batch_size: int = 32,
    max_new_tokens: int = 10,
    activations_save_dir: str = "./activations",
) -> dict:
    """
    Computes the mean, standard deviation, min, max, mode, and median of each neuron in the first MLP layer
    across all prompts using transformer_lens. All activations for each neuron are saved to disk and then loaded
    to compute the final statistics.
    Args:
    prompts (list of str): A list of text prompts to feed into the model.
    model_name (str): Name of the transformer model to use (default: "gpt2-small").
    initial_batch_size (int): Initial batch size, which adjusts dynamically based on memory availability (default: 32).
    max_new_tokens (int): Maximum number of tokens per prompt (default: 10).
    activations_save_dir (str): Directory to save intermediate results (default: "./activations").

    Returns:
        dict: A dictionary containing the mean, standard deviation, min, max, mode, and median of the activations.
    """

    # Ensure the directory for saving activations exists
    os.makedirs(activations_save_dir, exist_ok=True)

    model_path = MODEL_PATHS[model_name]
    transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(model_path)
    model = HookedTransformer.from_pretrained(model_path, device=device)
    model.eval()

    d_mlp = model.cfg.d_mlp

    # Temporary storage for activations in memory (per batch)
    temp_activations = []

    print(get_vram(device=device))
    print(get_ram(device=device))

    def hook_fn(activation, hook):
        nonlocal temp_activations
        print(get_vram(device=device))
        print(get_ram(device=device))
        # activation shape: [batch_size, seq_len, d_mlp]
        neurons_activations_cpu = activation.cpu().numpy().astype(np.float16)
        neurons_activations_cpu = neurons_activations_cpu.reshape(-1, d_mlp)
        # neurons_activations_cpu new shape (batch_size*seq_len, d_mlp)
        temp_activations.append(neurons_activations_cpu)  # concat at the end to be faster

    # Process prompts in dynamically adjusted batches
    batch_size = initial_batch_size
    pbar = tqdm(total=len(prompts))
    i = 0
    nb_batches = 0
    while i < len(prompts):
        batch = prompts[i : i + batch_size]

        # Tokenize batch
        input_encodings = model.to_tokens(batch, padding_side="left")

        with model.hooks(fwd_hooks=[(f"blocks.{model.cfg.n_layers-1}.mlp.hook_post", hook_fn)]):
            with torch.no_grad():
                _ = model.generate(
                    input_encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    padding_side="left",
                    verbose=False,
                )
        i += batch_size
        pbar.update(batch_size)
        print(get_vram(device=device))
        print(get_ram(device=device))
        
        if device == "cuda":
            torch.cuda.empty_cache()

        nb_batches += 1

    pbar.close()

    # concat all batches from shape (batch_size*seq_len, d_mlp) into (batch_size*seq_len*nb_batches, d_mlp)
    temp_activations = np.concatenate(temp_activations, axis=0, dtype=np.float16)
    # (batch_size*seq_len*nb_batches, d_mlp) -> (d_mlp, batch_size*seq_len*nb_batches)
    temp_activations = temp_activations.T

    mean_activations = np.mean(temp_activations, axis=1)
    std_activations = np.std(temp_activations, axis=1)
    min_activations = np.min(temp_activations, axis=1)
    max_activations = np.max(temp_activations, axis=1)
    median_activations = np.median(temp_activations, axis=1)

    file_path = os.path.join(activations_save_dir, "last_layer_mlp_l1_activations.npy")
    with open(file_path, "wb") as fo:
        np.save(fo, temp_activations)

    results = {
        "mean_activation": mean_activations,
        "std_activation": std_activations,
        "min_activation": min_activations,
        "max_activation": max_activations,
        "median_activation": median_activations,
    }

    return results


def generate_random_neurons_idxs(
    entropy_neurons_idxs: list, min_value: int, max_value: int, previous_random_neurons_idxs: list
):
    length = len(entropy_neurons_idxs)
    if max_value - min_value + 1 < length:
        raise ValueError("The range of values is too small to generate a unique array of the given length.")

    entropy_neurons_idxs_set = set(entropy_neurons_idxs)

    while True:
        random_neurons_idxs = np.random.choice(range(min_value, max_value + 1), size=length, replace=False)
        random_neurons_idxs_set = set(random_neurons_idxs)

        # Check if the new array has no elements in common with any previous array
        if random_neurons_idxs_set != entropy_neurons_idxs_set:
            if previous_random_neurons_idxs is not None:
                is_different = True
                for prev_array in previous_random_neurons_idxs:
                    prev_array_set = set(prev_array)
                    if random_neurons_idxs_set == prev_array_set:
                        is_different = False
                        break
                if is_different:
                    return random_neurons_idxs
            else:
                return random_neurons_idxs
