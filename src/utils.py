import argparse
import os
import psutil

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_ram(device):
    mem = psutil.virtual_memory()
    free = mem.available / 1024**3
    total = mem.total / 1024**3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f"RAM: {total - free:.2f}/{total:.2f}GB\t RAM:[" + (total_cubes - free_cubes) * "▮" + free_cubes * "▯" + "]"


def get_vram(device):
    if device == "cuda":
        free = torch.cuda.mem_get_info()[0] / 1024**3
        total = torch.cuda.mem_get_info()[1] / 1024**3
    else:
        return f"No VRAM in device {device}"
    
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return (
        f"VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[" + (total_cubes - free_cubes) * "▮" + free_cubes * "▯" + "]"
    )

def statement(query: str, obj: str) -> str:
    """returns a statement about a triplet knowledge

    Args:
        query (str): query containing a subject and relation
        obj (str): object of the knowledge triplet (subject, relation, object)

    Returns:
        str: a statement of the subject, relation, and object
    """
    return query + " " + obj + "."

def parse_args():
    parser = argparse.ArgumentParser(description="Identifying entropy neurons")
    parser.add_argument("--model_name", type=str, help="Model name to build the knowledge datasets.")
    parser.add_argument("--device", type=str, help="Device on which to run.")
    parser.add_argument("--datetime", type=str, required=True)
    parser.add_argument(
        "--nb_counter_parametric_knowledge",
        "-nb_cpk",
        type=int,
        help="Number of counter-parametric-knowledge triplets per triplet knowledge.",
    )

    return parser.parse_args()


def setup_directories(permanent_path, model_name):
    directories = [
        f"{permanent_path}",
        f"{permanent_path}/entropy_neurons_figures/",
        f"{permanent_path}/entropy_neurons_figures/{model_name}",
        f"{permanent_path}/entropy_neurons_measures/",
        f"{permanent_path}/entropy_neurons_measures/{model_name}",
        f"{permanent_path}/ablation_experiments",
        f"{permanent_path}/ablation_experiments/{model_name}",
        f"{permanent_path}/latex_tables/",
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)


def generate_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_txt: str,
    device: str,
    max_new_tokens: int,
    return_proba_distro_first_token: bool = False,
) -> tuple[torch.Tensor, str]:
    """
    Generates output from the model given an input text and a tokenizer.

    Args:
        model (AutoModelForCausalLM): The loaded LLM.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        input_txt (str): The input text for generation.
        device (str): The device to run inference on.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        tuple: A tuple containing the logits distribution of the first token and the output text.
    """
    inputs = tokenizer(input_txt, return_tensors="pt").to(device)
    with torch.no_grad():
        if return_proba_distro_first_token:
            # Get the logits for the first token
            logits = model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
            first_token_proba_distro = torch.softmax(logits[0, -1, :], dim=-1).tolist()

        # Generate the output IDs
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
        )

    output_txt = tokenizer.decode(output_ids[0][inputs.input_ids[0].shape[0] :], skip_special_tokens=True)

    if return_proba_distro_first_token:
        return first_token_proba_distro, output_ids, output_txt
    else:
        return output_ids, output_txt


def format_probing_prompt(query: str, context_object: str):
    return f"{query} {context_object}. {query}"

def identify_knowledge_source(
    parametric_knowledge_object: str, counter_knowledge_object: str, output_object: str
) -> str:
    parametric_knowledge_object = parametric_knowledge_object.lower().strip()
    output_object = output_object.lower().strip()
    counter_knowledge_object = counter_knowledge_object.lower().strip()

    return (
        "CK"
        if " ".join(output_object.split(" ")[: len(counter_knowledge_object.split(" "))]) == counter_knowledge_object
        else (
            "PK"
            if " ".join(output_object.split(" ")[: len(parametric_knowledge_object.split(" "))])
            == parametric_knowledge_object
            else "ND"
        )
    )