import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
from tqdm import tqdm

from src.utils import generate_output, identify_knowledge_source

def build_prompt_for_knowledge_probing(input_query: str, counter_knowledge_object: str):
    return f"{input_query} {counter_knowledge_object}. {input_query}"

def prompt_model(
    counter_parametric_knowledge_dataset: pd.DataFrame,
    device: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    model: AutoModelForCausalLM,
    max_gen_tokens: str = 10,
    split_regexp: str = r'"|\.|\n|\\n|,|\(|\<s\>',
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """
    Prompts the LLM to determine the knowledge source and saves the activations of specific modules (MLP, MLP-L1, MHSA).

    Args:
        parametric_knowledge (pd.DataFrame): Dataframe containing the parametric knowledge dataset.
        counter_parametric_knowledge_dataset (pd.DataFrame): Dataframe containing the counter-parametric-knowledge dataset.
        device (str): Device where inference on the LLM will be run.
        tokenizer (AutoTokenizer): Tokenizer for the LLM.
        model_name (str): Name of the model (mainly for printing and path reference purposes).
        model (AutoModelForCausalLM): Loaded LLM.
        split_regexp (str): the regexp pattern to parse the output of the LLM after prompting

    Returns:
        prompting_results: A DataFrame containing various results and metrics of the probe as well as the activations
    """

    prompting_results = []

    # We copy to avoid modifying inplace
    counter_parametric_knowledge_dataset = counter_parametric_knowledge_dataset.copy()
    for _, counter_param_knowledge_record in tqdm(
        counter_parametric_knowledge_dataset.iterrows(),
        total=counter_parametric_knowledge_dataset.shape[0],
        desc=f"Probing {model_name} with prompts...",
    ):
        # Example values:
        # - statement: "Paris is the capital of France"
        # - statement_object: "France"
        # - counter_knowledge_object: "Italy"
        # - statement_query: "Paris is the capital of"
        knowledge_probing_prompt_context = f"{counter_param_knowledge_record.statement_query} {counter_param_knowledge_record.counter_knowledge_object}."

        knowledge_probing_prompt_query = counter_param_knowledge_record.statement_query

        counter_param_knowledge_record.knowledge_probing_prompt = (
            f"{knowledge_probing_prompt_context} {knowledge_probing_prompt_query}"
        )

        first_token_proba_distro, _, output_txt = generate_output(
            model=model,
            tokenizer=tokenizer,
            input_txt=counter_param_knowledge_record.knowledge_probing_prompt,
            device=device,
            max_new_tokens=max_gen_tokens,
            return_proba_distro_first_token=True,
        )

        output_object = re.split(split_regexp, output_txt)[0]

        knowledge_source = identify_knowledge_source(
            parametric_knowledge_object=counter_param_knowledge_record["parametric_knowledge_object"],
            counter_knowledge_object=counter_param_knowledge_record["counter_knowledge_object"],
            output_object=output_object,
        )

        if knowledge_source == "CK":
            # context
            conflicting_counter_knowledge_object_proba = max(first_token_proba_distro)

            # parametric
            parametric_knowledge_first_token = tokenizer.decode(
                counter_param_knowledge_record["parametric_knowledge_first_token_id"]
            )
            parametric_knowledge_for_probing_prompt_first_token_id = tokenizer.encode(
                f" {parametric_knowledge_first_token}"
            )[0]

            conflicting_parametric_knowledge_object_proba = first_token_proba_distro[
                parametric_knowledge_for_probing_prompt_first_token_id
            ]

            # undefined
            first_token_proba_distro.pop(parametric_knowledge_for_probing_prompt_first_token_id)  # remove PK object
            first_token_proba_distro.pop(0)  # remove CK object (first)

            conflicting_undefined_knowledge_object_proba = first_token_proba_distro[0]
        elif knowledge_source == "PK":
            # parametric
            conflicting_parametric_knowledge_object_proba = max(first_token_proba_distro)

            # context
            counter_knowledge_first_token = tokenizer.decode(
                counter_param_knowledge_record["counter_knowledge_object_first_token_id"]
            )
            counter_knowledge_for_probing_prompt_first_token_id = tokenizer.encode(f" {counter_knowledge_first_token}")[
                0
            ]

            conflicting_counter_knowledge_object_proba = first_token_proba_distro[
                counter_knowledge_for_probing_prompt_first_token_id
            ]

            # undefined
            first_token_proba_distro.pop(counter_knowledge_for_probing_prompt_first_token_id)  # remove CK object
            first_token_proba_distro.pop(0)  # remove PK object (first)

            conflicting_undefined_knowledge_object_proba = first_token_proba_distro[0]
        else:  # ND
            # undefined
            conflicting_undefined_knowledge_object_proba = max(first_token_proba_distro)

            # parametric
            parametric_knowledge_first_token = tokenizer.decode(
                counter_param_knowledge_record["parametric_knowledge_first_token_id"]
            )
            parametric_knowledge_for_probing_prompt_first_token_id = tokenizer.encode(
                f" {parametric_knowledge_first_token}"
            )[0]

            conflicting_parametric_knowledge_object_proba = first_token_proba_distro[
                parametric_knowledge_for_probing_prompt_first_token_id
            ]

            # context
            counter_knowledge_first_token = tokenizer.decode(
                counter_param_knowledge_record["counter_knowledge_object_first_token_id"]
            )
            counter_knowledge_for_probing_prompt_first_token_id = tokenizer.encode(f" {counter_knowledge_first_token}")[
                0
            ]

            conflicting_counter_knowledge_object_proba = first_token_proba_distro[
                counter_knowledge_for_probing_prompt_first_token_id
            ]

        prompting_results.append(
            {
                "knowledge_source": knowledge_source,
                "output_txt": output_txt,
                "output_object": output_object.lower().strip(),
                "counter_knowledge_object": counter_param_knowledge_record["counter_knowledge_object"].lower().strip(),
                "log_prob_counter_knowledge": counter_param_knowledge_record["log_prob_counter_object"],
                "conflicting_counter_knowledge_object_proba": conflicting_counter_knowledge_object_proba,
                "parametric_knowledge_object": counter_param_knowledge_record["parametric_knowledge_object"]
                .lower()
                .strip(),
                "parametric_knowledge_first_token_id": counter_param_knowledge_record[
                    "parametric_knowledge_first_token_id"
                ].item(),
                "counter_knowledge_object_first_token_id": counter_param_knowledge_record[
                    "counter_knowledge_object_first_token_id"
                ].item(),
                "log_prob_parametric_knowledge": counter_param_knowledge_record["log_prob_parametric_object"],
                "conflicting_parametric_knowledge_object_proba": conflicting_parametric_knowledge_object_proba,
                "conflicting_undefined_knowledge_object_proba": conflicting_undefined_knowledge_object_proba,
                "statement_query": counter_param_knowledge_record["statement_query"],
                "statement_subject": counter_param_knowledge_record["statement_subject"],
                "rel_lemma": counter_param_knowledge_record.rel_lemma,
                "relation_group_id": counter_param_knowledge_record.relation_group_id,
                "knowledge_probing_prompt": counter_param_knowledge_record.knowledge_probing_prompt,
                "parametric_output": counter_param_knowledge_record.parametric_output,
            }
        )

    prompting_results_df = pd.DataFrame(prompting_results)

    return prompting_results_df
