from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd


def build_parametric_knowledge(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_gen_tokens: int,
) -> dict:
    """this function builds the parametric knowledge

    Args:
        prompt: a prompt in natural language format asking for the object given the subject and relation
        tokenizer: the loaded tokenizer
        model: the loaded LLM
        device: the device where to run the inference on the LLM
        max_gen_tokens: the maximum number of tokens to generate

    Returns:
        A dictionary containing the following information:
            - model_output: the model output sequence
            - log_probability_parametric_object_first_token: the log-probabilites of the first token
            - model_output_tokens_ids: the list of IDs of the generated tokens given the prompt
    """

    # print(f"prompt: {prompt}")
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids.input_ids,
            do_sample=False,  # greedy decoding strategy
            max_new_tokens=max_gen_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            attention_mask=prompt_ids.attention_mask,
        )
    model_output_tokens_ids = [token_id for token_id in output_ids.sequences[0][prompt_ids.input_ids.shape[1] :]]
    model_output = tokenizer.decode(model_output_tokens_ids, skip_special_tokens=True)
    log_probas_first_generated_token = torch.log(torch.softmax(output_ids.scores[0], dim=-1))[0].tolist()
    return {
        "model_output": model_output,
        "log_probability_parametric_object_first_token": log_probas_first_generated_token,
        "model_output_tokens_ids": model_output_tokens_ids,
    }


def is_parametric_object_not_in_the_prompt(
    parametric_object: str, statement_object: str, rel_lemma: str, relations_with_generated_description: pd.DataFrame
) -> bool:
    """this function checks whether the parametric_object is included in the
    one-shot examples that were used to guide the LLM during parametric knowledge
    building as it would mean that it's biased by the prompt.

    Args:
        parametric_object (str): the generated parametric object
        relations_with_generated_description (pd.DataFrame): the dataframe containing all the relations
        along with the objects and subjects that were used in the one-shot example in the prompt
        to guide the LLMs towards generating coherent objects (the used columns are `generation_object_1`
        and `generation_subject_1` which respectively represent the object and the subject used in the
        one-shot examples.)

    Returns:
        bool
    """

    if parametric_object.lower() == statement_object.lower():
        return True

    return (
        parametric_object.lower()
        not in relations_with_generated_description[relations_with_generated_description.relation == rel_lemma]
        .generation_object_1.str.lower()
        .tolist()
        and parametric_object.lower()
        not in relations_with_generated_description[relations_with_generated_description.relation == rel_lemma]
        .generation_subject_1.str.lower()
        .tolist()
    )
