import pandas as pd
from pathlib import Path

from src.utils import parse_args, setup_directories
from src.data import (
    remove_object_subject_overlap,
    generate_parametric_knowledge_dataset,
    generate_counter_parametric_knowledge_dataset,
    remove_entity_overlap_between_relation_groups,
)
from src.parametric_knowledge import is_parametric_object_not_in_the_prompt
from src.model import load_model_and_tokenizer
from src.prompt_model import prompt_model, build_prompt_for_knowledge_probing
from src.entropy_neurons.constants import PATH, SPLIT_REGEXP, MAX_GEN_TOKENS, MODEL_NAMES, NB_COUNTER_PARAMETRIC_KNOWLEDGE

if __name__ == "__main__":
    SUBJ_IN_OBJ_OVERLAP_STRING_SIMILARITY_RATIO = 0.8

    #############################################
    #### 0. parse the command line arguments ####
    #############################################
    args = parse_args()
    model_name = args.model_name
    device = args.device if args.device is not None else "cuda"
    nb_counter_parametric_knowledge = args.nb_counter_parametric_knowledge if args.nb_counter_parametric_knowledge is not None else NB_COUNTER_PARAMETRIC_KNOWLEDGE
    datetime = args.datetime

    model_names = MODEL_NAMES
    if model_name is not None:
        model_names = [model_name]

    run_path = Path(PATH) / datetime

    for model_name in model_names:

        # setup the environment
        setup_directories(run_path, model_name)

        #########################################
        #### 1. load the model and tokenizer ####
        #########################################
        model, tokenizer = load_model_and_tokenizer(model_name, device)

        #### 2. load the reformatted ParaRel
        raw_pararel_dataset_with_statements = pd.read_csv(
            f"input_data/knowledge-probing-data/raw_pararel_with_correctly_reformatted_statements.csv"
        )
        #### Load entropy neurons indexes ####
        relations_with_gen_desc = pd.read_excel(f"input_data/knowledge-probing-data/relation_groups_by_row_with_rel_gen_desc.xlsx")
        #######################################################################
        #### 3. build the parametric knowledge dataset for the current LLM ####
        #######################################################################
        parametric_knowledge_dataset = generate_parametric_knowledge_dataset(
            data=raw_pararel_dataset_with_statements,
            model=model,
            tokenizer=tokenizer,
            relations_with_gen_desc=relations_with_gen_desc,
            device=device,
            max_gen_tokens=MAX_GEN_TOKENS,
            split_regexp=SPLIT_REGEXP,
            model_name=model_name,
        )

        parametric_knowledge_dataset_without_subj_in_obj_overlap = remove_object_subject_overlap(
            parametric_knowledge_dataset, string_similarity_ratio=SUBJ_IN_OBJ_OVERLAP_STRING_SIMILARITY_RATIO
        )

        are_parametric_objects_not_in_the_prompt = parametric_knowledge_dataset_without_subj_in_obj_overlap.apply(
            lambda elt: is_parametric_object_not_in_the_prompt(
                parametric_object=elt.parametric_object,
                rel_lemma=elt.rel_lemma,
                statement_object=elt.statement_object,
                relations_with_generated_description=relations_with_gen_desc,
            ),
            axis=1,
        )

        parametric_knowledge_dataset = parametric_knowledge_dataset_without_subj_in_obj_overlap[
            are_parametric_objects_not_in_the_prompt
        ]

        parametric_knowledge_dataset = remove_entity_overlap_between_relation_groups(parametric_knowledge_dataset)

        ###############################################################################
        #### 4. build the counter parametric knowledge dataset for the current LLM ####
        ###############################################################################
        counter_parametric_knowledge_dataset = generate_counter_parametric_knowledge_dataset(
            parametric_knowledge_dataset, nb_counter_parametric_knowledge
        )

        counter_parametric_knowledge_dataset["knowledge_probing_prompt"] = counter_parametric_knowledge_dataset.apply(
            lambda counter_knowledge: build_prompt_for_knowledge_probing(
                input_query=counter_knowledge.statement_query,
                counter_knowledge_object=counter_knowledge.counter_knowledge_object,
            ),
            axis=1,
        ).tolist()

        #####################################################################################################################
        #### 5. prompt the current LLM and store its activations as features and identify the knowledge source (targets) ####
        #####################################################################################################################
        prompting_results = prompt_model(
            counter_parametric_knowledge_dataset=counter_parametric_knowledge_dataset,
            device=device,
            tokenizer=tokenizer,
            model_name=model_name,
            model=model,
            max_gen_tokens=MAX_GEN_TOKENS,
            split_regexp=SPLIT_REGEXP
        )

        prompting_results.to_csv(
            Path(run_path) / f"ablation_experiments/{model_name}/probing_prompts.csv"
        )