PATH = "."
MODEL_NAMES = ["gpt2-small", "Mistral-7B-v0.1", "Phi-1_5", "EleutherAI_pythia-1.4b", "Meta-Llama-3-8B"]
SPLIT_REGEXP = r'"|\.|\n|\\n|,|\(|\[|\]|\<'
NB_RANDOM_SAMPLES_FOR_ABLATION = 100
ABLATION_ALPHA = 3  # mean -/+ alpha*std ablation
MAX_GEN_TOKENS = 10
NB_COUNTER_PARAMETRIC_KNOWLEDGE = 3