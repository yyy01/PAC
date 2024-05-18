import torch
from tqdm import tqdm
from src.prob_gpts_batch import *
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_probs_gpt(prompt_list, api_key, model_engine, num_threads=1):
    chatbot = ChatbotWrapper(api_key, model_engine)
    all_probs = chatbot.ask_batch(prompt_list, num_threads)
    
    return all_probs

def load_model(model_path) :
    model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True, device_map='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def calculate_probs_others(prompt_list, model_path):
    model, tokenizer = load_model(model_path)
    all_probs = []
    for prompt in tqdm(prompt_list):
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        logits = outputs[1]
        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)

        probs = []
        input_ids_processed = input_ids[0][1:]
        for i, token_id in enumerate(input_ids_processed):
            probability = probabilities[0, i, token_id].item()
            probs.append(probability)
        all_probs.append(probs)
    
    return all_probs

def calculate_probs(prompt_list, api_key = None, model_engine = None, model_path = None, num_threads=1):
    if model_engine is not None:
        return calculate_probs_gpt(prompt_list, api_key, model_engine, num_threads)
    elif model_path is not None:
        return calculate_probs_others(prompt_list, model_path)
    else:
        raise ValueError("model_engine or model_path should be provided")