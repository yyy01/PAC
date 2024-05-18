import os
import argparse
import jsonlines
from src.eda import *
from src.prob import *
from copy import deepcopy
from sklearn.metrics import roc_auc_score

import numpy as np

def load_dataset(dataset_path:str) :
    json_list = []
    with jsonlines.open(dataset_path, 'r') as reader:
        for obj in reader : json_list.append(obj)
    return json_list

def output_dataset(output_dir:str, json_list:list) :
    with jsonlines.open(output_dir, 'w') as writer:
        for obj in json_list : writer.write(obj)

def calculate_Polarized_Distance(prob_list:list, ratio_local = 0.3, ratio_far = 0.05) :
    local_region_length = max(int(len(prob_list)*ratio_local), 1)
    far_region_length = max(int(len(prob_list)*ratio_far), 1)
    local_region = np.sort(prob_list)[:local_region_length]
    far_region = np.sort(prob_list)[::-1][:far_region_length]
    return np.mean(far_region)-np.mean(local_region)

def calculate_PAC(prompt_list, api_key = None, model_engine = None, model_path = None, 
                  N = 5, num_threads = 1) :
    new_prompt_list = []
    print('Augmenting samples...')
    for prompt in prompt_list :
        newprompts = eda(prompt, alpha = 0.3, num_aug = N)
        new_prompt_list.extend(deepcopy(newprompts))
    
    print('Augmented samples:', len(new_prompt_list))
    print('Calculating probabilities for raw samples...')
    all_probs = calculate_probs(prompt_list, api_key, model_engine, model_path, num_threads)
    print('Calculating probabilities for augmented samples...')
    new_all_probs = calculate_probs(new_prompt_list, api_key, model_engine, model_path, num_threads)
    print('Calculating PAC...')

    pds = [calculate_Polarized_Distance(prob_list) for prob_list in all_probs]
    new_pds = [calculate_Polarized_Distance(prob_list) for prob_list in new_all_probs]
    calibrated_pds = [np.mean(new_pds[i:i+N]) for i in range(0, len(new_pds), N)]
    return np.array(pds)-np.array(calibrated_pds)

def calculate_auc(label_list:list, PAC_list:list) :
    return roc_auc_score(label_list, [-_ for _ in PAC_list])

def main(data_path = None, prompt = None, api_key = None, model_engine = None, model_path = None, num_threads = 1) :
    if data_path is not None :
        json_list = load_dataset(data_path)
        prompt_list = [obj['snippet'] for obj in json_list]
        label_list = [obj['label'] for obj in json_list if 'label' in obj]
        PAC_list = list(calculate_PAC(prompt_list, api_key = api_key, model_engine = model_engine, model_path = model_path, 
                                      num_threads = num_threads))
        for id, obj in enumerate(json_list) : obj['PAC'] = PAC_list[id]
        output_dir = './output'
        if not os.path.exists(output_dir) : os.makedirs(output_dir)
        _model_engine = model_engine if model_engine is not None else ''
        _model_path = model_path.split('/')[-1] if model_path is not None else ''
        output_path = os.path.join(output_dir, 'PAC_'+(_model_engine if _model_engine != '' else _model_path)+'.jsonl')
        output_dataset(output_path, json_list)
        print('The PAC result is saved at:', output_path)
        if 'label' in json_list[0] :
            auc = calculate_auc(label_list, PAC_list)
            print('AUC: {:.3f}'.format(auc))
    elif prompt is not None :
        prompt_list = [prompt]
        PAC_list = list(calculate_PAC(prompt_list, api_key = api_key, model_engine = model_engine, model_path = model_path,
                                      num_threads = num_threads))
        print('PAC: {:.3f}'.format(PAC_list[0]))
    else :
        raise ValueError('Either data_path or prompt should be provided')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", required=False, type=str, nargs = '?', default=None, help="dataset path")
    parser.add_argument("--snippet", required=False, type=str, nargs = '?', default=None, help="input sentence")
    parser.add_argument("--api_key", required=False, type=str, nargs = '?', default=None, help="OpenAI API key")
    parser.add_argument("--model_engine", required=False, type=str, nargs = '?', default=None, help="OpenAI model engine")
    parser.add_argument("--model_path", required=False, type=str, nargs = '?', default=None, help="model path")
    parser.add_argument("--num_threads", required=False, type=int, nargs = '?', default=1, help="number of threads")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    prompt = args.snippet
    api_key = args.api_key
    model_engine = args.model_engine
    model_path = args.model_path
    num_threads = args.num_threads

    main(dataset_path, prompt, api_key, model_engine, model_path, num_threads)