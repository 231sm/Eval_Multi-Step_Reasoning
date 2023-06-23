#!/usr/bin/env python3

# Copyright (c) EMNLP 2023 Submission

import json
import re
import argparse
import csv
from utils import (
    extract_if_correct, get_rationale, read_jsonl, read_json, approx_eq, approx_in, approx_overlap, tokenize,  
    extract_nums, find_nums, find_formula, extract_answer, test_answer)
import numpy as np
import pandas as pd

from constants import (
    DEFAULT_INPUT_ANN_PATH, 
    DEFAULT_INPUT_GEN_PATH, 
    DEFAULT_INPUT_RES_PATH, 
    DEFAULT_PROMPT_PATH, 
    DEFAULT_OUTPUT_PATH,
    DATASETS,
    STR_GEN_STOP, 
    DICT_STR_SPLIT_RATIONALE
)


def util_gsm8k(file_path, num_test):
    # file contains a list of dict with keys: question, answer, example_result, ground_truth, query, ans_*
    dataset_name = "gsm8k"
    data = read_jsonl(DEFAULT_INPUT_ANN_PATH + dataset_name + "/grade_school_math/data/test.jsonl")
    for i in range(len(data)):
        item = data[i]
        ans = extract_if_correct(item['answer'])
        rationale, _ = item['answer'].split('####')
        rationale = rationale.strip().split("\n")
        # rationale = item['answer'].strip().split(". ")
        item['rationale'] = rationale
        item['answer'] = ans
         
        ## processing the formula to get leaf nums and non_leaf_nums
        leaf_nums = []
        non_leaf_nums = []
        
        for step in rationale:
            if step.count("<<") == 0:
                # no formula in this step - treat the last num as non_leaf_num
                nums = find_nums(step)
                if not nums:
                    continue
                for num in nums[:-1]:
                    if num in leaf_nums or num in non_leaf_nums:
                        continue
                    # this num is a new leaf num
                    leaf_nums.append(num)
                non_leaf_nums.append(nums[-1])
            elif step.count("<<") == 1:
                formula = find_formula(step)
                left, right = formula.split("=")
                left_nums = find_nums(left)
                
                # when "X/Y" is only num on the left, split them
                if len(left_nums) == 1 and "/" in left_nums[0]:
                    left_nums = left_nums[0].split("/")
                
                for num in left_nums:
                    if num in leaf_nums or num in non_leaf_nums:
                        continue
                    # this num is a new leaf num
                    leaf_nums.append(num)
                non_leaf_nums.append(right)
            else:
                # assert no step with >1 formulas
                assert False

        ## ------
        item['leaf_nums'] = leaf_nums
        item['non_leaf_nums'] = non_leaf_nums
        data[i] = item

    # dictionary maps question to id
    question2id = dict()
    for i in range(len(data)):
        item = data[i]
        question2id[item['question']] = i
    dict_output_path = DEFAULT_INPUT_GEN_PATH + "gsm8k_question2id.json" 
    with open(dict_output_path, 'w') as outfile:
        json.dump(question2id, outfile)
        outfile.write('\n')
    print(f"Saved gsm8k question2id dict in {dict_output_path}")
    list_questionid = [] 
        
    if file_path.endswith(".jsonl"):
        temp = read_jsonl(file_path)
        # last element is the prompt
        result, prompt = temp[:-1], temp[-1]
    else:
        with open(file_path, "r") as f:
            temp = json.load(f)
            # last element is the prompt
            result, prompt = temp[:-1], temp[-1]
    if num_test == -1:
       num_test = len(result) 
    assert len(result) == num_test
    
    non_leaf_coverage_l, non_leaf_precision_l, non_leaf_F1_l = [], [], []
    accuracy_l = []
    
    bins_non_leaf_coverage = dict()
    bins_non_leaf_precision = dict()
    bins_non_leaf_F1 = dict()
    bins_ans_accuracy = dict()
    
    count_no_nonleaf = 0
    for index in range(len(result)):
        predicted_item = result[index]
        data_entry = data[question2id[predicted_item['question']]]
        assert data_entry['question'] == predicted_item['question']

        for key_ in predicted_item.keys():
            if key_.startswith('ans_'):
                # result[index][key_[4:]] = predicted_item[key_]
                predicted_rationale, raw_ans = get_rationale(predicted_item[key_]) 
                setting_name = key_[4:]
                # raw_ans = extract_answer(predicted_item[key_], dataset_name)
                break
            else:
                predicted_rationale = predicted_item['example_result'].split('\nA: ')[0]
                # raw_ans = predicted_item['example_result']
                raw_ans = str(extract_answer(predicted_item['example_result'], dataset_name))
                setting_name = "example_result" 
         
        # get rid of in-token ",", artifact for long numbers
        predicted_rationale_tokens = predicted_rationale.split()
        for i in range(len(predicted_rationale_tokens)):
            token = predicted_rationale_tokens[i]
            if "," in token[1:-1]:
                token = token[0] + token[1:-1].replace(",","") + token[-1]
            predicted_rationale_tokens[i] = token
        predicted_rationale = " ".join(predicted_rationale_tokens)
        
        predicted_steps = predicted_rationale.split(DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP])
        predicted_nonleaf_nums = set()
        for step in predicted_steps:
            if '=' in step:
                nums = find_nums(step)
                if nums:
                    predicted_nonleaf_nums.add(nums[-1])
        nums = find_nums(predicted_steps[-1])
        if nums:
            predicted_nonleaf_nums.add(nums[-1])
        
        non_leaf_nums = set(data_entry['non_leaf_nums'])
        
        # get recall & precision & f1
        if not non_leaf_nums:
            count_no_nonleaf += 1
            non_leaf_coverage, non_leaf_precision = 1, 1
        else:
            non_leaf_overap = approx_overlap(predicted_nonleaf_nums, non_leaf_nums)
            non_leaf_coverage = non_leaf_overap/len(non_leaf_nums)
            if not predicted_nonleaf_nums:
                non_leaf_precision = 0
            else:
                non_leaf_precision = non_leaf_overap/len(predicted_nonleaf_nums)
            
        # f1
        if non_leaf_precision * non_leaf_coverage <= 0.01:
            non_leaf_F1 = 0
        else:
            non_leaf_F1 = 2 * non_leaf_precision * non_leaf_coverage / (non_leaf_precision + non_leaf_coverage)

        key = len(non_leaf_nums)   # depth of problem
        if key not in bins_non_leaf_coverage.keys():
            bins_non_leaf_coverage[key] = []
            bins_non_leaf_precision[key] = []
            bins_non_leaf_F1[key] = []
            bins_ans_accuracy[key] = []
            
        bins_non_leaf_coverage[key].append(non_leaf_coverage)
        bins_non_leaf_precision[key].append(non_leaf_precision)
        bins_non_leaf_F1[key].append(non_leaf_F1)
        
        # answer accuracy
        nums = extract_nums(raw_ans)
        if nums and approx_eq(eval(data_entry['answer']), nums[-1]):
            ans_accuracy = 1
            predicted_item["is_correct"] = True
            predicted_ans = nums[-1]
        else:
            ans_accuracy = 0
            predicted_item["is_correct"] = False 
            predicted_ans = ""
        bins_ans_accuracy[key].append(ans_accuracy)

        predicted_item['answer'] = predicted_ans
        
        non_leaf_coverage_l.append(non_leaf_coverage)
        non_leaf_precision_l.append(non_leaf_precision)
        non_leaf_F1_l.append(non_leaf_F1)
        accuracy_l.append(ans_accuracy)

        predicted_item["rationale"] = predicted_rationale
        predicted_item["rationale_tokens"] = predicted_rationale_tokens
        predicted_item["nonleaf_nums"] = predicted_nonleaf_nums
        predicted_item["steps"] = predicted_steps

        question_id = question2id[predicted_item["question"]]
        list_questionid.append(question_id)
        data[question_id]["ground_truth"] = predicted_item["ground_truth"]
        predicted_item.pop("ground_truth")
        data[question_id]["prediction"] = predicted_item
        data[question_id]["prediction"].pop("question")

    accuracy = np.mean(accuracy_l) # round(np.mean(accuracy_l), 2)
    recall = np.mean(non_leaf_coverage_l) # average non-leaf num recall
    precision = np.mean(non_leaf_precision_l) # average non-leaf num precision 
    f1 = np.mean(non_leaf_F1_l) # average non-leaf num f1 
    dict_eval_others = dict()
    dict_eval_others["accuracy"] = accuracy
    dict_eval_others["recall"] = recall 
    dict_eval_others["precision"] = precision 
    dict_eval_others["f1"] = f1
    accuracy_depth = []
    coverage_depth = []
    precision_depth = []
    F1_depth = []
    count_depth = dict()
    for depth in range(1, 9):
        if depth in bins_ans_accuracy.keys(): 
            count_depth[depth] = len(bins_ans_accuracy[depth])
            accuracy_depth.append(np.mean(bins_ans_accuracy[depth]))
            coverage_depth.append(np.mean(bins_non_leaf_coverage[depth]))
            precision_depth.append(np.mean(bins_non_leaf_precision[depth]))
            F1_depth.append(np.mean(bins_non_leaf_F1[depth])) 
    dict_eval_others["accuracy_depth"] = accuracy_depth
    dict_eval_others["coverage_depth"] = coverage_depth 
    dict_eval_others["precision_depth"] = precision_depth 
    dict_eval_others["F1_depth"] = F1_depth
    dict_eval_others["count_depth"] = count_depth

    for i in list_questionid: 
        data[i]["eval_others"] = dict_eval_others
 
    return data, question2id, list_questionid 


def util_svamp(file_path, num_test):
    # file contains a list of dict with keys: question, answer, equation, type, id, query, ans_*
    if file_path.endswith(".jsonl"):
        temp = read_jsonl(file_path)
        # last element is the prompt
        result, prompt = temp[:-1], temp[-1]
    else:
        with open(file_path, "r") as f:
            temp = json.load(f)
            # last element is the prompt
            result, prompt = temp[:-1], temp[-1]
    if num_test == -1:
       num_test = len(result)
    assert len(result) == num_test

    list_accuracy = []
    for index in range(len(result)):
        predicted_item = result[index]
        result[index]["prediction"] = {}
        for key_ in predicted_item.keys():
            if key_.startswith('ans_'):
                result[index]["prediction"][key_[4:]] = predicted_item[key_]
                predicted_rationale, raw_answer = get_rationale(predicted_item[key_]) 
                result[index]["prediction"]["rationale"] = predicted_rationale
                # answer accuracy
                predicted_ans = extract_answer(raw_answer, dataset="svamp")  
                if predicted_ans != None and approx_eq(eval(str(result[index]['answer'])), predicted_ans):
                    ans_accuracy = 1
                    result[index]["prediction"]["is_correct"] = True
                else:
                    ans_accuracy = 0
                    result[index]["prediction"]["is_correct"] = False 
                result[index]["prediction"]['answer'] = predicted_ans
                list_accuracy.append(ans_accuracy)
                break
    accuracy = np.mean(list_accuracy)
    for index in range(len(result)):
        result[index]["prediction"]["eval_others"] = {"accuracy": accuracy} # round(accuracy, 2)
    return result 


def util_multiarith(file_path, num_test):
    # file contains a list of dict with keys: question, answer, equation, equation_number, id, query, ans_*
    if file_path.endswith(".jsonl"):
        temp = read_jsonl(file_path)
        # last element is the prompt
        result, prompt = temp[:-1], temp[-1]
    else:
        with open(file_path, "r") as f:
            temp = json.load(f)
            # last element is the prompt
            result, prompt = temp[:-1], temp[-1]
    if num_test == -1:
       num_test = len(result) 
    assert len(result) == num_test

    list_accuracy = []
    for index in range(len(result)):
        predicted_item = result[index]
        result[index]["prediction"] = {}
        for key_ in predicted_item.keys():
            if key_.startswith('ans_'):
                result[index]["prediction"][key_[4:]] = predicted_item[key_]
                predicted_rationale, raw_answer = get_rationale(predicted_item[key_]) 
                # predicted_rationale = predicted_item[key_].split("Q:")[0].strip(" .")
                result[index]["prediction"]["rationale"] = predicted_rationale
                # answer accuracy
                if "zero_shot" not in file_path: 
                    predicted_ans = extract_answer(raw_answer, dataset="multiarith")
                else:
                    predicted_ans = extract_answer(raw_answer, dataset="multiarith|zero_shot") 
                # nums = extract_nums(raw_answer)
                if predicted_ans != None and approx_eq(eval(str(result[index]['answer'][0])), predicted_ans):
                    ans_accuracy = 1
                    result[index]["prediction"]["is_correct"] = True
                else:
                    ans_accuracy = 0
                    result[index]["prediction"]["is_correct"] = False 
                result[index]["prediction"]['answer'] = predicted_ans 
                list_accuracy.append(ans_accuracy)
                break
    accuracy = np.mean(list_accuracy)
    for index in range(len(result)):
        result[index]["prediction"]["eval_others"] = {"accuracy": accuracy} # round(accuracy, 2)
    return result


def util_mathqa(file_path, num_test):
    # file contains a list of dict with keys: question, answer, rationale, equation, equation_linear, type, query, ans_*
    if file_path.endswith(".jsonl"):
        temp = read_jsonl(file_path)
        # last element is the prompt
        result, prompt = temp[:-1], temp[-1]
    else:
        with open(file_path, "r") as f:
            temp = json.load(f)
            # last element is the prompt
            result, prompt = temp[:-1], temp[-1]
    if num_test == -1:
       num_test = len(result) 
    assert len(result) == num_test

    # dictionary maps question to id
    dataset_name = "mathqa"
    data = read_json(DEFAULT_INPUT_ANN_PATH + dataset_name + "/test.json") 
    question2id = dict()
    for i in range(len(data)):
        item = data[i]
        question2id[item['Problem']] = i
    dict_output_path = DEFAULT_INPUT_GEN_PATH + "mathqa_question2id.json" 
    with open(dict_output_path, 'w') as outfile:
        json.dump(question2id, outfile)
        outfile.write('\n')
    print(f"Saved gsm8k question2id dict in {dict_output_path}")
    list_questionid = []

    list_accuracy = []
    for index in range(len(result)):
        predicted_item = result[index]
        data_entry = data[question2id[predicted_item['question']]]
        assert data_entry['Problem'] == predicted_item['question']
        question_id = question2id[predicted_item["question"]]
        list_questionid.append(question_id)

        result[index]["prediction"] = {}
        for key_ in predicted_item.keys():
            if key_.startswith('ans_'):
                result[index]["prediction"][key_[4:]] = predicted_item[key_] 
                predicted_rationale, predicted_ans = get_rationale(predicted_item[key_]) 
                result[index]["prediction"]["rationale"] = predicted_rationale
                # result[index]["prediction"]['answer'] = extract_answer(predicted_ans, dataset_name)
                result[index]["prediction"]['answer'] = extract_answer(predicted_item[key_], dataset_name)
                if result[index]["prediction"]['answer'].lower() == result[index]['answer'].lower():
                    result[index]["prediction"]["is_correct"] = True
                    ans_accuracy = 1
                else:
                    result[index]["prediction"]["is_correct"] = False
                    ans_accuracy = 0
                    # if result[index]["options"][result[index]['answer']] in raw_ans: 
                    #     # directly give the choice content w/o choice
                    #     result[index]["prediction"]["is_correct"] = True
                    #     ans_accuracy = 1
                    # else: 
                    #     result[index]["prediction"]["is_correct"] = False
                    #     ans_accuracy = 0
                list_accuracy.append(ans_accuracy)
                result[index]["id"] = question_id
                break
    accuracy = np.mean(list_accuracy)
    for index in range(len(result)):
        result[index]["prediction"]["eval_others"] = {"accuracy": accuracy} # round(accuracy, 2)
    return result
 

def util_csqa(file_path, num_test):
    # file contains a list of dict with keys: question, answer, concept, id, query, ans_*
    dataset_name = "csqa" 
    if file_path.endswith(".jsonl"):
        temp = read_jsonl(file_path)
        # last element is the prompt
        result, prompt = temp[:-1], temp[-1]
    else:
        with open(file_path, "r") as f:
            temp = json.load(f)
            # last element is the prompt
            result, prompt = temp[:-1], temp[-1]
    if num_test == -1:
       num_test = len(result) 
    assert len(result) == num_test

    list_accuracy = []
    for index in range(len(result)):
        predicted_item = result[index]
        result[index]["prediction"] = {}
        for key_ in predicted_item.keys():
            if key_.startswith('ans_'):
                result[index]["prediction"][key_[4:]] = predicted_item[key_]
                predicted_rationale, predicted_ans = get_rationale(predicted_item[key_])  
                # predicted_rationale = predicted_item[key_].lower().split("the answer ")[0].strip(" .")
                result[index]["prediction"]["rationale"] = predicted_rationale
                # result[index]["prediction"]['answer'] = extract_answer(predicted_ans, dataset_name)
                result[index]["prediction"]['answer'] = extract_answer(predicted_item[key_], dataset_name)
                if result[index]["prediction"]['answer'].lower() == result[index]['answer'].lower():
                    result[index]["prediction"]["is_correct"] = True
                    ans_accuracy = 1
                else:
                    result[index]["prediction"]["is_correct"] = False
                    ans_accuracy = 0
                list_accuracy.append(ans_accuracy) 
                break
    accuracy = np.mean(list_accuracy)
    for index in range(len(result)):
        result[index]["prediction"]["eval_others"] = {"accuracy": accuracy} # round(accuracy, 2)
    return result


def util_strategyqa(file_path, num_test):
    if file_path.endswith(".jsonl"):
        temp = read_jsonl(file_path)
        # last element is the prompt
        result, prompt = temp[:-1], temp[-1]
    else:
        with open(file_path, "r") as f:
            temp = json.load(f)
            # last element is the prompt
            result, prompt = temp[:-1], temp[-1]
    if num_test == -1:
       num_test = len(result) 
    assert len(result) == num_test

    for index in range(len(result)):
        predicted_item = result[index]
        for key_ in predicted_item.keys():
            if key_.startswith('ans_'):
                result[index]["prediction"][key_[4:]] = predicted_item[key_] 
                continue
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_name", default=None, type=str, required=True, help="type for prompt")
    parser.add_argument("--dataset", default=None, type=str, required=True, help="dataset for experiments")  
    parser.add_argument("--engine", default="gpt-3.5-turbo", type=str, required=True, help="engine")
    parser.add_argument("--num_test", default=-1, type=int, help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float, help="temperature for generation")
    parser.add_argument("--max_tokens", default=256, type=int, help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str, help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix", default="", type=str, help="")
    parser.add_argument("--apikey_file", default="./api_key.txt", type=str, help="file path for api key.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached generated rationale files in jsonl format")
    parser.add_argument("--overwrite_prediction", action="store_true", help="Overwrite the LLM-generated prediction result files in jsonl format")
    parser.add_argument(
        "--learning_type", default='few_shot', type=str, help='zero shot or few shot',
        choices=['zero_shot', 'few_shot']
    )
    parser.add_argument(
        "--reasoning_strategy", default='complex_cot', type=str, help='The reasoning strategy LLM applied to generate prediction',
        choices=['complex_cot', 'plan_solve']
    )
    parser.add_argument("--self_consistency", '--sc', action="store_true", 
        help="Whether apply self consistency or not"
    )
    parser.add_argument("--self_verification", '--sv', action="store_true", 
        help="Whether apply self verification or not"
    )
    parser.add_argument("--dialog_icl", action="store_true", 
        help="Whether apply dialog in-context learning or not"
    )

    args = parser.parse_args()
    print(args)

    test_model = args.engine + "|" + args.prompt_name 

    # scale down. -1 if not.
    NUM_TEST = args.num_test
    
    file_name = DEFAULT_INPUT_GEN_PATH + "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}.jsonl".format(
            args.dataset, args.prompt_name, args.engine, NUM_TEST, args.learning_type, 
            args.reasoning_strategy, args.self_consistency, args.self_verification, args.dialog_icl, args.suffix_ans) 

    if args.dataset == "gsm8k":
        util_gsm8k(file_name, NUM_TEST)
    elif args.dataset == "svamp":
        util_svamp(file_name, NUM_TEST)
    elif args.dataset == "multiarith":
        util_multiarith(file_name, NUM_TEST)
    elif args.dataset == "mathqa":
        util_mathqa(file_name, NUM_TEST)
    elif args.dataset == "csqa":
        util_csqa(file_name, NUM_TEST)
    elif args.dataset == "strategyqa":
        util_strategyqa(file_name, NUM_TEST)