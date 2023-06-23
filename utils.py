#!/usr/bin/env python3

# Copyright (c) EMNLP 2023 Submission

import datetime
import os
import json
import numpy as np
import re
import pathlib
from copy import deepcopy
from nltk.tokenize import sent_tokenize
from typing import Callable, Dict, Iterable, List, Tuple, Union

import torch

from constants import (
    COT_TRIGGER,
    STR_GEN_STOP, 
    DICT_STR_SPLIT_RATIONALE
)


def read_jsonl(path: str):
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def read_json(path: str):
    with open(path, "r", encoding='utf-8') as file:
        return json.load(file)
    
def mkpath(path):
    if not os.path.exists(path):
        os.mkdir(path)


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


def get_update_inputs(dataset, prompt, question, model, engine, learning_type, dialog_icl):
    if engine in ["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"]:
        list_messages = []
        system_content = "Follow the given examples and answer the question."
        if learning_type == "zero_shot":
            system_content = "Please answer the question following the given instruction."
        list_messages.append({"role": "system", "content": system_content})
        if not dialog_icl: # "zero_shot" must be not dialog_icl 
            prompt_query = "Question: {}{}Answer: {}".format(question, DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP], DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP])
            if learning_type == "few_shot": 
                prompt_query = prompt + "{}{}".format(STR_GEN_STOP, prompt_query)
            if model == "plan_solve":
                prompt_query += make_plan_slove_instruction(dataset)
            else:
                if COT_TRIGGER.lower() in prompt.lower() or learning_type == "zero_shot" and model != "self_verification": # cot, otherwise direct answer
                    prompt_query += COT_TRIGGER + DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP]
                elif model == "self_verification":
                    prompt_query += DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP]
            list_messages.append({"role": "user", "content": prompt_query})
        else:
            list_messages.extend(make_dialog_prompt(dataset, prompt, question, model))
        return list_messages  
    else: # ["davinci", "text-davinci-002", "text-davinci-003"]
        inputs = prompt + "{}Question: {}{}Answer: {}".format(STR_GEN_STOP, question, DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP], DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP]) 
        if model == "plan_solve":
            inputs += make_plan_slove_instruction(dataset)
        return inputs
 
def make_dialog_prompt(dataset, prompt, query, model):
    messages = []
    # messages.append({"role": "system", "content": "Follow the given examples and answer the question."})
    list_cases = prompt.split(STR_GEN_STOP)
    for case in list_cases:
        str_split = "Answer: "
        assert str_split in case
        question = case.split(str_split)[0].strip()
        messages.append({"role": "user", "content": question})
        answer = case.split(str_split)[-1].strip()
        messages.append({"role": "assistant", "content": answer})
    if model == "plan_solve":
        instruction = make_plan_slove_instruction(dataset)
    else:
        if COT_TRIGGER.lower() in prompt.lower():
            instruction = COT_TRIGGER
        else: # direct
            instruction = "The answer is " # ""  
    messages.append({"role": "user", "content": query + "\n" + instruction})
    return messages

def make_plan_slove_instruction(dataset):
    prompt_101 = "Let's think step by step."
    prompt_201 = "Let's first understand the problem and devise a plan to solve the problem. " \
                "Then, let's carry out the plan to solve the problem step by step."
    prompt_301 = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                "and devise a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to " \
                "correct numeral calculation and commonsense), solve the problem step by step, and show the answer."
    prompt_302 = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                "and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
                "(pay attention to correct numerical calculation and commonsense), " \
                "solve the problem step by step, and show the answer."
    prompt_303 = "Let's devise a plan and solve the problem step by step."
    prompt_304 = "Let's first understand the problem and devise a complete plan. " \
                "Then, let's carry out the plan and reason problem step by step. Every step answer the subquestion, " \
                "\"does the person flip and what is the coin's current state?\". According to the coin's last state, " \
                "give the final answer (pay attention to every flip and the coin’s turning state)."
    prompt_305 = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                "and make a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay " \
                "attention to correct numerical calculation and commonsense), " \
                "solve the problem step by step, and show the answer."
    prompt_306 = "Let's first prepare relevant information and make a plan. Then, let's answer the question step by step " \
                "(pay attention to commonsense and logical coherence)."
    prompt_307 = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
                "(pay attention to correct numerical calculation and commonsense), " \
                "solve the problem step by step, and show the answer."
    prompt_308 = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                "and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
                "(pay attention to correct numerical calculation), " \
                "solve the problem step by step, and show the answer."
    prompt_309 = "Let's first understand the problem, extract relevant commonsense or facts, " \
                "and devise a complete plan. Then, let's carry out the plan, calculate intermediate commonsense or facts " \
                "(pay attention to correct commonsense), " \
                "solve the problem step by step, and show the answer."
    if dataset in ["gsm8k", "svamp", "multiarith", "mathqa"]:
        instruct_id = 308
    elif dataset == "csqa":
        instruct_id = 309
    try:
        return eval('prompt_{}'.format(str(instruct_id)))
    except NameError as e:
        raise NameError('can\'t find prompt_id: {}'.format(instruct_id))

def tokenize(a):
    """
    lower, split, strip each token
    """
    b = a.lower().split()
    for ii in range(len(b)):
        b[ii] = b[ii].strip().strip('?.,\"\'').strip()
    return b

def approx_eq(a, b):
    return abs(a - b) < 0.01

def approx_in(b, array):
    for a in array:
        if approx_eq(a, b):
            return True
    return False

def approx_overlap(a, b):
    count = 0
    a = {eval(var) for var in a}
    b = {eval(var) for var in b}
    for item in a:
        if approx_in(item, b):
            count += 1
    return count
    
def find_nums(string): # str list
    return re.findall(r"\d*\.?\d+(?:/\d*\.?\d+)?", string)

def extract_nums(s): # int list
    s = s.replace(",", "")
    nums = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
    return_list = []
    for i in range(len(nums)):
        try:
            return_list.append(eval(nums[i].strip().lstrip(" 0")))
        except:
            pass
    return return_list

def extract_number(text: str, dataset) -> Union[float, None]:
    # text = text.replace(',', '')
    # pred = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    # if pred:
    #     pred_answer = float(pred[-1])
    # else:
    #     pred_answer = None
    return_list = extract_nums(text) 
    if return_list:
        if dataset == "multiarith|zero_shot": # "multiarith" 
        # pred_answer = return_list[-1]
            pred_answer = return_list[0]
        else:
            pred_answer = return_list[-1]
    else:
        pred_answer = None
    return pred_answer

def find_formula(step):
    assert step.count("<<") == step.count(">>") == 1
    left, right = step.find("<<")+2, step.find(">>")
    return step[left: right]

def extract_finance(text):
    pattern = '-?\d+\.?\d*%?'
    pred = re.findall(pattern, text)
    if pred:
        if '%' == pred[-1][-1]:
            pred_answer = eval(pred[-1][:-1] + '/100')
        else:
            pred_answer = float(pred[-1])
        return pred_answer
    pattern = 'yes|no'
    pred = re.findall(pattern, text)
    if pred:
        return pred[-1]
    return None

def extract_answer(text, dataset):
    if dataset in ["svamp", "gsm8k", "multiarith", "multiarith|zero_shot"]:
        pred_answer = extract_number(text, dataset)
    elif dataset == "csqa":
        pred = text.strip()
        pred = re.sub("\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        list_candidate = [i for i in pred if i in ('A|B|C|D|E')]
        if len(list_candidate) > 0:
            pred_answer = list_candidate[-1]
        else:
            pred_answer = "Z" 
        # pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "mathqa":
        pred = text.strip()
        pred = re.sub("\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        list_candidate = [i for i in pred if i in ('a|b|c|d|e')]
        if len(list_candidate) > 0:
            pred_answer = list_candidate[-1]
        else:
            pred_answer = "z" 
        # pred_answer = [i for i in pred if i in ('A|B|C|D|E')][-1]
        # pred_answer = re.findall(r'a|b|c|d|e', pred)[0]
        return pred_answer
    elif dataset == "strategyqa" or dataset == 'coin_flip':
        pred = text.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", text)
        pred_answer = pred
        return pred_answer
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def extract_if_correct(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        assert False

def test_answer(pred_str, ans_str):
    """Find the last number as the predicted answer"""
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1):
        # print(pred_str)
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        # print(ans_str)
        gold = gold[-1]
        return pred == gold
    else: return False 

def save_scores(score_dict: Dict, out_path: str) -> None:
    # create destination directory if not exists
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    score_list = list(score_dict.keys())
    with open(out_path, 'w') as output_file:
        n_scores = len(score_list)
        out_line = "{:<8} " + " ".join(["{:<15}" for i in range(n_scores)])
        print(out_line.format('ID', *score_list), file=output_file)
        n_samples = len(score_dict[score_list[0]])
        for i in range(n_samples):
            scores = []
            for score in score_list:
                scores.append(score_dict[score][i])
            print(out_line.format(i, *scores), file=output_file)
    print(f"Scores written to {out_path}")

def get_rationale(input): # str --> str, str 
    rationale = input
    raw_answer = input.split(DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP])[-1].strip() 
    list_answer_trigger = [
        "Therefore, the answer ", "therefore, the answer ", "therefore, The answer ", "Therefore, The answer ",
        "Therefore the answer ", "therefore the answer ", "therefore The answer ", "Therefore The answer ", 
        "Thus, the answer ", "thus, the answer ", "thus, The answer ", "Thus, The answer ", 
        "Thus the answer ", "thus the answer ", "thus The answer ", "Thus The answer ",
        "So, the answer ", "so, the answer ", "so, The answer ", "So, The answer ", 
        "So the answer ", "so the answer ", "so The answer ", "So The answer ",
        "Answer: ", "answer: ", "A: ",
        "The answer: ", "The Answer: ", "the answer: ", "the Answer: ", 
        "The answer ", "The Answer ", "the answer ", "the Answer ",
        "The answer\n", "The Answer\n", "the answer\n", "the Answer\n"  
    ]
    for answer_trigger in list_answer_trigger: 
        if answer_trigger in input :
            index = input.rfind(answer_trigger)
            rationale = input[:index].strip()
            # raw_answer = input[index:].strip()
            raw_answer = input[index+len(answer_trigger):].replace("is", "").strip().split(". ")[0].strip()
            # raw_answer = input[index+len(answer_trigger):].replace("is", "").strip()
            break 
    return rationale, raw_answer


def print_and_reset_max_gpu_memory() -> None:
    max_gpu_mem_alloc = int(torch.cuda.max_memory_allocated() // 1e6)
    print(f"Max GPU Memory Allocated: {max_gpu_mem_alloc} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def cosine_similarity_scaled(list1: np.ndarray, list2: np.ndarray) -> float:
    """
    Normalized cosine similarity for *normalized* embeddings.

    Normalized cosine similarity takes values from [0;1]
    """
    cosine_sim = np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))
    return (1.0 + cosine_sim) / 2.0


def embedding_alignment(ref_emb: np.ndarray, hypo_emb: np.ndarray) -> List[float]:
    """
    Return embedding matching alignment for each item in hypo_emb
    ref_emb: list of reference embeddings
    hypo_emb: list oh hypothesises embeddings
    """
    scores = []
    for he in hypo_emb:
        # some embeddings can be empty. For example, for latex-style equations, or empty string
        if len(he) > 0:
            out = [cosine_similarity_scaled(he, re) for re in ref_emb if len(re) > 0]
            if len(out) > 0:
                scores.append(max(out))
    return scores


def al_mean(alignment_scores) -> float:
    if len(alignment_scores) == 0:
        return 0.0
    return sum(alignment_scores) / len(alignment_scores)


# def split_gsm8k_gpt3_generations_to_steps(reasoning: str) -> List[str]:(reasoning: str) -> List[str]:
def split_gsm8k_llm_generations_to_steps(reasoning: str) -> List[str]:
    """
    This logic is copied directly from the code that parsed GSM8K generations into steps
    for annotation.
    """
    predicted_rationale = reasoning.strip(" .")
    if DICT_STR_SPLIT_RATIONALE["\n"] not in predicted_rationale and DICT_STR_SPLIT_RATIONALE["\n\n"] not in predicted_rationale or len(predicted_rationale) < 2:
        return ["None", "None"]
    # get rid of in-token ",", artifact for long numbers
    predicted_rationale_tokens = predicted_rationale.split()
    for i in range(len(predicted_rationale_tokens)):
        token = predicted_rationale_tokens[i]
        if "," in token[1:-1]:
            token = token[0] + token[1:-1].replace(",","") + token[-1]
        predicted_rationale_tokens[i] = token
    predicted_rationale = " ".join(predicted_rationale_tokens)
    predicted_steps = predicted_rationale.split(DICT_STR_SPLIT_RATIONALE[STR_GEN_STOP])
    return predicted_steps
    # return [
    #     split
    #     for s in sent_tokenize(reasoning)
    #     # for split in s.split("\n")
    #     for split in s.split(". ")
    #     if len(split) > 0
    # ]


def assert_all_elements_same_length(
    elements: Iterable,
    error_msg: str = 'not all elements have the same length',
) -> int:
    """
    Asserts that all elements in the iterable have the same length.

    Can be useful when you have a list of lists representing rows or columns, for
    example. Returns the length.
    """
    unique_lengths = set(len(l) for l in elements)
    assert len(unique_lengths) == 1, f"{error_msg} | {unique_lengths}"
    return list(unique_lengths)[0]


def split_list(
    input_list: Iterable[str],
    include_condition: Callable[[str], bool],
) -> Tuple[List[str], List[str]]:
    """
    Splits a list into two based on a condition applied to each element.
    """
    matching_list = [x for x in input_list if include_condition(x)]
    other_list = [x for x in input_list if x not in matching_list]
    return matching_list, other_list


def ordered_union(list_of_lists: List[List[str]]) -> List[str]:
    """
    Unpacks a list of lists, ensuring there are no duplicates in the final list.
    """
    union_list = []
    for l in list_of_lists:
        for item in l:
            if item not in union_list:
                union_list.append(item)
    return union_list


def raw2id_on_question(filepath, subfold):
    for root, _dirnames, filenames in os.walk(filepath + subfold):
            for filename in filenames:
                if ".json" not in filename:
                    continue
                raw_data = read_json(filename)
    raw2id = dict() 
    for item in raw_data:
        question = item["sQuestion"]
        raw2id[question] = item
        raw2id[question]["id"] = item["iIndex"] + "_" + subfold
    return raw2id


def preprocess_multiarith(filepath, outpath):
    dict_preprocessed = raw2id_on_question(filepath, "/commoncore")
    dict_preprocessed.update(raw2id_on_question(filepath, "/illinois"))
    list_preprocessed = []
    for question in dict_preprocessed.keys():
        list_preprocessed.append(dict_preprocessed[question])
    with open(outpath, 'w') as outfile:
        for line in list_preprocessed:
            json.dump(line, outfile)
            outfile.write('\n')
    # return list_preprocessed 
     


