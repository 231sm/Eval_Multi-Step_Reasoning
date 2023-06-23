#!/usr/bin/env python3

# Copyright (c) EMNLP 2023 Submission

from collections import Counter
import os
import random
import re
import time
import logging
import openai
import json
import jsonlines
import numpy as np
from utils import get_rationale, get_update_inputs, mkpath, preprocess_multiarith, print_now, read_jsonl, read_json, extract_answer
import argparse
import pandas as pd

import datetime

from constants import (
    MAX_RETRY,
    API_TIME_INTERVAL,
    DEFAULT_INPUT_ANN_PATH, 
    DEFAULT_INPUT_GEN_PATH, 
    DEFAULT_INPUT_RES_PATH, 
    DEFAULT_PROMPT_PATH, 
    DEFAULT_OUTPUT_PATH,
    DEFAULT_LOG_PATH,
    DATASETS,
    STR_GEN_STOP,
    DICT_STR_SPLIT_RATIONALE,
    PROMPT_NAME_ZSL,
    N_FOR_MULTI_CHAINS,
    TEMP_FOR_MULTI_CHAINS,
    MAX_TOKEN_FOR_SELFVERIFY
)

log_file = DEFAULT_LOG_PATH + "0-temp.log" 
# logging.basicConfig(filename=log_file)
formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
sh = logging.StreamHandler()
fh = logging.FileHandler(filename=log_file)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(fh)
logger.addHandler(sh)

# now = print_now(1).split(' ')[0].replace('/', '-')

# Result_Folder = 'result/{}'.format(now)
# mkpath('result')
# mkpath(Result_Folder)
# mkpath(f'{Result_Folder}/{args.dataset}')

# Log_Folder = DEFAULT_LOG_PATH + '{}'.format(now)
# mkpath('log')
# mkpath(Log_Folder)
# mkpath(f'{Log_Folder}/{args.dataset}')


def get_answer_from_gpt(args, dataset, prompt, question, model, learning_type, dialog_icl, if_return_all, engine, max_tokens=256, temperature=0.0):
    inputs = get_update_inputs(dataset, prompt, question, model, engine, learning_type, dialog_icl)
    # print(inputs)
    n = N_FOR_MULTI_CHAINS if if_return_all else 1
    if engine in ["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"]:
        if args.reasoning_strategy != "plan_solve" and args.learning_type != "zero_shot" and learning_type != "zero_shot":
            response = openai.ChatCompletion.create(
                model=engine,
                messages=inputs,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                n=n,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=[STR_GEN_STOP]
            )
        else:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=inputs,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                n=n,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        if not if_return_all: 
            return response['choices'][0]['message']['content'].strip()
        else:
            text = response["choices"]
            tem_rational = []
            for i in range(len(text)):
                tem_rational.append(text[i]['message']['content'].strip())
            return tem_rational 
    else: # ["davinci", "text-davinci-002", "text-davinci-003"] 
        if args.reasoning_strategy != "plan_solve" and args.learning_type != "zero_shot" and learning_type != "zero_shot": 
            response = openai.Completion.create(
                engine=engine,
                prompt=inputs,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                n=n,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=[STR_GEN_STOP]
            )
        else:
            response = openai.Completion.create(
                engine=engine,
                prompt=inputs,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                n=n,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ) 
        if not if_return_all: 
            return response['choices'][0]['text'].strip()
        else:
            text = response["choices"]
            tem_rational = []
            for i in range(len(text)):
                tem_rational.append(text[i]['text'].strip())
            return tem_rational


def get_prompt(prompt_file_name):
    # load prompts
    with open(prompt_file_name, "r", encoding='utf-8') as f:
        prompt = f.read().strip()
    return prompt


def get_annotated_data(annotation_dataset_path, dataset_name):
    if dataset_name == "gsm8k":
        test_data = read_jsonl(annotation_dataset_path + dataset_name + "/grade_school_math/data/test.jsonl")
        example_data = read_jsonl(annotation_dataset_path + dataset_name + "/grade_school_math/data/example_model_solutions.jsonl")
        for i in range(len(test_data)):
            test_data[i]["add_example"] = example_data[i] 
        qa_pairs = [(instance['question'], 
                    {"question": instance['question'], 
                     "answer": extract_answer(instance['answer'], dataset_name), 
                     "example_result": instance["add_example"]["175b_verification"]["solution"],
                     "ground_truth": instance["add_example"]["ground_truth"]}) for instance in test_data]
    elif dataset_name == "svamp":
        # test_data = read_json(annotation_dataset_path + dataset_name + "/testset.json")
        test_data = read_json(annotation_dataset_path + dataset_name + "/SVAMP.json")  
        qa_pairs = [(instance['Body']+" "+instance['Question'], 
                    {"question": instance['Question'],
                     "body": instance['Body'], 
                     "answer": instance['Answer'], 
                     "equation": instance['Equation'], 
                     "type": instance['Type'],
                     "id": instance['ID']}) for instance in test_data] 
    elif dataset_name == "multiarith":
        # save_path = annotation_dataset_path + dataset_name + "/precessed.json" 
        # preprocess_multiarith(annotation_dataset_path + dataset_name, save_path)
        # test_data = read_json(save_path) 
        test_data = read_json(annotation_dataset_path + dataset_name + "/commoncore/questions.json")
        qa_pairs = [(instance["sQuestion"],
                     {"question": instance["sQuestion"],
                      "answer": instance["lSolutions"], # list
                      "equation": instance["lEquations"], # list
                      "equation_number": instance["lAlignments"], # list
                      "id": instance["iIndex"]}) for instance in test_data]
                    #   "id": instance["id"] 
    elif dataset_name == "mathqa":
        test_data = read_json(annotation_dataset_path + dataset_name + "/test.json") 
        qa_pairs = [(instance['Problem'] + "\nOptions: " + instance['options'], 
                    {"question": instance['Problem'],
                     "options": instance['options'], # str_choice_cont --> dict_choice_cont 
                     "answer": instance['correct'], # choice 
                     "rationale": instance['Rationale'],  
                     "equation": instance['annotated_formula'], # e.g., add(add(divide(1000, const_10), multiply(subtract(const_10, 1), const_10)), const_2)
                     "equation_linear": instance['linear_formula'], # e.g., divide(n0,const_10)|subtract(const_10,n1)|multiply(#1,const_10)|add(#0,#2)|add(#3,const_2) 
                     "type": instance['category']}) for instance in test_data]
        for i in range(len(qa_pairs)):
            answer = qa_pairs[i][1] 
            options_str = answer["options"].strip()
            list_choice_cont = options_str.split(" , ")
            qa_pairs[i][1] ["options"] = dict()
            for choice_cont in list_choice_cont: 
                temp = choice_cont.split(" ) ")
                # there are some bad annotations, such as a ) a )
                if len(temp) == 2: 
                    qa_pairs[i][1] ["options"][temp[0].strip()] = temp[1].strip()
                elif len(temp) > 2:
                    qa_pairs[i][1] ["options"][temp[-2].strip()] = temp[-1].strip()
                else:
                    qa_pairs[i][1] ["options"]["all"] = options_str.strip() 
    elif dataset_name == "csqa":
        test_data = read_jsonl(annotation_dataset_path + dataset_name + "/dev_rand_split.jsonl")  
        qa_pairs = [] 
        for instance in test_data: 
            query = instance['question']['stem'] + "\nAnswer Choices: \n"
            for option in instance['question']["choices"]:
                query += "(" + option["label"] + ") " + option["text"] + "\n" 
            answer = {"question": instance['question']['stem'],
                      "options": instance['question']["choices"], 
                      "answer": instance['answerKey'], 
                      "concept": instance['question']["question_concept"], 
                      "id": instance['id']}
            dict_polish = dict()
            for dict_label_text in answer["options"]:
                dict_polish[dict_label_text["label"]] = dict_label_text["text"]
            answer["options"] = dict_polish 
            qa_pairs.append((query, answer)) 
    # elif dataset_name == "strategyqa":
    #     test_data = 
    #     qa_pairs =   
    # elif dataset_name == "bamboogle":
    #     data = pd.read_csv(annotation_dataset_path + dataset_name + "/Bamboogle Prerelease - Sheet1.csv", encoding='unicode_escape') # encoding='unicode_escape', "utf-8"
    #     qa_pairs = list(zip(data.Question, data.Answer)) 
 
    return qa_pairs


def get_rationale_from_gpt(dataset, qa_pairs, engine="gpt-3.5-turbo", max_tokens=256, temperature=0.0):
    times_limit = 100
    decoder_error_file = DEFAULT_LOG_PATH + "decode_error/" + "decode_error_gen-gold_{}.jsonl".format(dataset) 
    writer_error = jsonlines.open(decoder_error_file, mode='w')
    for i in range(len(qa_pairs)):
        answer = qa_pairs[i][1]
        list_messages = [] 
        if dataset == "svamp":
            instruction = "Please generate the rationale with the given question, equation, and answer for multi-step arithmetic reasoning tasks on SVAMP dataset."
            query = "Question: " + answer["question"] + "\n" + "Equation: " + answer["equation"] + "\n" + "Answer: " + str(answer["answer"])  + "\n" + "Therefore, the rationale is: " 
        if dataset == "multiarith":
            instruction = "Please generate the rationale with the given question, equation, and answer for multi-step arithmetic reasoning tasks on MultiArith dataset." 
            query = "Question: " + answer["question"] + "\n" + "Equation: " + answer["equation"][0] + "\n" + "Answer: " + str(answer["answer"][0])  + "\n" + "Therefore, the rationale is: "
        elif dataset == "mathqa":
            instruction = "Please regenerate the rationale to make it expressed in natural language, with the given question, rationale, equation, and answer for multi-step reasoning tasks on MathQA dataset."
            query = "Question: " + answer["question"] + "\n" + "Rationale: " + answer["rationale"] + "\n" + "Equation: " + answer["equation"] + "\n" + "Answer: " + answer["answer"]  + "\n" + "Therefore, the rationale is: "
        elif dataset == "csqa":
            instruction = "Please generate the rationale with the given question and answer for multi-step commonsense reasoning tasks on CommonsenseQA dataset."
            query = "Question: " + answer["question"] + "\n" + "Answer: " + answer["answer"] + "\n" + "Therefore, the rationale is: "
        
        print("currently", dataset, engine, "#", i)
        if_get_result = False
        retry = 0
        while not if_get_result:
            try:
                if engine in ["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"]:  
                    list_messages.append({"role": "system", "content": instruction})
                    list_messages.append({"role": "user", "content": query})
                    # print(list_messages) 
                    response = openai.ChatCompletion.create(
                        model=engine,
                        messages=list_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                        # stop=[STR_GEN_STOP]
                    )
                    qa_pairs[i][1]["rationale_example"] = response['choices'][0]['message']['content'].replace("\n\n", "\n").replace("Rationale:", "").replace("rationale:", "").strip()
                    # .replace("\n\n", "\n").replace("Rationale:", "").replace("rationale:", "")
                else: # ["davinci", "text-davinci-002", "text-davinci-003"]  
                    response = openai.Completion.create(
                        engine=engine,
                        prompt=instruction + "\n" + query,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                        # stop=[STR_GEN_STOP]
                    )
                    qa_pairs[i][1]["rationale_example"] = response['choices'][0]['text'].replace("\n\n", "\n").replace("Rationale:", "").replace("rationale:", "").strip()
                    # .replace("\n\n", "\n").replace("Rationale:", "").replace("rationale:", "")
                if_get_result = True
            except openai.error.RateLimitError as e:
                if e.user_message == 'You exceeded your current quota, please check your plan and billing details.':
                    raise e
                elif retry < MAX_RETRY:
                    time.sleep(API_TIME_INTERVAL)
                    retry += 1
                else:
                    print(e.user_message)
                    break
            except Exception as e:
                decode_error_data = {
                    "dataset": dataset, "count": i, 
                    "question": answer["question"] 
                }
                writer_error.write(decode_error_data)
                continue
                # raise e
        if i % times_limit == 0: 
            time.sleep(API_TIME_INTERVAL)
    writer_error.close() 
    return qa_pairs


def question_turn_decalrative(args, dataset, text, answer, answers_0, declarative, prompt, model, learning_type, dialog_icl, if_return_all, engine):
    global new_question
    list_str_trigger_choices = ["Options: ", "Answer Choices: "] # 'Answer Choices' 
    # if 'Answer Choices' in text:
    #     text = text.split('Answer Choices')[0]
    for str_trigger_choices in list_str_trigger_choices:
        if str_trigger_choices in text:
            text = text.split(str_trigger_choices)[0]
            break
    if dataset in ("csqa"):
        verifier_text = ' Judge whether this statement is normal (yes or no)'
    else:
        verifier_text = " What is the answer of 'X'?"
    try:
        if dataset in ("csqa"):
            text = text.replace(',', '.')
            position_fullstop = text[::-1].find('.')

            question = text[len(text) - position_fullstop:]
            ts = text[:len(text) - position_fullstop]

            if ts[0] == ' ':
                ts = ts[1:]
            if ts[-1] != ' ':
                ts += ' '
            ts = ts.replace(' .', '.')

            declar = "Please change the questions and answers into a complete declarative sentences '{} The answer is {}'".format(question, answer)
    
            if declarative == '':
                try:
                    declarative = get_answer_from_gpt(args, dataset, prompt, declar, model, learning_type, dialog_icl, if_return_all, engine)
                except:
                    declarative = get_answer_from_gpt(args, dataset, prompt, declar, model, learning_type, dialog_icl, if_return_all, engine)
        
            sentences, ans = [], []
            sentences.append('"' + ts + declarative + '"' + verifier_text)
            ans.append('yes') 
            return sentences, ans, declarative 
    
        text = text.replace(',', '.')
        position_fullstop = text[::-1].find('.')

        question = text[len(text) - position_fullstop:]
        ts = text[:len(text) - position_fullstop]

        declar = "Please change the questions and answers into a complete declarative sentences '{} The answer is {}'".format(question, answer)

        if declarative == '':
            try:
               declarative = get_answer_from_gpt(args, dataset, prompt, declar, model, learning_type, dialog_icl, if_return_all, engine)
            except:
                declarative = get_answer_from_gpt(args, dataset, prompt, declar, model, learning_type, dialog_icl, if_return_all, engine)
        else: 
            if answers_0 in declarative:
                declarative = declarative[:len(declarative) - declarative[::-1].find(answers_0[::-1]) - len(
                    answers_0)] + answer + declarative[len(declarative) - declarative[::-1].find(answers_0[::-1]):]
            else:
                try:
                    declarative = get_answer_from_gpt(args, dataset, prompt, declar, model, learning_type, dialog_icl, if_return_all, engine) 
                except: 
                    declarative = "{} The answer is {}.".format(question, answer)
            
        new_question_number = [s for s in re.findall(r'-?\d+\.?\d*', ts)]

        sentences, ans = [], []
        for nqn in range(len(new_question_number)):
            new_ts = ''
            number_find = False
            for i in ts.split('.'):
                if new_question_number[nqn] in i and number_find == False:
                    new_question = [p for p in i]
                    new_question[
                    i.find(new_question_number[nqn]):i.find(new_question_number[nqn]) + len(new_question_number[nqn])] = "'X'"
                    new_question = ''.join(new_question) + '.'
                    new_question.replace(' .', '.')
                    new_ts += new_question
                else:
                    new_ts += i + '.'
            new_ts = new_ts.replace('..', '.')

            if new_ts[0] == ' ':
                new_ts = new_ts[1:]
            if new_ts[-1] != ' ':
                new_ts += ' '
            new_ts = new_ts.replace(' .', '.')

            sentences.append('"' + new_ts + declarative + '"' + verifier_text)
            ans.append(new_question_number[nqn].replace('.', ''))
        return sentences[:3], ans[:3], declarative
    except:
        return '', '', ''

def get_chains_from_pre_generated(args, prompt_name, NUM_TEST):
    for if_self_consistency in [True, False]:
        for if_self_verification in [True, False]: 
            file_name_other = args.out_dir + "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}.jsonl".format(
                    args.dataset, prompt_name, args.engine, NUM_TEST, args.learning_type, 
                    args.reasoning_strategy, if_self_consistency, if_self_verification, args.dialog_icl, args.suffix_ans)
            if os.path.exists(file_name_other) and not args.overwrite_prediction:
                data = read_jsonl(file_name_other)
                if data is not None and len(data) > 0 and 'pred_chains_all' in data[0].keys():
                    return data 
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        '-d',
        type=str,
        required=False,
        default=DEFAULT_INPUT_ANN_PATH,
        help='Path to files with raw annotated data',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=False,
        default=DEFAULT_INPUT_GEN_PATH,
        help='Path where llm generation results will be saved.',
    )
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
    parser.add_argument("--prompt_name", default=None, type=str, required=False, help="type for prompt")
    parser.add_argument("--dataset", default=None, type=str, required=True, help="dataset for experiments")  
    parser.add_argument("--engine", default="gpt-3.5-turbo", type=str, help="engine")
    parser.add_argument("--num_test", default=-1, type=int, help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float, help="temperature for generation")
    parser.add_argument("--max_tokens", default=256, type=int, help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str, help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix_ans", default="", type=str, help="")
    parser.add_argument("--apikey_file", default="./api_key.txt", type=str, help="file path for api key.") 
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached generated rationale files in jsonl format")
    parser.add_argument("--overwrite_prediction", action="store_true", help="Overwrite the LLM-generated prediction result files in jsonl format")

    args = parser.parse_args()
    print(args)

    # load annotated data
    annotation_dataset_path = args.dataset_path
    qa_pairs = get_annotated_data(annotation_dataset_path, args.dataset)
    print("loading", args.dataset, "dataset completely. all together with", len(qa_pairs), "questions")
    # generate ground rationales with LLM
    with open(args.apikey_file, "r") as f:
        openai.api_key = f.read().strip() 

    if args.dataset in ["svamp", "multiarith", "mathqa", "csqa"]: 
        print("########## Use", args.engine, "API at", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "to generate absent rationales.")
        savefile_name = DEFAULT_INPUT_GEN_PATH + "rationale_generated/" + args.dataset + ".jsonl" 
        if not os.path.exists(savefile_name) or args.overwrite_cache:
            qa_pairs = get_rationale_from_gpt(args.dataset, qa_pairs) 
            writer = jsonlines.open(savefile_name, mode='w')
            for (_, answer) in qa_pairs:
                writer.write(answer) 
            writer.close()
        else:
            print(f"llm generated rationale file for {savefile_name} already exists. Skipping.")
            answer_update = read_jsonl(savefile_name)
            qa_pairs_update = []
            # print(len(answer_update), len(qa_pairs))
            for i in range(len(qa_pairs)):
                qa_pairs_update.append((qa_pairs[i][0], answer_update[i]))
            qa_pairs = qa_pairs_update 

    # scale down. -1 if not.
    NUM_TEST = args.num_test
    if NUM_TEST == -1:
        qa_pairs_test = qa_pairs
    else:
        if args.test_ind is None:
            np.random.seed(args.seed)
            rand_indices = np.random.choice(len(qa_pairs), NUM_TEST, replace=False)
            qa_pairs_test = [qa_pairs[i] for i in rand_indices]
        else:
            if ".npy" in args.test_ind: 
                test_index = np.load(args.test_ind)
            elif ".json" in args.test_ind:
                with open(args.test_ind, "r") as f:
                    test_index = json.load(f)
                    if len(test_index) != NUM_TEST:
                        test_index = np.random.choice(len(qa_pairs), NUM_TEST, replace=False)
                assert len(test_index) == NUM_TEST
            else:
                raise NotImplementedError(f"Index file {args.test_ind} not recognized.") 
            qa_pairs_test = [qa_pairs[i] for i in test_index]
    print("testing on", len(qa_pairs_test), "samples")

    # load prompts
    dict_prompt = dict() 
    for root, _dirnames, filenames in os.walk(DEFAULT_PROMPT_PATH + args.dataset):
        for filename in filenames:
            if ".txt" not in filename:
                continue 
            prompt_name = filename.split(".txt")[0]
            # prompt_name = os.path.basename(file)[:-4] 
            prompt_file_name = DEFAULT_PROMPT_PATH + args.dataset + "/" + filename
            prompt = get_prompt(prompt_file_name)
            dict_prompt[prompt_name] = prompt
            print("loading prompts completely. from", prompt_file_name, "for", args.dataset) 

    # generate results with LLM
    print("########## Use", args.engine, "API at", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "to generate predicted answer.") 
    decoder_error_file = DEFAULT_LOG_PATH + "decode_error/" + "decode_error_pred_{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}.jsonl".format(
            args.dataset, prompt_name, args.engine, NUM_TEST, args.learning_type, 
            args.reasoning_strategy, args.self_consistency, args.self_verification, args.dialog_icl, args.suffix_ans) 
    writer_error = jsonlines.open(decoder_error_file, mode='w')    

    for prompt_name in dict_prompt.keys():
        if args.prompt_name is not None and prompt_name != args.prompt_name:
            continue 
        prompt = dict_prompt[prompt_name]
        if args.learning_type == "zero_shot":
            prompt_name = PROMPT_NAME_ZSL 
        file_name = args.out_dir + "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}.jsonl".format(
            args.dataset, prompt_name, args.engine, NUM_TEST, args.learning_type, 
            args.reasoning_strategy, args.self_consistency, args.self_verification, args.dialog_icl, args.suffix_ans)
        if os.path.exists(file_name) and not args.overwrite_prediction:
            print(f"Prediction result file for {file_name} already exists. Skipping.")
            continue

        reasoning_strategy = args.reasoning_strategy if not args.self_verification else "complex_cot"
        # step1 for self_verification is like standard cot and need to return all reasoning chains 

        general_load_pre_result = False
        if args.self_consistency or args.self_verification: 
            list_dict_pred_all_chains = get_chains_from_pre_generated(args, prompt_name, NUM_TEST)

            if len(list_dict_pred_all_chains) > 1:
                general_load_pre_result = True
                list_dict_pred_all_chains = list_dict_pred_all_chains[:-1] # the last one is prompt
                len_loaded_file = len(list_dict_pred_all_chains)
                # print("len_loaded_file:", len_loaded_file)
                list_query_load = []
                for i in range(len_loaded_file):
                    list_query_load.append(list_dict_pred_all_chains[i]['query'])

        writer = jsonlines.open(file_name, mode='w')  
        count = 0
        times_limit = 100
        
        for (query, answer) in qa_pairs_test:
            load_pre_result = general_load_pre_result 
            result = dict() 
            result['query'] = query
            result.update(answer) 
            count += 1
            if load_pre_result and ( count > len_loaded_file or isinstance(list_dict_pred_all_chains[count-1], str) ): # len_loaded_file < len(qa_pairs_test) and count == len_loaded_file + 1
                # print(count, len_loaded_file, isinstance(list_dict_pred_all_chains[count-1], str))
                # break
                load_pre_result = False
            print("currently", args.dataset, args.engine, prompt_name, "#", count)

            if load_pre_result: # load all predicted chains from pre-generated files
                if len(list_query_load) > count - 1 and query == list_query_load[count-1]: 
                    current_pred_all_chains = list_dict_pred_all_chains[count-1]['pred_chains_all']
                    if args.reasoning_strategy == "plan_solve":
                        current_pred_plan = list_dict_pred_all_chains[count-1]['pred_plan']
                else:
                    load_pre_result = False
                    for i in range(len(list_query_load)):
                        if qa_pairs_test[count-1][0] == list_query_load[i]:
                            load_pre_result = True
                            current_pred_all_chains = list_dict_pred_all_chains[i]['pred_chains_all']
                            if args.reasoning_strategy == "plan_solve":
                                current_pred_plan = list_dict_pred_all_chains[i]['pred_plan']
                         
             
            max_tokens = args.max_tokens
            if "_direct" in prompt_name:
                if args.dataset == "gsm8k":
                    max_tokens = 15
                else:
                    max_tokens = 30
            if_get_result = False
            retry = 0
            while not if_get_result:
                try:
                    temp = args.temp 
                    max_tokens_step1 = max_tokens  
                    if_return_all = False
                    # if_self_consistency = args.self_consistency if args.reasoning_strategy != "plan_solve" else False
                    if args.self_consistency or args.self_verification:
                        if_return_all = True
                    if args.reasoning_strategy == "plan_solve":
                        if_return_all = False # step1 for "plan_solve" don't need return all chains
                        temp = 0.0
                        max_tokens_step1 = 256  

                    if load_pre_result and args.reasoning_strategy != "plan_solve": 
                        output_step1 = current_pred_all_chains
                        print("------ Directly load all chains from the pre-generated one.") 
                    elif load_pre_result and args.reasoning_strategy == "plan_solve":
                        output_step1 = current_pred_plan
                    else:  
                        output_step1 = get_answer_from_gpt(args, args.dataset, prompt, query, 
                                reasoning_strategy, args.learning_type, args.dialog_icl, if_return_all, 
                                engine=args.engine, max_tokens=max_tokens_step1, temperature=temp)
                    if args.reasoning_strategy != "plan_solve":
                        pred = output_step1 
                    else:
                        if args.self_consistency or args.self_verification:
                            if_return_all = True # step2 for "plan_solve" should follow if_return_all
                        
                        if load_pre_result :
                            pred = current_pred_all_chains
                            print("------ Directly load all chains from the pre-generated one.") 
                        else: 
                            # step2 for "plan_solve" is just like zero-shot CoT
                            pred = get_answer_from_gpt(args, args.dataset, prompt, output_step1, 
                                "complex_cot", "zero_shot", args.dialog_icl, if_return_all, 
                                engine=args.engine, max_tokens=max_tokens, temperature=args.temp)
                    if not args.self_verification: 
                        if args.self_consistency:
                            answer_list = []
                            dict_ans2pred = dict()
                            for i in range(len(pred)):
                                # input_ = prompt + query
                                _, pred_answer = get_rationale(pred[i])
                                pred_answer = extract_answer(pred_answer, args.dataset)
                                if dict_ans2pred.get(pred_answer) == None:
                                    dict_ans2pred[pred_answer] = []
                                dict_ans2pred[pred_answer].append(pred[i]) 
                                answer_list.append(pred_answer)
                            print("------ Voting on all the predicted chains.") 
                            collection_words = Counter(answer_list)
                            pred_answer = collection_words.most_common(1)[0][0]
                            result["sc_pred"] = pred_answer
                            result['ans_'+args.engine+"|"+prompt_name] = dict_ans2pred[pred_answer][0]
                            result["pred_chains_all"] = pred
                            result["pred_chains_sc"] = dict_ans2pred[pred_answer] 
                        else:
                            result['ans_'+args.engine+"|"+prompt_name] = pred
                    else:
                        if len(pred) == 0:
                            continue
                        # if len(pred) == 1:
                        #     pred.append(pred[0]) 

                        declarative = ""
                        answers_pred = []

                        index_temp = 0
                        for i in range(len(pred)):
                            p_it = pred[i]
                            _, pred_answer = get_rationale(p_it)
                            try:
                                p_item = extract_answer(pred_answer, args.dataset)
                            except:
                                pass
                            if not isinstance(p_item, str):
                                p_item = str(p_item)
                            if p_item != '' or p_item != 'None' and p_item not in answers_pred:
                            # if p_item != '':
                                index_temp = i
                                answers_pred.append(p_item)
                        if len(answers_pred) == 0:
                            result["sv_pred"] = ""
                            result['ans_'+args.engine+"|"+prompt_name] = pred[0]
                            result["pred_chains_all"] = pred
                        else:
                            if len(answers_pred) == 1:
                                scores = {0: 1}
                                pred_verifier = []
                                result["sv_pred"] = answers_pred[0]
                                result['ans_'+args.engine+"|"+prompt_name] = pred[index_temp]
                                result["pred_chains_all"] = pred
                                result["pred_chains_sv_index"] = index_temp
                            else:
                                scores = {i: 0 for i in range(len(answers_pred))}
                                pred_verifier = {i: [] for i in range(len(answers_pred))}
                                for A in range(len(answers_pred)):
                                    # round 1 for self_verification is like zero_shot cot, but the input is a little different, so we use the model as "self_verification" here 
                                    decl_query, answer_ver, declarative = question_turn_decalrative(args, args.dataset, query, answers_pred[A], answers_pred[0], declarative, prompt, "self_verification", "zero_shot", args.dialog_icl, False, args.engine) 
                                    for d in range(len(decl_query)):
                                        pred_v = get_answer_from_gpt(args, args.dataset, prompt, decl_query[d], "self_verification", "zero_shot", args.dialog_icl, True, args.engine, max_tokens=MAX_TOKEN_FOR_SELFVERIFY, temperature=0.0)
                                        answers_verifier = []
                                        for p in range(len(pred_v)):
                                            p_item_v = extract_answer(pred_v[p], args.dataset)
                                            try:
                                                answers_verifier.append(float(p_item_v))
                                            except:
                                                try:
                                                    answers_verifier.append(p_item_v)
                                                except:
                                                    pass
                                        try:
                                            score = sum(np.array(answers_verifier) == np.array(float(answer_ver[d])))
                                        except:
                                            try:
                                                score = sum(np.array(answers_verifier) == np.array(answer_ver[d]))
                                            except:
                                                score = 0
                                        if not isinstance(pred_v, str):
                                            pred_v = str(pred_v) 
                                        pred_verifier[A].append(pred_v)
                                        scores[A] += score
                            # verifier_is_ture = list(scores.values())
                            # if args.dataset in ("csqa"): # , "aqua"
                                # ground_ans = answer["options"][answer["answer"]] # answer["answer"]
                                # answers_is_ture = (np.array(answers) == np.array([ground_ans])).tolist()
                                # is_true = (np.array(answers) == np.array([ground_ans] * len(answers))).tolist()
                                # verifier_result = (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array(
                                #     [ground_ans])).sum().item()
                                # if_correct = (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array(
                                #     [ground_ans])).sum().item()

                            # else:
                                # ground_ans = answer["answer"]
                                # answers_is_ture = (np.array(answers) == np.array([ground_ans])).tolist()
                                # is_true = (np.array(answers) == np.array([ground_ans] * len(answers))).tolist()
                                # verifier_result = (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array(
                                #     [ground_ans])).sum().item()
                                # if_correct = (np.array([answers[np.argmax(np.array(verifier_is_ture)).item()]]) == np.array(
                                    # [ground_ans])).sum().item()
                                 
                            if len(answers_pred) > 1:
                                index_ver_result = max(scores, key=scores.get)
                                result["sv_pred"] = answers_pred[index_ver_result]
                                result['ans_'+args.engine+"|"+prompt_name] = pred[index_ver_result]
                                result["pred_chains_all"] = pred
                                result["pred_chains_sv_index"] = index_ver_result
                    if args.reasoning_strategy == "plan_solve":
                        result["pred_plan"] = output_step1
                    writer.write(result)
                    if_get_result = True 
                except openai.error.RateLimitError as e:
                    if e.user_message == 'You exceeded your current quota, please check your plan and billing details.':
                        raise e
                    elif retry < MAX_RETRY:
                        time.sleep(API_TIME_INTERVAL)
                        retry += 1
                    else:
                        print(e.user_message)
                        break
                except Exception as e:
                    decode_error_data = {
                        "dataset": args.dataset, "count": count - 1, 
                        'question': answer['question'], "prompt_name": prompt_name, 
                        "model": args.reasoning_strategy, "learning_type": args.learning_type,  
                        "sc": args.self_consistency, "sv": args.self_verification, "dialog": args.dialog_icl}
                    writer_error.write(decode_error_data)
                    logger.warning(
                        f"an error raised when predicting ({args.dataset} question count: {count-1}). "
                        f"ERROR: {getattr(e.__class__, '__name__')}:{str(e)}"
                    )
                    continue
                    # raise e
            if count % times_limit == 0:
                time.sleep(API_TIME_INTERVAL) 
        # the last element is the prompt
        writer.write(prompt)
        writer.close()
        print("save llm generated prediction results in", file_name)
        if args.learning_type == "zero_shot":
            break
    writer_error.close()

if __name__ == '__main__':
    main()