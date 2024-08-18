#!/usr/bin/env python3

# Copyright (c) # Copyright (c) ACL 2024, Natural Language Reasoning and Structured Explanations Workshop

"""
Restore data with reasoning chains.
"""

from constants import (
    DEFAULT_INPUT_ANN_PATH,
    DEFAULT_INPUT_GEN_PATH,
    DEFAULT_INPUT_RES_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_OUTPUT_PATH,
    DATASETS,
    STR_GEN_STOP,
    DICT_STR_SPLIT_RATIONALE,
    PROMPT_NAME_ZSL,
    STR_TRIGGER_RESTORE
)
from eval_utils import (
    util_gsm8k,
    util_svamp,
    util_multiarith,
    util_mathqa,
    util_csqa,
    util_strategyqa,
)
import argparse
import xmltodict
import re
import os
import csv
import json
import sys

from utils import get_rationale, read_jsonl
import numpy as np
sys.path.append("./")
sys.path.append("../")


def write_to_file(list_of_json_dict, output_path):
    with open(output_path, 'w') as outfile:
        for line in list_of_json_dict:
            json.dump(line, outfile)
            outfile.write('\n')


def parse_gsm8k(filename, savefile, test_model, num_test):
    all_data, question2id, list_questionid = util_gsm8k(filename, num_test)

    structs = []
    for question_id in list_questionid:
        data = all_data[question_id]
        blob = {}
        blob["premise"] = data["prediction"]["query"]
        blob["hypothesis"] = (
            "IGNORE THIS. Ground truth here for reference. " +
            data["ground_truth"]
            # "IGNORE THIS. Ground truth here for reference. " + data["prediction"]["example_result"]
            # data["prediction"]["rationale"]
        )
        if "ans_"+test_model in data["prediction"].keys():
            blob[test_model] = data["prediction"]["ans_"+test_model]
        else:
            blob[test_model] = data["prediction"]["example_result"]
        blob["answer"] = "yes" if data["prediction"]["is_correct"] else "no"
        blob["key"] = question2id[data["question"]]
        blob["eval_others"] = data["eval_others"]
        structs.append(blob)

    write_to_file(structs, savefile)


def parse_svamp(filename, savefile, test_model, num_test):
    result = util_svamp(filename, num_test)
    structs = []
    for data in result:
        blob = {}
        blob["premise"] = data["query"]
        blob["hypothesis"] = data["rationale_example"].strip()  # data["equation"]
        if "ans_"+test_model in data.keys():
            blob[test_model] = data["ans_"+test_model]
        blob["answer"] = "yes" if data["prediction"]["is_correct"] else "no"
        blob["key"] = data["id"]
        blob["eval_others"] = data["prediction"]["eval_others"]
        structs.append(blob)
    write_to_file(structs, savefile)


def parse_multiarith(filename, savefile, test_model, num_test):
    result = util_multiarith(filename, num_test)
    structs = []
    for data in result:
        blob = {}
        blob["premise"] = data["query"]
        # data["equation"][0]
        blob["hypothesis"] = data["rationale_example"].strip()
        if "ans_"+test_model in data.keys():
            blob[test_model] = data["ans_"+test_model]
        blob["answer"] = "yes" if data["prediction"]["is_correct"] else "no"
        blob["key"] = data["id"]
        blob["eval_others"] = data["prediction"]["eval_others"]
        structs.append(blob)
    write_to_file(structs, savefile)


def parse_mathqa(filename, savefile, test_model, num_test):
    result = util_mathqa(filename, num_test)
    structs = []
    for data in result:
        blob = {}
        blob["premise"] = data["query"]
        # data["rationale"]
        blob["hypothesis"] = data["rationale_example"].strip()
        if "ans_"+test_model in data.keys():
            blob[test_model] = data["ans_"+test_model]
        blob["answer"] = "yes" if data["prediction"]["is_correct"] else "no"
        # blob["answer_equation"] = "yes" if data["prediction"]["is_correct_equation"] else "no"
        blob["key"] = data["id"]
        blob["eval_others"] = data["prediction"]["eval_others"]
        structs.append(blob)
    write_to_file(structs, savefile)


def parse_csqa(filename, savefile, test_model, num_test):
    result = util_csqa(filename, num_test)
    structs = []
    for data in result:
        blob = {}
        blob["premise"] = data["query"]
        blob["hypothesis"] = data["rationale_example"].strip()  # data["answer"]
        if "ans_"+test_model in data.keys():
            blob[test_model] = data["ans_"+test_model]
        blob["answer"] = "yes" if data["prediction"]["is_correct"] else "no"
        blob["key"] = data["id"]
        blob["eval_others"] = data["prediction"]["eval_others"]
        structs.append(blob)
    write_to_file(structs, savefile)


def parse_strategyqa(filename, savefile, test_model, num_test):
    result = util_strategyqa(filename, num_test)
    structs = []
    for data in result:
        blob = {}
        blob["premise"] = data["query"]
        # data["rationale"]
        blob["hypothesis"] = data["rationale_example"].strip()
        if "ans_"+test_model in data.keys():
            blob[test_model] = data["ans_"+test_model]
        blob["answer"] = "yes" if data["prediction"]["is_correct"] else "no"
        blob["key"] = data["id"]
        blob["eval_others"] = data["prediction"]["eval_others"]
        structs.append(blob)
    write_to_file(structs, savefile)


def main(args):
    # The line that constrcuts the context is:
    # context = "Premise: " + struct["premise"] + "\nHypothesis: " + struct["hypothesis"] + "\nExplanation: "
    # The "Explanation" is followed by the LLM generations
    path_to_gen_data = args.dataset_path
    output_path = args.out_dir

    for root, _dirnames, filenames in os.walk(path_to_gen_data):
        for filename in filenames:
            if ".jsonl" not in filename or "|" not in filename:
                continue
            para_list = filename.split("|")
            if len(para_list) != 9:
                continue
            if not args.batch_action:
                dataset, prompt_name, engine = args.dataset, args.prompt_name, args.engine,
                num_test, learning_type, reasoning_strategy = args.num_test, args.learning_type, args.reasoning_strategy
                self_consistency, self_verification = args.self_consistency, args.self_verification
                dialog_icl, suffix_ans = args.dialog_icl, args.suffix_ans
            else:
                dataset = para_list[0].split(
                    "_")[0] if args.dataset is None else args.dataset
                prompt_name = para_list[0].split(
                    args.dataset)[-1][1:] if args.prompt_name is None else args.prompt_name
                engine = para_list[1].split(
                    "engine")[-1] if args.engine == "gpt-3.5-turbo" else args.engine
                num_test = eval(para_list[2].split(
                    "samp")[-1]) if args.num_test == -1 else args.num_test
                learning_type = para_list[3] if args.learning_type == "few_shot" else args.learning_type
                reasoning_strategy = para_list[4] if args.reasoning_strategy == "complex_cot" else args.reasoning_strategy
                self_consistency = eval(para_list[5].split(
                    "sc-")[-1]) if not args.self_consistency else args.self_consistency
                self_verification = eval(para_list[6].split(
                    "sv-")[-1]) if not args.self_verification else args.self_verification
                dialog_icl = eval(para_list[7].split(
                    "dial-")[-1]) if not args.dialog_icl else args.dialog_icl
                suffix_ans = para_list[-1].split(
                    ".jsonl")[0] if para_list[-1] != ".jsonl" else args.suffix_ans

            filename = "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}.jsonl".format(
                dataset, prompt_name, engine, num_test, learning_type,
                reasoning_strategy, self_consistency, self_verification, dialog_icl, suffix_ans)

            if not os.path.exists(path_to_gen_data + filename):
                continue

            input_file = path_to_gen_data + filename

            # save_file = os.path.join(output_path, dataset + "_" + prompt_name + "_" + args.engine + "_sample" + str(args.num_test) + "|restore.jsonl")
            save_file = output_path + \
                filename.split(".jsonl")[0] + "|" + \
                STR_TRIGGER_RESTORE + ".jsonl"
            # "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}|{}.jsonl".format(
            #     dataset, prompt_name, args.engine, args.num_test, args.learning_type,
            #     args.reasoning_strategy, args.self_consistency, args.self_verification, args.dialog_icl, args.suffix_ans, STR_TRIGGER_RESTORE)

            test_model = engine + "|" + prompt_name
            if args.dataset == 'gsm8k':
                parse_gsm8k(input_file, save_file, test_model, num_test)
                print(f"Saved GSM8K dataset in {save_file}")
            elif args.dataset == 'svamp':
                parse_svamp(input_file, save_file, test_model, num_test)
                print(f"Saved SVAMP dataset in {save_file}")
            elif args.dataset == 'multiarith':
                parse_multiarith(input_file, save_file, test_model, num_test)
                print(f"Saved MultiArith dataset in {save_file}")
            elif args.dataset == 'mathqa':
                parse_mathqa(input_file, save_file, test_model, num_test)
                print(f"Saved MathQA dataset in {save_file}")
            elif args.dataset == 'csqa':
                parse_csqa(input_file, save_file, test_model, num_test)
                print(f"Saved CSQA dataset in {save_file}")
            elif args.dataset == 'strategyqa':
                parse_strategyqa(input_file, save_file, test_model, num_test)
                print(f"Saved StrategyQA dataset in {save_file}")
            else:
                raise NotImplementedError(f"Dataset {dataset} not recognized")

            # if prompt_name == PROMPT_NAME_ZSL:
            #     break
            # list_step_number = []
            # data = read_jsonl(input_file)
            # for one_dict in data[:-1]:
            #     if self_consistency or self_verification:
            #         len_temp = 0
            #         for chain in one_dict['pred_chains_all']:
            #             chain, _ = get_rationale(chain)
            #             len_temp += len(chain.split("\n"))
            #         len_temp /= 10
            #     else:
            #         # if 'ans_' + test_model in one_dict.keys():
            #         chain = one_dict['ans_' + test_model]
            #         chain, _ = get_rationale(chain)
            #         len_temp = len(chain.split("\n"))
            #     list_step_number.append(len_temp)
            # print(np.mean(list_step_number), "/////////", save_file)

            if not args.batch_action:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        '-d',
        type=str,
        required=False,
        default=DEFAULT_INPUT_GEN_PATH,
        help='Path to files with questions',
    )
    parser.add_argument(
        '--datasets',
        '-s',
        type=str,
        default=DATASETS,
        choices=DATASETS,
        nargs="*",
        required=False,
        help='Dataset name',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=False,
        default=DEFAULT_INPUT_RES_PATH,
        help='Path where mixes will be saved. Path to files with restored generations',
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
    parser.add_argument("--prompt_name", default=None,
                        type=str, required=False, help="type for prompt")
    parser.add_argument("--dataset", default=None, type=str,
                        required=False, help="dataset for experiments")
    parser.add_argument("--engine", default="gpt-3.5-turbo",
                        type=str, help="engine")
    parser.add_argument("--num_test", default=-1, type=int,
                        help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float,
                        help="temperature for generation")
    parser.add_argument("--max_tokens", default=256, type=int,
                        help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str,
                        help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix_ans", default="", type=str, help="")
    parser.add_argument("--apikey_file", default="./api_key.txt",
                        type=str, help="file path for api key.")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached generated rationale files in jsonl format")
    parser.add_argument("--overwrite_prediction", action="store_true",
                        help="Overwrite the LLM-generated prediction result files in jsonl format")
    parser.add_argument("--batch_action", action="store_true",
                        help="Batch restore all prediction files that satisfy some input args.")

    args = parser.parse_args()

    main(args)
