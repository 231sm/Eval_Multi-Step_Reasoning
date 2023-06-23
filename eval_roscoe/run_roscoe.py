#!/usr/bin/env python3

# Copyright (c) EMNLP 2023 Submission

"""
Evaluate dataset of generated chains-of-resoning.

Usage:
python roscoe.py
"""

import json
import os
import pandas
from typing import List

import sys
sys.path.append("./")
sys.path.append("../")

from nltk.tokenize import sent_tokenize

from eval_roscoe.score import (
    SEQ_EMB_MODEL_TYPES,
    Chain,
    Evaluator,
    REASONING_SCORES,
    UNSUPERVISED_SCORES,
    SENT_TRANS,
)
from utils import (
    print_and_reset_max_gpu_memory,
    save_scores,
    split_gsm8k_llm_generations_to_steps,
)

import argparse

from constants import (
    DEFAULT_INPUT_ANN_PATH, 
    DEFAULT_INPUT_GEN_PATH, 
    DEFAULT_INPUT_RES_PATH, 
    DEFAULT_PROMPT_PATH, 
    DEFAULT_OUTPUT_PATH,
    DATASETS,
    PROMPT_NAME_ZSL,
    STR_TRIGGER_RESTORE
)


class ReasoningSteps(Chain):
    def __init__(self, line: str, type="regular") -> None:
        self.chain = self.parse_chain(line, type=type)

    def parse_chain(self, chain: str, type: str) -> List[str]:
        """
        Change formatting.

        Returns list of steps in reasoning chain.
        """
        # if type == "gsm8k_ref":
        #     return chain.split("IGNORE THIS. Ground truth here for reference. ")[
        #         1
        #     ].split('\n')
        if type == "gsm8k_ref":
            return chain.split("IGNORE THIS. Ground truth here for reference. ")[1]
        elif type == "gsm8k_hypo":
            return split_gsm8k_llm_generations_to_steps(reasoning=chain)
        elif type == "regular":
            return sent_tokenize(chain)
        else:
            raise NotImplementedError(f"{type} chain type is not supported")


class ReasoningEvaluator(Evaluator):
    def __init__(
        self,
        model_type: str,
        transformer_model: str,
        discourse_batch: int,
        coherence_batch: int,
        **kwargs,
    ) -> None:
        super().__init__(
            hypos=[],
            context=[],
            references=[],
            model_type=model_type,
            transformer_model=transformer_model,
            discourse_batch=discourse_batch,
            coherence_batch=coherence_batch,
            **kwargs,
        )

    def update_evaluator(self, in_file: str, test_model):
        hypothesises = []
        contexts = []
        refs = []
        with open(in_file) as _f:
            for line in _f:
                jline = json.loads(line)
                h_chain = ReasoningSteps(line=jline[test_model])
                context = ReasoningSteps(
                    line=jline["premise"] + " " + jline["hypothesis"]
                )
                if "gsm8k" in in_file and "_gsm8k" not in in_file:
                    context = ReasoningSteps(line=jline["premise"])
                    h_chain = ReasoningSteps(line=jline[test_model], type="gsm8k_hypo")
                    r_chain = ReasoningSteps(line=jline["hypothesis"], type="gsm8k_ref")
                    refs.append(r_chain)
                hypothesises.append(h_chain)
                contexts.append(context)
                if "esnli" in in_file:
                    r_chain = ReasoningSteps(
                        line=jline["explanation_1"]
                        + " "
                        + jline["explanation_2"]
                        + " "
                        + jline["explanation_3"]
                    )
                    refs.append(r_chain)
        super().set_hypos(hypothesises)
        super().set_context(contexts)
        super().set_references(refs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        '-p',
        type=str,
        required=False,
        default=DEFAULT_INPUT_RES_PATH,
        help='Path to files with predictions in restored files',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs="+",
        default=DATASETS,
        help='Which datasets to score.',
    )
    parser.add_argument(
        '--suffix',
        '-s',
        type=str,
        required=False,
        default="json",
        help='File suffix to match',
    )
    parser.add_argument(
        '--seq_emb_model_type', # model_type 
        '-t',
        type=str,
        required=False,
        default=SENT_TRANS,
        choices=SEQ_EMB_MODEL_TYPES,
        help='Model type for embedding sequences.',
    )
    parser.add_argument(
        '--trans_model_name', # model_name 
        '-m',
        type=str,
        required=False,
        default="all-MiniLM-L6-v2", # all-mpnet-base-v2
        help='Transformer model name for embeddings. Must be compatible with model_type',
    )
    parser.add_argument(
        '--ppl_model_name',
        type=str,
        required=False,
        default="gpt2-large", # TurkuNLP/gpt3-finnish-small 
        help='Transformer HuggingFace model name for calculating perplexity-based metrics.',
    )
    parser.add_argument(
        '--discourse_batch',
        '-db',
        type=int,
        required=False,
        default=64,
        help='Batch size for discourse calculation',
    )
    parser.add_argument(
        '--coherence_batch',
        '-cb',
        type=int,
        required=False,
        default=16,
        help='Batch size for coherence calculation',
    )
    parser.add_argument(
        '--scores',
        type=str,
        nargs="*",
        default=REASONING_SCORES,
        choices=REASONING_SCORES,
        help=(
            'Scores to calculate. If the data is incompatible with a specified score '
            '(e.g. no reference is available) the score will be ignored.'
        ),
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Where to save the scores.',
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
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
    parser.add_argument("--dataset", default=None, type=str, required=False, help="dataset for experiments")  
    parser.add_argument("--prompt_name", default=None, type=str, required=False, help="type for prompt")
    parser.add_argument("--engine", default="gpt-3.5-turbo", type=str, help="engine")
    parser.add_argument("--num_test", default=-1, type=int, help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float, help="temperature for generation")
    parser.add_argument("--max_tokens", default=256, type=int, help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str, help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix_ans", default="", type=str, help="")
    parser.add_argument("--batch_action", action="store_true", help="Batch restore all prediction files that satisfy some input args.") 

    args = parser.parse_args()
    print(args)

    def make_path(p):
        import pathlib  # make sure out path exists

        pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)

    def save_scores_map(metric_to_score, path):
        print('saving scores to', path)
        datas = pandas.DataFrame.from_dict(metric_to_score)
        ordered = [x for x in list(metric_to_score.keys())]
        datas = datas.reindex(columns=ordered)
        make_path(path)
        with open(path, "w+") as f:
            datas.to_csv(f)

    evaluator = ReasoningEvaluator(
        score_types=args.scores,
        model_type=args.seq_emb_model_type,
        transformer_model=args.trans_model_name,
        ppl_model=args.ppl_model_name,
        discourse_batch=args.discourse_batch,
        coherence_batch=args.coherence_batch,
    )
 
    datasets = DATASETS
    if args.datasets:
        datasets = args.datasets
    dict_dataset2prompt = dict() 
    for dataset in datasets:
        dict_dataset2prompt[dataset] = []
        for root, _dirnames, filenames in os.walk(DEFAULT_PROMPT_PATH + dataset):
            for filename in filenames:
                if ".txt" not in filename:
                    continue 
                prompt_name = filename.split(".txt")[0]
                dict_dataset2prompt[dataset].append(prompt_name)
        dict_dataset2prompt[dataset].append(PROMPT_NAME_ZSL)
    
    for root, _dirnames, filenames in os.walk(DEFAULT_INPUT_RES_PATH):
        for filename in filenames:
            if ".jsonl" not in filename or "|" not in filename or STR_TRIGGER_RESTORE not in filename:
                continue
            para_list = filename.split("|")
            if len(para_list) != 10:
                continue

            if not args.batch_action:
                dataset, prompt_name, engine = args.dataset, args.prompt_name, args.engine, 
                num_test, learning_type, reasoning_strategy = args.num_test, args.learning_type, args.reasoning_strategy 
                self_consistency, self_verification = args.self_consistency, args.self_verification 
                dialog_icl, suffix_ans = args.dialog_icl, args.suffix_ans 
            else:
                dataset = para_list[0].split("_")[0] if args.dataset is None else args.dataset
                prompt_name = para_list[0].split(args.dataset)[-1][1:] if args.prompt_name is None else args.prompt_name 
                engine = para_list[1].split("engine")[-1] if args.engine == "gpt-3.5-turbo" else args.engine 
                num_test = eval(para_list[2].split("samp")[-1]) if args.num_test == -1 else args.num_test 
                learning_type = para_list[3] if args.learning_type == "few_shot" else args.learning_type
                reasoning_strategy = para_list[4] if args.reasoning_strategy == "complex_cot" else args.reasoning_strategy 
                self_consistency = eval(para_list[5].split("sc-")[-1]) if not args.self_consistency else args.self_consistency
                self_verification = eval(para_list[6].split("sv-")[-1]) if not args.self_verification else args.self_verification 
                dialog_icl = eval(para_list[7].split("dial-")[-1]) if not args.dialog_icl else args.dialog_icl 
                suffix_ans = para_list[8] if args.suffix_ans  == "" else args.suffix_ans 

            filename = "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}|{}.jsonl".format(
                dataset, prompt_name, engine, num_test, learning_type, 
                reasoning_strategy, self_consistency, self_verification, dialog_icl, suffix_ans, STR_TRIGGER_RESTORE)  
             
            if not os.path.exists(args.dataset_path + filename):
                continue 

    # for root, _dirnames, filenames in os.walk(args.dataset_path):
    #     for filename in filenames:
    #         if args.dataset is not None and args.dataset not in filename: 
    #             continue
    #         if args.prompt_name is not None and args.dataset is not None and "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}".format(
    #                 args.dataset, args.prompt_name, args.engine, args.num_test, args.learning_type, 
    #                 args.reasoning_strategy, args.self_consistency, args.self_verification, args.dialog_icl, args.suffix_ans) not in filename:
    #             continue
            if args.suffix not in filename or ".swp" in filename or "|"+STR_TRIGGER_RESTORE not in filename:
                continue
            if not any(filename.startswith(d) for d in args.datasets):
                print(f"Skipping due to --datasets filter: {filename}")
                continue
            out_p = (
                args.output_directory 
                + "roscoe" 
                # + args.trans_model_name.split('/')[-1]
                # + args.engine + "_" + prompt_name + "_" + args.num_test 
                + f"/{filename.split('|' + STR_TRIGGER_RESTORE + '.jsonl')[0]}.csv"
            )
            if os.path.exists(out_p) and not args.overwrite_output_dir:
                print(f"Score file for {filename} already exists. Skipping.")
            else:
                # if "|engine"+args.engine+"|" in filename:
                #     for dataset in dict_dataset2prompt.keys():
                #         if dataset in filename:
                #             list_prompt = dict_dataset2prompt[dataset]
                #             # if not args.dialog_icl:
                #             #     list_prompt.append(PROMPT_NAME_ZSL) 
                #             for prompt_name in list_prompt: 
                #                 # if prompt_name + "|engine" + args.engine not in filename:
                #                 if "{}_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}".format(
                #                     dataset, prompt_name, args.engine, args.num_test, args.learning_type, 
                #                     args.reasoning_strategy, args.self_consistency, args.self_verification, args.dialog_icl, args.suffix_ans) not in filename:
                #                     continue
                test_model = engine + "|" + prompt_name
                print(f"Evaluating {os.path.join(root, filename)}") 
                evaluator.update_evaluator(os.path.join(root, filename), test_model)
                score_types = (
                    REASONING_SCORES
                    if "esnli" in filename or "gsm8k" in filename
                    else UNSUPERVISED_SCORES
                )
                score_types = [st for st in score_types if st in args.scores]
                scores = evaluator.evaluate(score_types=score_types)
                # list_prompt.remove(prompt_name) 
                # save_scores(scores, out_p)
                save_scores_map(scores, out_p)
                print_and_reset_max_gpu_memory()
            if not args.batch_action:
                break
