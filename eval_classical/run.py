#!/usr/bin/env python3

# Copyright (c) EMNLP 2023 Submission

import argparse
import glob
import os
from typing import Dict, List
import pandas
import json

import sys
sys.path.append("./")
sys.path.append("../")

from utils import (
    split_gsm8k_llm_generations_to_steps,  # to normalize same as in annotations
)
# from synthetic_roscoe import (
#     SyntheticChain,
# )
from constants import (
    STR_TRIGGER_RESTORE,
    Example,
    UseRef,
    BASELINE_SCORES,
    INPUT_DATA_SYNTHETIC,
    INPUT_DATA_FILES_HUMAN,
    INPUT_DATA_HUMAN,
    DEFAULT_PROMPT_PATH,
    DEFAULT_INPUT_RES_PATH,
    DATASETS,
    PERTURBATIONS,
    PROMPT_NAME_ZSL,
)
from eval_classical.scores import (
    SCORES_TO_CLASS,
)

from nltk import sent_tokenize


################# Default settings
OUT_PATH = "./scores/classical/"
SYNTHETIC_PATH = "./data/generated/synthetic_50%/"


class BaselineDataLoader:
    @classmethod
    def load_human(cls, dataset, filename, test_model) -> List[Example]:
        # with open(INPUT_DATA_FILES_HUMAN[dataset]) as f:
        with open(filename) as f: 
            raw = f.readlines()

        def parse_to_example(line, test_model):
            def normalize_SyntheticChain(s):
                return " ".join(sent_tokenize(s))

            def normalize_ParsedChain(s):
                return " ".join(s)

            jline = json.loads(line)
            r_chain = None
            # following kept in sync with `real_scorer.py`
            # We use the normalize functions as described above since baselines do not operate on steps, but we need to take
            # into account any characters that get changed when `real_scorer.py` parses things into steps.
            # Conflusingly, `real_scorer.py` uses "SyntheticChain" when it's real data...
            for key in jline.keys():
                if "|" in key:
                    test_model = key
                    break 
            h_chain = normalize_SyntheticChain(jline[test_model])
            context = normalize_SyntheticChain(
                jline["premise"] + " " + jline["hypothesis"]
            )
            if "gsm8k" in dataset:
                context = normalize_ParsedChain(sent_tokenize(jline["premise"]))
                h_chain = normalize_ParsedChain(
                    split_gsm8k_llm_generations_to_steps(jline[test_model])
                )
                r_chain = normalize_ParsedChain(
                    jline["hypothesis"]
                    .split("IGNORE THIS. Ground truth here for reference. ")[1]
                    .split('\n')
                )
            if "esnli" in dataset:
                r_chain = normalize_SyntheticChain(
                    jline["explanation_1"]
                    + " "
                    + jline["explanation_2"]
                    + " "
                    + jline["explanation_3"]
                )
            return Example(context, h_chain, r_chain)

        return [parse_to_example(x, test_model) for x in raw]

    # @classmethod
    # def load_synthetic_datas(cls, dataset) -> Dict[str, List[Example]]:
    #     dataset_path_name = dataset
    #     if dataset == "math":
    #         dataset_path_name = "math_dataset"

    #     def parse_to_example(
    #         line,
    #     ):
    #         jline = json.loads(line)
    #         # This should be aligned with what is in synthetic_scorer.py
    #         h_chain = SyntheticChain(line=jline["dialog"][0][0]["steps"])
    #         r_chain = SyntheticChain(line=jline["dialog"][0][0]["original_steps"])
    #         context = SyntheticChain(line=jline["dialog"][0][0]["question"].split(". "))
    #         return Example(
    #             " ".join(context.chain),
    #             " ".join(h_chain.chain),
    #             " ".join(r_chain.chain),
    #         )

    #     result = {}
    #     for f_name in glob.glob(
    #         SYNTHETIC_PATH + f"{dataset_path_name}_synthetic/50*test.jsonl"
    #     ):
    #         seen_ps = 0
    #         for p in PERTURBATIONS:
    #             if p in f_name:
    #                 seen_ps += 1
    #         if seen_ps != 1:
    #             continue
    #         print("Synthetic data filename:", f_name)
    #         with open(f_name) as f:
    #             raw = f.readlines()
    #         result[
    #             dataset + "_" + f_name.split("/")[-1].replace(".jsonl", "_scores")
    #         ] = [parse_to_example(x, test_model) for x in raw]
    #     return result

    @classmethod
    def load_data(cls, dataset, filename, test_model) -> Dict[str, List[Example]]:
        """
        Given a short name for the dataset, return a map that has descriptor for what
        the data from the dataset is, plus the examples (useful since synthetic data has
        different folders for Perturbations)
        """
        if dataset == "test":
            return {"test": [Example(x, x, x + 2) for x in range(8)]}
        if dataset in INPUT_DATA_HUMAN:
            return {dataset: cls.load_human(dataset, filename, test_model)}
        assert dataset in INPUT_DATA_SYNTHETIC
        return cls.load_synthetic_datas(dataset)


def main(args):
    def make_path(p):
        import pathlib  # make sure out path exists

        pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)

    def save_scores_map(path, metric_to_score):
        print('saving scores to', path)
        datas = pandas.DataFrame.from_dict(metric_to_score)
        ordered = [x for x in BASELINE_SCORES if x in list(metric_to_score.keys())]
        datas = datas.reindex(columns=ordered)
        make_path(path)
        with open(path, "w+") as f:
            datas.to_csv(f)

    print("========================================= Done with module includes")
    datasets = args.datasets
    if "test" in datasets:
        datasets = ["test"]
    elif "human" in datasets:
        datasets = INPUT_DATA_FILES_HUMAN
    elif "synthetic" in datasets:
        datasets = INPUT_DATA_SYNTHETIC
    print("Getting scores for datasets", datasets)
    print("On scores", args.score)

    scorers_classes = set([SCORES_TO_CLASS[x] for x in args.score])
    scorers = [x() for x in scorers_classes]
    score_path_name = "-".join(args.score)

    use_ref = [
        x for x in UseRef if x.value in args.use_ref
    ]  # string -> enum conversion

    print("Using reference types", use_ref)

    print(
        "========================================= Done loading scorers; iterating through datasets"
    )

    # for dataset in datasets:
    #     if args.dataset is not None and args.dataset != dataset:
    #         continue
    #     print("Scoring dataset", dataset)
    #     for root, _dirnames, filenames in os.walk(DEFAULT_PROMPT_PATH + dataset):
    #         for filename in filenames:
    #             if ".txt" not in filename:
    #                 continue 
    #             prompt_name = filename.split(".txt")[0]
    #             if args.prompt_name is not None and prompt_name != args.prompt_name and args.prompt_name != "NoNeed":
    #                 continue 
    #             if args.prompt_name is not None:
    #                 prompt_name = args.prompt_name 
    #             if args.learning_type == "zero_shot":
    #                 prompt_name = PROMPT_NAME_ZSL
    #             test_model = args.engine + "|" + prompt_name
    # print("Scoring dataset", dataset)

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
             
            if not os.path.exists(DEFAULT_INPUT_RES_PATH + filename):
                continue

            test_model = engine + "|" + prompt_name        
            datas = BaselineDataLoader.load_data(dataset, DEFAULT_INPUT_RES_PATH + filename, test_model) 
            print("Data loaded, about to start scoring", test_model)
            print(f"Evaluating {os.path.join(root, filename)}") 
            for out_name, examples in datas.items():
                with_ref_scores = {}
                no_ref_scores = {}
                for scorer in scorers:
                    scores_raw = scorer.score_data(examples, use_ref)
                    if UseRef.YES in scores_raw:
                        for metric, scores in scores_raw[UseRef.YES].items():
                            if len(scores) > 0:
                                with_ref_scores[metric] = scores
                            else:
                                print("with ref length of 0:", dataset, out_name, metric)
                    if UseRef.NO in scores_raw:
                        for metric, scores in scores_raw[UseRef.NO].items():
                            if len(scores) > 0:
                                no_ref_scores[metric] = scores
                            else:
                                print("no ref length of 0:", dataset, out_name, metric)
                # Now we've got all our scores, save them
                out_for_dataset = os.path.join(args.out_dir, dataset, score_path_name)
                make_path(out_for_dataset)

                # out_name += "_" + prompt_name + "_" + args.engine + "_sample" + str(args.num_test)
                out_name = f"{filename.split('|' + STR_TRIGGER_RESTORE + '.jsonl')[0]}" 
                # "_{}|engine{}|samp{}|{}|{}|sc-{}|sv-{}|dial-{}|{}".format(
                #     prompt_name, engine, num_test, learning_type, 
                #     reasoning_strategy, self_consistency, self_verification, dialog_icl, suffix_ans)
                if len(with_ref_scores) > 0:
                    out_p = os.path.join(out_for_dataset, f"{out_name}-with_ref.csv") 
                    if os.path.exists(out_p) and not args.overwrite_output_dir:
                        print(f"Score file {out_p} already exists. Skipping.")
                    else:
                        save_scores_map(
                            out_p,
                            with_ref_scores,
                        )
                if len(no_ref_scores) > 0:
                    out_p = os.path.join(out_for_dataset, f"{out_name}-no_ref.csv")
                    if os.path.exists(out_p) and not args.overwrite_output_dir:
                        print(f"Score file {out_p} already exists. Skipping.")
                    else:
                        save_scores_map(
                            out_p,
                            no_ref_scores,
                        )
            # if args.learning_type == "zero_shot":
            #     break
            if not args.batch_action:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        type=str,
        choices=DATASETS + ["test", "human", "synthetic"],
        nargs="+",
        default=DATASETS,
        help='name of datasets to score. If "test", "human", or "synthetic" used, will replace entire input with those.',
    )
    parser.add_argument(
        '--score',
        type=str,
        choices=list(SCORES_TO_CLASS.keys()),
        nargs="+",
        default=list(SCORES_TO_CLASS.keys()),
        help='name of scores to gen',
    )
    parser.add_argument(
        '--use_ref',
        type=str,
        choices=[x.value for x in UseRef],
        nargs="+",
        default=UseRef.NO.value,
        help='do we want to generate reference-based or reference-free scores',
    )
    parser.add_argument(
        '--out_dir', type=str, help='output directory', default=OUT_PATH
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
    main(args)
