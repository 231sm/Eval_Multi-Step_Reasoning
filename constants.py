#!/usr/bin/env python3

# Copyright (c) EMNLP 2023 Submission

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class UseRef(Enum):
    YES = "with_ref"
    NO = "no_ref"


@dataclass
class Example:
    context: str
    hypo: str
    ref: Optional[str] = None


@dataclass
class ScoreMe:
    hypos: List[str]
    context_ref: List[str]
    use_ref: UseRef

    def __init__(self, exs: List[Example], use_ref: UseRef):
        self.use_ref = use_ref
        if self.use_ref is UseRef.YES:
            self.context_ref = [x.ref for x in exs]
        else:
            self.context_ref = [x.context for x in exs]
        assert len(self.context_ref) > 0 and self.context_ref[0] is not None
        self.hypos = [x.hypo for x in exs]


MAX_RETRY = 3
API_TIME_INTERVAL = 2.0

TEMP_FOR_MULTI_CHAINS = 0.7
N_FOR_MULTI_CHAINS = 10
MAX_TOKEN_FOR_SELFVERIFY = 30

STR_GEN_STOP = "\n\n" # "\n"
DICT_STR_SPLIT_RATIONALE = {"\n\n": "\n", "\n": ". "}

STR_TRIGGER_RESTORE = "restore"

COT_TRIGGER = "Let's think step by step." 

PROMPT_NAME_ZSL = "NoNeed"


############### SCORES
ROUGE_1 = "rouge_1"
ROUGE_2 = "rouge_2"
ROUGE_L = "rouge_l"
BLEURT = "bleurt"
BERTSCORE_F = "bertScore_f"
BARTSCORE_F = "bartScore_f"
BARTSCORE_CNN_F = "bartScore_cnn_f"
BARTSCORE_CNN_PARA_F = "bartscore_cnn_para_f"
BARTSCORE_FINETUNED_F = "bartscore_finetuned_f"
PRISM_AVG = "prism_avg"  # note: we're actually using PRISM where it changes underlying behavior depending on references or not
CTC_RELEVANCE_SUMMARY = "ctc_relevance_summary"
CTC_CONSISTENCY_SUMMARY = "ctc_consistency_summary"

BASELINE_SCORES = [  # Use this to hide metrics we don't want to use anymore
    ROUGE_1,
    ROUGE_2,
    ROUGE_L,
    BLEURT,
    BERTSCORE_F,
    BARTSCORE_F,
    #    BARTSCORE_CNN_F,
    BARTSCORE_CNN_PARA_F,
    BARTSCORE_FINETUNED_F,
    PRISM_AVG,
    CTC_RELEVANCE_SUMMARY,
    CTC_CONSISTENCY_SUMMARY,
]

################ Paths
DEFAULT_INPUT_ANN_PATH = f"./data/annotated/"
DEFAULT_INPUT_GEN_PATH = f"./data/generated/"
DEFAULT_INPUT_RES_PATH = f"./data/restored/"
DEFAULT_PROMPT_PATH = f"./prompts/" 
DEFAULT_OUTPUT_PATH = f"./scores/"
DEFAULT_LOG_PATH = f"./log/"


DEFAULT_INPUT_PATH = f"./data/restored"
################ Datasets
INPUT_DATA_FILES_HUMAN = {
    # "drop": f"{DEFAULT_INPUT_PATH}/drop.json",
    # "esnli": f"{DEFAULT_INPUT_PATH}/esnli.json",
    # "cosmos": f"{DEFAULT_INPUT_PATH}/cosmos.json",
    # "semeval": f"{DEFAULT_INPUT_PATH}/semevalcommonsense.json",
    "gsm8k": f"{DEFAULT_INPUT_PATH}/gsm8k.jsonl",
    "svamp": f"{DEFAULT_INPUT_PATH}/svamp.jsonl",
    "multiarith": f"{DEFAULT_INPUT_PATH}/multiarith.jsonl",
    "mathqa": f"{DEFAULT_INPUT_PATH}/mathqa.jsonl",
    "csqa": f"{DEFAULT_INPUT_PATH}/csqa.jsonl",
    # "strategyqa": f"{DEFAULT_INPUT_PATH}/strategyqa.jsonl", 
}
INPUT_DATA_HUMAN = list(INPUT_DATA_FILES_HUMAN.keys())

INPUT_DATA_SYNTHETIC = [
    "aqua",
    "asdiv",
    "entailment_bank",
    "eqasc",
    "math",
    "proofwriter",
    # "strategy_qa", # used for train + valid only
]

# DATASETS = INPUT_DATA_HUMAN + INPUT_DATA_SYNTHETIC
DATASETS = INPUT_DATA_HUMAN

############### Perturbations

PERTURBATIONS = [
    "ShuffleSteps",
    "DuplicateOneStep",
    "RemoveOneStep",
    "SwapOneStep",
    "ExtrinsicHallucinatedStep",
    "ParaphraseSteps",
    "GrammaticalErrorStep",
    "NegateStep",
    "SemanticChangeStep",
    "ShuffleNumbers",
    "ShuffleOperations",
    "RandomNumber",
    "RandomOperation",
]
