#!/bin/bash
# Usage:
# sh generate_answers.sh

ENGINE_LIST="gpt-3.5-turbo"
# "text-davinci-002 text-davinci-003 gpt-3.5-turbo gpt-4"
# ENGINE="text-davinci-002"
# gpt-3.5-turbo-0301

DATASET_LIST="multiarith svamp csqa gsm8k mathqa"
# DATASET="gsm8k"
NUM_TEST=-1 # 800, -1, 
SEED=1357
TEMP=0.7

INDEX_PATH="./index/"
DEFAULT_INPUT_GEN_PATH="./data/generated/"

APIKEY_PATH="./api_key.txt"

PORMPT_NAME="question_complex"
# "question_hardest"
# "NoNeed"
# "path_no_relevance"
# "path_no_coherence"
# "path_invalid_reasoning"

MODEL_LIST="complex_cot plan_solve" 

suffix_ans="001"


echo "Generating Data by LLMs ... "

for DATASET in ${DATASET_LIST}
    do
    echo "====== Now the dataset is "${DATASET} 
    for ENGINE in ${ENGINE_LIST}
        do
        echo "======== Now the employed LLM engine/model is "${ENGINE}

        if [ "${DATASET}" == "csqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ] || [ "${DATASET}" == "mathqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ]
        then
            PORMPT_NAME="question_complex"
        else
            PORMPT_NAME="question_hardest"
        fi

        echo "======== Now the reasoning strategy is complex_cot (FSL)"
        python generate_answer.py --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        echo "======== Now the reasoning strategy is complex_cot + self_verification (FSL)"
        python generate_answer.py --self_verification --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}
        echo "======== Now the reasoning strategy is complex_cot + self_consistency (FSL)" 
        python generate_answer.py --self_consistency --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}  
        PORMPT_NAME="NoNeed" 
        echo "======== Now the reasoning strategy is complex_cot (ZSL)"
        python generate_answer.py --learning_type zero_shot --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        echo "======== Now the reasoning strategy is complex_cot + self_consistency (ZSL)" 
        python generate_answer.py --self_consistency --learning_type zero_shot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}    
        echo "======== Now the reasoning strategy is complex_cot + self_verification (ZSL)"
        python generate_answer.py --self_verification --learning_type zero_shot --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}
        # echo "======== Now the reasoning strategy is plan_solve" 
        # python generate_answer.py --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        # echo "======== Now the reasoning strategy is plan_solve + self_verification" 
        # python generate_answer.py --self_verification --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}
        # echo "======== Now the reasoning strategy is plan_solve + self_consistency"  
        # python generate_answer.py --self_consistency --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}   
        done
    done
