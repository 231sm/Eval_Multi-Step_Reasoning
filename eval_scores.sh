#!/bin/bash
# Usage:
# sh eval_scores.sh

DEFAULT_INPUT_GEN_PATH="./data/generated/"
DEFAULT_INPUT_RES_PATH="./data/restored/"

ENGINE_LIST="gpt-3.5-turbo"
# "text-davinci-002 text-davinci-003 gpt-3.5-turbo gpt-3.5-turbo-0301 gpt-4"
# ENGINE="text-davinci-002"
# gpt-3.5-turbo
# gpt-3.5-turbo-0301	
# text-davinci-003
# text-davinci-002

DATASET_LIST="multiarith svamp csqa gsm8k mathqa"
# DATASET="svamp"
# DATASET="gsm8k" # gsm8k,

MODEL_LIST="complex_cot plan_solve" 


PORMPT_NAME="question_complex"
# "question_hardest"
# "NoNeed"
# "path_no_relevance"
# "path_no_coherence"
# "path_invalid_reasoning"

NUM_TEST=-1 # 800, -1, 
SEED=1357
TEMP=0.0

suffix_ans="001"

for ENGINE in ${ENGINE_LIST}
    do
    echo "======== Restoring Data ..."
    for DATASET in ${DATASET_LIST}
        do

       if [ "${DATASET}" == "csqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ] || [ "${DATASET}" == "mathqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ]
        then
            PORMPT_NAME="question_complex"
        else
            PORMPT_NAME="question_hardest"
        fi

        python restore_data.py --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        python restore_data.py --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        python restore_data.py --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 

        PORMPT_NAME="NoNeed" 
        python restore_data.py --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        python restore_data.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        python restore_data.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 
        
        # python restore_data.py --reasoning_strategy plan_solve --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # python restore_data.py --reasoning_strategy plan_solve --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # python restore_data.py --reasoning_strategy plan_solve --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 
        
        # --batch_action

        done
    done

for ENGINE in ${ENGINE_LIST}
    do
    echo "====== Evaluating ROSCOE Scores ..."
    echo "======== Currently the employed LLM engine/model is "${ENGINE} 
    for DATASET in ${DATASET_LIST}
        do
        echo "========== On the dataset "${DATASET}

        if [ "${DATASET}" == "csqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ] || [ "${DATASET}" == "mathqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ]
        then
            PORMPT_NAME="question_complex"
        else
            PORMPT_NAME="question_hardest"
        fi

        python eval_roscoe/run_roscoe.py --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_roscoe/run_roscoe.py --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_roscoe/run_roscoe.py --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  

        PORMPT_NAME="NoNeed" 
        python eval_roscoe/run_roscoe.py --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_roscoe/run_roscoe.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_roscoe/run_roscoe.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        
        # python eval_roscoe/run_roscoe.py --reasoning_strategy plan_solve --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_roscoe/run_roscoe.py --reasoning_strategy plan_solve --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_roscoe/run_roscoe.py --reasoning_strategy plan_solve --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        
        # --batch_action

        done

    echo "====== Evaluating Classical Scores ..."
    echo "======== Currently the employed LLM engine/model is "${ENGINE} 
    for DATASET in ${DATASET_LIST}
        do
        echo "========== Currently evaluating on "${DATASET}

        if [ "${DATASET}" == "csqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ] || [ "${DATASET}" == "mathqa" ] && [ "${PORMPT_NAME}" == "question_hardest" ]
        then
            PORMPT_NAME="question_complex"
        else
            PORMPT_NAME="question_hardest"
        fi

        python eval_classical/run.py --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_classical/run.py --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_classical/run.py --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        
        PORMPT_NAME="NoNeed" 
        python eval_classical/run.py --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_classical/run.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        python eval_classical/run.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        
        # python eval_classical/run.py --reasoning_strategy plan_solve --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --reasoning_strategy plan_solve --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --reasoning_strategy plan_solve --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
         
        # --batch_action

        done
    done
