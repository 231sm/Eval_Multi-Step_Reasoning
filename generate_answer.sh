#!/bin/bash
# Usage:
# sh generate_answers.sh

ENGINE_LIST="gpt-3.5-turbo"
# "text-davinci-002 text-davinci-003 gpt-3.5-turbo"
# ENGINE="text-davinci-002"
# gpt-3.5-turbo-0301

DATASET_LIST="gsm8k svamp multiarith" 
# "gsm8k svamp multiarith mathqa csqa" 
# "gsm8k svamp multiarith mathqa csqa strategyqa" 
# DATASET="gsm8k" # gsm8k,
# "svamp"
# cosmosqa

MODEL_LIST="complex_cot plan_solve" 

NUM_TEST=-1 # 800, -1, 
SEED=1357
TEMP=0.7

INDEX_PATH="./index/"
DEFAULT_INPUT_GEN_PATH="./data/generated/"

APIKEY_PATH="./api_key.txt"

PORMPT_NAME="question_hardest"

suffix_ans="001"
# suffix_ans="0.25"
# suffix_ans="0.75"
# suffix_ans="0.95"
# suffix_ans_list="0.98"
# suffix_ans_list="0.05 0.60 0.95"
# suffix_ans_list="0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90"
# suffix_ans_list="0.111 0.294 0.556 0.909"
suffix_ans_list="0.05 0.10 0.111 0.20 0.294 0.30 0.40 0.50 0.556 0.60 0.70 0.80 0.90 0.909 0.95 0.98"


echo "Generating Data by LLMs ... "

for DATASET in ${DATASET_LIST}
    do
    echo "====== Now the dataset is "${DATASET} 
    for ENGINE in ${ENGINE_LIST}
        do
        echo "======== Now the employed LLM engine/model is "${ENGINE}

        echo "======== Now the reasoning strategy is complex_cot (FSL)"
        python generate_answer.py --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}

        echo "======== Now the reasoning strategy is complex_cot + self_consistency (FSL)" 
        python generate_answer.py --self_consistency --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP} 
        
        echo "======== Now the reasoning strategy is complex_cot + self_verification (FSL)" 
        python generate_answer.py --self_verification --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP} 
        
        echo "======== Now the reasoning strategy is complex_cot (ZSL)"
        python generate_answer.py --learning_type zero_shot --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        
        echo "======== Now the reasoning strategy is complex_cot + self_consistency (ZSL)" 
        python generate_answer.py --self_consistency --learning_type zero_shot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}    
        
        echo "======== Now the reasoning strategy is complex_cot + self_verification (ZSL)"
        python generate_answer.py --self_verification --learning_type zero_shot --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}
        
        # echo "======== Now the reasoning strategy is plan_solve" 
        # python generate_answer.py --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        
        # echo "======== Now the reasoning strategy is plan_solve + self_consistency"  
        # python generate_answer.py --self_consistency --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}  
        
        # echo "======== Now the reasoning strategy is plan_solve + self_verification" 
        # python generate_answer.py --self_verification --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}
             
        for suffix_ans in ${suffix_ans_list}
            do
            echo "======== Now alpha is "${suffix_ans} 

            echo "======== Now the reasoning strategy is complex_cot + (self_verification + self_consistency) (ZSL)" 
            python generate_answer.py --self_consistency --self_verification --learning_type zero_shot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}    
            
            echo "======== Now the reasoning strategy is complex_cot + (self_verification + self_consistency) (FSL)" 
            python generate_answer.py --self_consistency --self_verification --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}
            
            # --overwrite_prediction
            # --learning_type zero_shot
            # --reasoning_strategy ${MODEL} 
            # --dialog_icl
            # --self_consistency
            # --self_verification
            # --prompt_name ${PORMPT_NAME} 
            # --test_ind ${INDEX_PATH}${DATASET}"/validation_index.npy" --out_dir ${DEFAULT_INPUT_GEN_PATH}${DATASET}
            # --apikey_file ${APIKEY_PATH}
            done
        done
    done

PORMPT_NAME="question_complex"
DATASET_LIST="mathqa csqa"
# "gsm8k svamp multiarith mathqa csqa"
suffix_ans="001"
suffix_ans_list="0.05 0.10 0.111 0.20 0.294 0.30 0.40 0.50 0.556 0.60 0.70 0.80 0.90 0.909 0.95 0.98"

for DATASET in ${DATASET_LIST}
    do
    echo "====== Now the dataset is "${DATASET} 
    for ENGINE in ${ENGINE_LIST}
        do
        echo "======== Now the employed LLM engine/model is "${ENGINE}

        echo "======== Now the reasoning strategy is complex_cot (FSL)"
        python generate_answer.py --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        
        echo "======== Now the reasoning strategy is complex_cot + self_consistency (FSL)" 
        python generate_answer.py --self_consistency --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP} 
        
        echo "======== Now the reasoning strategy is complex_cot + self_verification (FSL)" 
        python generate_answer.py --self_verification --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP} 
        
        echo "======== Now the reasoning strategy is complex_cot (ZSL)"
        python generate_answer.py --learning_type zero_shot --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        
        echo "======== Now the reasoning strategy is complex_cot + self_consistency (ZSL)" 
        python generate_answer.py --self_consistency --learning_type zero_shot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}    
        
        echo "======== Now the reasoning strategy is complex_cot + self_verification (ZSL)"
        python generate_answer.py --self_verification --learning_type zero_shot --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}

        # echo "======== Now the reasoning strategy is plan_solve" 
        # python generate_answer.py --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED}
        
        # echo "======== Now the reasoning strategy is plan_solve + self_consistency"  
        # python generate_answer.py --self_consistency --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}  
        
        # echo "======== Now the reasoning strategy is plan_solve + self_verification" 
        # python generate_answer.py --self_verification --learning_type zero_shot --reasoning_strategy plan_solve --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}
              
        for suffix_ans in ${suffix_ans_list}
            do
            echo "======== Now alpha is "${suffix_ans} 

            echo "======== Now the reasoning strategy is complex_cot + (self_verification + self_consistency) (ZSL)" 
            python generate_answer.py --self_consistency --self_verification --learning_type zero_shot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP}    
            
            echo "======== Now the reasoning strategy is complex_cot + (self_verification + self_consistency) (FSL)" 
            python generate_answer.py --self_consistency --self_verification --reasoning_strategy complex_cot --suffix_ans ${suffix_ans} --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --apikey_file ${APIKEY_PATH} --seed ${SEED} --temp ${TEMP} 

            # --overwrite_prediction
            # --learning_type zero_shot
            # --reasoning_strategy ${MODEL} 
            # --dialog_icl
            # --self_consistency
            # --self_verification
            # --prompt_name ${PORMPT_NAME} 
            # --test_ind ${INDEX_PATH}${DATASET}"/validation_index.npy" --out_dir ${DEFAULT_INPUT_GEN_PATH}${DATASET}
            # --apikey_file ${APIKEY_PATH}
            done
        done
    done


