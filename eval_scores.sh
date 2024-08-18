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

DATASET_LIST="gsm8k svamp multiarith mathqa csqa"
# "gsm8k svamp multiarith mathqa csqa"
# "mathqa"
# "csqa gsm8k mathqa svamp multiarith"
# "mathqa"
# "gsm8k svamp multiarith"
# "gsm8k svamp multiarith csqa"
# "svamp"
# "gsm8k svamp multiarith mathqa"
# "gsm8k svamp multiarith mathqa csqa strategyqa"

# DATASET="svamp"
# DATASET="gsm8k" # gsm8k,

MODEL_LIST="complex_cot"
# "complex_cot plan_solve"  

# PORMPT_NAME="question_hardest_direct"
# PORMPT_NAME="question_hardest"
PORMPT_NAME="question_complex"
PORMPT_NAME_ZSL="NoNeed"
# PORMPT_NAME="NoNeed"
# PORMPT_NAME="path_no_coherence"
# PORMPT_NAME="path_no_relevance"
# "question_hardest"
# "path_invalid_reasoning"
# "path_no_coherence"
# "path_no_relevance"

NUM_TEST=-1 # 800, -1, 
SEED=1357
TEMP=0.0

suffix_ans="001"
# suffix_ans="001-1ver*10"
# suffix_ans="001neweval-1ver*10"
# suffix_ans="001-1ver-loadpre"

for ENGINE in ${ENGINE_LIST}
    do
    echo "======== Restoring Data ..."
    for DATASET in ${DATASET_LIST}
        do
        # # python restore_data.py --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # # python restore_data.py --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # python restore_data.py --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 
        # python restore_data.py --self_verification --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 
         
        # # # python restore_data.py --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # # # python restore_data.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # # # python restore_data.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 
        # python restore_data.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME_ZSL} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 
        # # # python restore_data.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME_ZSL} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # python restore_data.py --learning_type zero_shot --self_verification --self_consistency --prompt_name ${PORMPT_NAME_ZSL} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 


        python restore_data.py --self_verification --self_consistency --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --batch_action 
        


        # python restore_data.py --reasoning_strategy plan_solve --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # python restore_data.py --reasoning_strategy plan_solve --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST}
        # python restore_data.py --reasoning_strategy plan_solve --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} 
        # --batch_action

        # --prompt_name ${PORMPT_NAME} 
        # echo "======== For reasoning strategy of complex_cot"
        # python restore_data.py --reasoning_strategy complex_cot --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
        # python restore_data.py --self_consistency --reasoning_strategy complex_cot --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
        # python restore_data.py --self_verification --reasoning_strategy complex_cot --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
        # python restore_data.py --self_consistency --reasoning_strategy plan_solve --learning_type zero_shot --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
        # echo "======== For reasoning strategy of plan_solve" 
        # python restore_data.py --reasoning_strategy plan_solve --learning_type zero_shot --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
        # python restore_data.py --suffix_ans ${suffix_ans} --self_consistency --reasoning_strategy plan_solve --dataset ${DATASET} --learning_type zero_shot --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
        # python restore_data.py --self_verification --reasoning_strategy plan_solve --dataset ${DATASET} --learning_type zero_shot --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
        done 
    # python restore_data.py --learning_type zero_shot --prompt_name ${PORMPT_NAME} --dataset ${DATASET_LIST} --engine ${ENGINE} --num_test ${NUM_TEST} --seed ${SEED} --temp ${TEMP}
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

for ENGINE in ${ENGINE_LIST}
    do
    echo "====== Evaluating ROSCOE Scores ..."
    echo "======== Currently the employed LLM engine/model is "${ENGINE} 
    for DATASET in ${DATASET_LIST}
        do
        echo "========== On the dataset "${DATASET}
        # # python eval_roscoe/run_roscoe.py --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # # python eval_roscoe/run_roscoe.py --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_roscoe/run_roscoe.py --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        # python eval_roscoe/run_roscoe.py --self_consistency --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
          
        # # # python eval_roscoe/run_roscoe.py --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # # # python eval_roscoe/run_roscoe.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # # # python eval_roscoe/run_roscoe.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        # python eval_roscoe/run_roscoe.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME_ZSL} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        # # # python eval_roscoe/run_roscoe.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME_ZSL} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_roscoe/run_roscoe.py --learning_type zero_shot --self_verification --self_consistency --prompt_name ${PORMPT_NAME_ZSL} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        

        python eval_roscoe/run_roscoe.py --self_verification --self_consistency --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --batch_action
         


        # python eval_roscoe/run_roscoe.py --reasoning_strategy plan_solve --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_roscoe/run_roscoe.py --reasoning_strategy plan_solve --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_roscoe/run_roscoe.py --reasoning_strategy plan_solve --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        # --batch_action

        # # python eval_roscoe/run_roscoe.py --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir
        # echo "========== For reasoning strategy of complex_cot" 
        # python eval_roscoe/run_roscoe.py --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --suffix_ans ${suffix_ans} --overwrite_output_dir
        # python eval_roscoe/run_roscoe.py --self_consistency --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --suffix_ans ${suffix_ans} --overwrite_output_dir
        # python eval_roscoe/run_roscoe.py --self_verification --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --suffix_ans ${suffix_ans} --overwrite_output_dir
        # echo "========== For reasoning strategy of plan_solve"
        # python eval_roscoe/run_roscoe.py --reasoning_strategy plan_solve --learning_type zero_shot --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --suffix_ans ${suffix_ans} --overwrite_output_dir
        # python eval_roscoe/run_roscoe.py --self_consistency --reasoning_strategy plan_solve --learning_type zero_shot --dataset ${DATASET} --engine ${ENGINE} --suffix_ans ${suffix_ans} --num_test ${NUM_TEST} --overwrite_output_dir
        # python eval_roscoe/run_roscoe.py --self_verification --reasoning_strategy plan_solve --learning_type zero_shot --dataset ${DATASET} --engine ${ENGINE} --suffix_ans ${suffix_ans} --num_test ${NUM_TEST} --overwrite_output_dir
        done

    echo "====== Evaluating Classical Scores ..."
    echo "======== Currently the employed LLM engine/model is "${ENGINE} 
    for DATASET in ${DATASET_LIST}
        do
        echo "========== Currently evaluating on "${DATASET}
        # python eval_classical/run.py --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        
        # python eval_classical/run.py --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        # python eval_classical/run.py --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME_ZSL} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
        
        # python eval_classical/run.py --reasoning_strategy plan_solve --learning_type zero_shot --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --reasoning_strategy plan_solve --learning_type zero_shot --self_consistency --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir 
        # python eval_classical/run.py --reasoning_strategy plan_solve --learning_type zero_shot --self_verification --prompt_name ${PORMPT_NAME} --suffix_ans ${suffix_ans} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir  
         
        # --batch_action

        # python eval_classical/run.py --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --suffix_ans ${suffix_ans} --overwrite_output_dir
        done
    done




 
#  for file in ./data/restored/*
#         do
#         # # echo "========== Currently scanning on "${file}
#         # temp=${file}
#         # for DATASET in ${DATASET_LIST}
#         #     do
#         #     # echo ${DATASET} 
#         #     temp=${temp%%_*}
#         #     # echo ${temp}"---"${DEFAULT_INPUT_RES_PATH}${DATASET}
#         #     if [ "${temp}" = "${DEFAULT_INPUT_RES_PATH}${DATASET}" ]
#         #     then
#         #     # && ${file##*|} == "restore.jsonl"
#         #         echo "========== Currently evaluating on "${file} 
#         #         TEST_FILE=${file}
#         #         TEST_FILE_TEMP=${DATASET}".jsonl"
#         #         cp -f ${TEST_FILE} ${DEFAULT_INPUT_RES_PATH}${TEST_FILE_TEMP} # -i
                 
#         #         python eval_classical/run.py --prompt_name ${PORMPT_NAME} --dataset ${DATASET} --engine ${ENGINE} --num_test ${NUM_TEST} --overwrite_output_dir
#         #         # --learning_type zero_shot 
#         #         break
#         #     else
#         #         continue
#         #     fi
#         #     done  
#         done