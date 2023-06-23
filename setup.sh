#!/bin/bash
# Usage:
# sh setup.sh

PROJECT_NAME="rethinking_answer_calibration_for_CoT"

DIR_EVAL_BASELINES="base_eval_models"


mkdir -p ../${DIR_EVAL_BASELINES}

cd ../${DIR_EVAL_BASELINES}


# prepare baseline models for calculation of evaluation scores 

pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ..

git clone https://github.com/neulab/BARTScore.git
cd BARTScore 
wget https://dl.fbaipublicfiles.com/parlai/projects/roscoe/fine_tuned_bartscore.pth
cd ..
# mv fine_tuned_bartscore.pth BARTScore/ 

git clone https://github.com/thompsonb/prism
cd prism
wget http://data.statmt.org/prism/m39v1.tar
tar xf m39v1.tar
export MODEL_DIR=m39v1/
cd ..
pip install nltk
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('stopwords')"


# enter the folder of our project 

cd ../${PROJECT_NAME}

pip install -r requirements.txt
