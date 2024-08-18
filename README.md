# Towards A Unified View of Answer Calibration for Multi-Step Reasoning 🚀

<p align="center">
    <font size=6><strong>Towards A Unified View of Answer Calibration for Multi-Step Reasoning Hyperspheres</strong></font>
</p>

This repository is an offical project for the paper ''[**Towards A Unified View of Answer Calibration for Multi-Step Reasoning**](https://aclanthology.org/2024.nlrse-1.3/)'', accepted by [ACL 2024, Natural Language Reasoning and Structured Explanations Workshop](https://nl-reasoning-workshop.github.io/). 

## Usage 🛠️

First put your OpenAI API key in a file named ```api_key.txt```.

### Setup

```bash
sh setup.sh
```

### Run LLM generation

```bash
sh generate_answer.sh
```
```./data/generated/*``` contains the cached generation results. 
```./data/restored/*``` contains the cached reformulated generation results and accuracy results.

### Evaluation

```bash
sh eval_scores.sh
```
```./scores/*``` contains the cached evaluation results.


## How to Cite 📝
📋 Thank you very much for your interest in our work. If you use or extend our work, please cite the following paper:

```bibtex
@inproceedings{ACL2024_NLRSE_EvalReasoning,
    author    = {Shumin Deng and
               Ningyu Zhang and
               Nay Oo and
               Bryan Hooi},
  title       = {Towards A Unified View of Answer Calibration for Multi-Step Reasoning},
  booktitle   = {Proceedings of the 2nd Workshop on Natural Language Reasoning and Structured Explanations (@ACL 2024)},
  publisher   = {Association for Computational Linguistics},
  pages       = {25--38}
  year        = {2024},
  url         = {https://aclanthology.org/2024.nlrse-1.3}
}
```

