# GMP

## About Repo
The Repo is based on [Generated Multimodal Prompt](https://github.com/YangXiaocui1215/GMP), with paper: **"Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt"**.
The code is running on Python.


## Data
The image-text data can be downloaded from [Kaggle](https://www.kaggle.com/datasets/tisdang/twitter-data-for-my-seminar).



## Task Training
To Train the JMASA, MASC, and MATE tasks on two twitter datasets, you can just run the following code. 
Note that you need to change all the file path in file "GMP/src/data/jsons/few_shot_for_prompt/twitter_2015/" and "GMP/src/data/jsons/few_shot_for_prompt/twitter17_info.json" to your own path.

### For the MATE task,
```
sh scripts/15MATE_pretrain_for_generated_prompt_multitasks.sh
sh scripts/17MATE_pretrain_for_generated_prompt_multitasks.sh
```
### For the JMASA task,
```
sh scripts/15_pretrain_full_for_generated_dual_prompts_multitasks_Aspect.sh
sh scripts/17_pretrain_full_for_generated_dual_prompts_multitasks_Aspect.sh
```
### For the MASC task,
```
sh scripts/15MASC_pretrain_for_generated_prompt.sh
sh scripts/17MASC_pretrain_for_generated_prompt.sh
```

## Acknowledgements
Some codes are based on the codes of [VLP-MABSA](https://github.com/NUSTM/VLP-MABSA), many thanks!
