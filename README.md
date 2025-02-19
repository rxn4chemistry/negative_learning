# Negative learning

Code for negative learning of chemical reactions

## Installation instructions
```bash
conda create -n rxn-neg-learn python=3.10
conda activate rxn-neg-learn
pip install poetry

git clone ...
PATH_TO_REPO=path/to/this/repo
cd $PATH_TO_REPO
poetry install
```

To add a package do `poetry add <my_package>`
To run tests just launch `pytest tests/ -W ignore::DeprecationWarning`

Before commiting please run the following commands:
```
ruff check . --fix # Styling/Linting
ruff format . # Formatting
```

## Instructions on datasets extraction
First download all datasets in the `data/source_data` folder.
```bash
conda activate rxn-neg-learn
brew update
brew install p7zip
bash $PATH_TO_REPO/data/download_datasets.sh
```

**Generation of regioSQM20**: simply call the command  ```generate-regiosqm-dataset $PATH_TO_REPO/data/regiosqm``` from the command line. 
The extracted data will be saved as ```$PATH_TO_REPO/data/regiosqm/regiosqm.csv```

**Generation of USPTO**: simply call the command  ```generate-uspto-dataset $PATH_TO_REPO/data/uspto``` from the command line. 
The extracted data will be saved as ```$PATH_TO_REPO/data/uspto/uspto.csv```

Above, if the OUTPUT_FOLDER is not empty the code throws an error (you have to delete the files inside first).


## Dataset splitting
To generate new splits for a dataset many options are available

### Option 1: Simple random splitting
Splits randomly a dataset according to a specified ratio for test and validation 
```bash
split-simple input_file_csv \
             output_folder \
             --split-ratio 0.1 \
             --reaction_column_name 'rxn'
```
The above combined with a removal of overlap with the finetuning datasets was used for USPTO:
```bash
remove-overlap ${PATH_TO_REPO}/data/uspto/uspto.csv \
               --input_files_csv ${PATH_TO_REPO}/data/regiosqm/regiosqm.csv
               (--input_files_csv ...) # in principle can be many
               
split-simple ${PATH_TO_REPO}/data/uspto/uspto.csv.nooverlap \
             data/uspto/0.1split-ratio \
             --split_ratio 0.1 \
             --reaction_column_name 'rxn'
```

### Option 2 : Generate splits where the number of positives is decreasing
```bash
generate-decreasing-positive-splits  input_file_csv \
                                     output_file_csv \
                                     --reaction_column_name "rxn" \
                                     --idx_column_name "idx" \
                                     --split_ratio 0.1 \
                                     --splitting_method random \
                                     --seed 42 \
                                     --validation_set
```
The above was used for RegioSQM20 for the experiment with decreasing positives:
```bash
DATA_SEED=42
generate-decreasing-positive-splits  ${PATH_TO_REPO}/data/regiosqm/regiosqm.csv \
                                     ${PATH_TO_REPO}/data/regiosqm/regiosqm.csv.decreasingpos.ratio0.3.seed${DATA_SEED} \
                                     --reaction_column_name "rxn" \
                                     --idx_column_name "idx" \
                                     --split_ratio 0.3 \
                                     --splitting_method random \
                                     --seed ${DATA_SEED} \
                                     --validation_set
```
Different seeds were used to get new splits for error analysis.

### Option 3 : Generate multiple splits according to different methods (not only random)
Use the following script template to generate a new split for a dataset according to a certain method (Not used in the experiments of the paper)
```bash
generate_multiple_splits input_csv_file \
                         output_csv_file \
                         --reaction_column_name "rxn" \
                         --split_ratio 0.1 \
                         --splitting_method random \
                         --seed 45 \
                         (--seed 22) # Add more to generate multiple splits at the same time
                         --validation_set \
                         --remove_duplicate_columns # to remove duplicate splits (can happen for small datasets)
```
Check the script for the list of available methods.

In case you want to reserve a ratio of the dataset for training another model (e.g. the scorer) you can set
the flag `--reserve_dataset_split_ratio` giving a float between 0 and 1.

## Save split files
We can now save all the regioSQM20 splits in appropriate folders by running on appropriate columns of the `input_file_csv` provided below:
```bash
DATA_SEED=42
OUTPUT_PATH=${PATH_TO_REPO}/data/regiosqm
save-split-files ${PATH_TO_REPO}/data/regiosqm/regiosqm.csv.decreasingpos.ratio0.3.seed${DATA_SEED} \
                 ${OUTPUT_PATH} \
                 --split_columns "sratio0.3_random_seed${DATA_SEED}_k0" \
                 --split_columns "sratio0.3_random_seed${DATA_SEED}_k1" \
                 --split_columns "sratio0.3_random_seed${DATA_SEED}_k2" \
                 --split_columns "sratio0.3_random_seed${DATA_SEED}_k3" \
                 --split_columns "sratio0.3_random_seed${DATA_SEED}_k4" \
                 --reaction_column_name 'rxn'
```

Files are saved in `${OUTPUT_PATH}` under `decreasingpos`

## Create a joint vocabulary
For simplicity, create a vocabulary that already includes the tokens from both the base dataset (USPTO) and the
finetuning dataset(s).

```bash
OUTPUT_PATH=${PATH_TO_REPO}/data/additional
create-vocab ${OUTPUT_PATH} \
             --file_jsonl ${PATH_TO_REPO}/data/regiosqm/decreasingpos/sratio0.3_random_seed42_k0/all/data-train-with-valid-all.jsonl \
             --file_jsonl ${PATH_TO_REPO}/data/regiosqm/decreasingpos/sratio0.3_random_seed42_k0/all/data-test-all.jsonl \
             --file_jsonl ${PATH_TO_REPO}/data/uspto/0.1split-ratio/data-train-with-valid.jsonl \
             --file_jsonl ${PATH_TO_REPO}/data/uspto/0.1split-ratio/data-test.jsonl
             
```

The vocab file is saved as `${OUTPUT_PATH}/vocab.txt` .

## Base model pretraining
To train the base Transformer model with usual MLE, scripts are provided in the folder `${PATH_TO_REPO}/bin`.
Check the scripts before launching them: few entries need to be set, default parameters are provided.

ATTENTION: GPU needed

```bash
cd ${PATH_TO_REPO}/bin
bash pretrain.sh
```

The base model can continue training for more steps with the following script:
```bash
cd ${PATH_TO_REPO}/bin
bash restart_pretrain.sh
```
Remember to set the starting checkpoint in the script above.

For testing
```bash
cd ${PATH_TO_REPO}/bin
bash test_pretrain.sh
```

## MLE Finetuning
To finetune the base Transformer model with usual MLE, scripts are provided in the folder `${PATH_TO_REPO}/bin`.
Check the scripts before launching them: few entries need to be set, default parameters are provided.

ATTENTION: GPU needed

```bash
cd ${PATH_TO_REPO}/bin
bash finetune_mle.sh
```

For testing
```bash
cd ${PATH_TO_REPO}/bin
bash test_finetune_mle.sh
```

## Reward Model training
When we know all the positives for each negative we could use the ideal scorer during training of the rl.
However, in common situations we might have negative reactions for which we do not know the negative counterpart.
In this latter case we need to train a reward model.

### Step 1: mlm training
The first step in building a reward model for small datasets leverages pretraining with a big unlabeled corpus.
The task is performed training with masked language modeling (MLM) a BERT model on the USPTO reactions.
Launching the following script pretrains BERT

ATTENTION: GPU needed

```bash
cd ${PATH_TO_REPO}/bin
bash train_mlm.sh
```

### Step 2: intermediate classifier training
With small datasets we saw it is beneficial to "prepare" the embeddings of the pretrained model with an intermediate
classification training on USPTO positives and artificially generated negatives. 
First, we collect predictions from the pretrained base model on USPTO (for forward prediction). We use beam search
and collect the top 3 predictions for the USPTO validation dataset:

ATTENTION: GPU needed

```bash
cd ${PATH_TO_REPO}/bin
bash test_pretrain.sh
```

Once the predictions are collected we can create the dataset for finetuning:

```bash
convert-jsonl-dataset-to-txt --jsonl_file ${PATH_TO_REPO}/data/uspto/0.1split-ratio/data-valid.jsonl \
                             --output_file ${PATH_TO_REPO}/data/uspto/0.1split-ratio/data-valid.txt

generate-synthetic-classification-dataset \
            --predictions_file_txt ${PATH_TO_REPO}/data/uspto/0.1split-ratio/predictions-valid_beam10_3preds.txt \
            --targets_file_txt ${PATH_TO_REPO}/data/uspto/0.1split-ratio/data-valid.txt \
            --strategy all --randomize
```
The script saves in `${PATH_TO_REPO}/data/uspto/0.1split-ratio` two files (`dataset_for_reward_model_train.csv`, `dataset_for_reward_model_valid.csv`)
that are used for finetuning the BERT with a classification head:

```bash
cd ${PATH_TO_REPO}/bin
bash finetune_reward_model.sh
```

The "intermediate" classification model is saved under `${PATH_TO_REPO}/models/reward_model_intermediate`

### Step 3: SVM Training
For the training no gpu is needed. The following script
performs k-fold cross validation across a provided set of hyperparameters. A sample of these can be found in 
`${PATH_TO_REPO}/data/additional/hyperparams_search_svm.json` . The pretrained model used 
is the intermediate reward model of the previous section but any pretrained reaction BERT can be used.

```bash
PRETRAINED_MODEL_PATH=${PATH_TO_REPO}/models/reward_model_intermediate/checkpoint-168525
svm-train --embeddings_model_path  ${PRETRAINED_MODEL_PATH} \
          --training_dataset_file ${PATH_TO_REPO}/data/regiosqm/decreasingpos/sratio0.3_random_seed42_k3/all/data-train-all.jsonl \
          --validation_dataset_file ${PATH_TO_REPO}/data/regiosqm/decreasingpos/sratio0.3_random_seed42_k3/all/data-valid-all.jsonl \
          --grid_search_parameters_file ${PATH_TO_REPO}/data/additional/hyperparams_search_svm.json \
          --output_path ${PATH_TO_REPO}/models/svm_classification/k3_models/comparison \
          --k_fold_cross_validation
```
The above script saves in the output folder a `.csv` file with the performance comparison from all models.
To train and save the selected best model run:

```bash
PRETRAINED_MODEL_PATH=${PATH_TO_REPO}/models/reward_model_intermediate/checkpoint-168525
svm-train --embeddings_model_path  ${PRETRAINED_MODEL_PATH} \
          --training_dataset_file ${PATH_TO_REPO}/data/regiosqm/decreasingpos/sratio0.3_random_seed42_k3/all/data-train-all.jsonl \
          --validation_dataset_file ${PATH_TO_REPO}/data/regiosqm/decreasingpos/sratio0.3_random_seed42_k3/all/data-valid-all.jsonl \
          --grid_search_parameters_file ${PATH_TO_REPO}/data/additional/hyperparams_search_svm_selected.json \
          --output_path ${PATH_TO_REPO}/models/svm_classification/k3_models \
          --save_models
```

## RL Finetuning

To finetune the base Transformer model with Reinforcement Learning, scripts are provided in the folder `${PATH_TO_REPO}/bin`.
Check the scripts before launching them: few entries need to be set, default parameters are provided.

First, generate the lookup table for the baseline model training. A table is needed for each seed and for each subset of
decreasing positives.

```bash
K=0
DATA_SEED=42
generate-baseline-lookup-table -f ${PATH_TO_REPO}/data/regiosqm/decreasingpos/sratio0.3_random_seed${DATA_SEED}_k${K}/all/data-train-all.jsonl \
                               --output_file ${PATH_TO_REPO}/data/additional/baseline_targets_k${K}_seed${DATA_SEED}_augm3.json \
                               --lookup_mode statistical --augment --augmentation_number 3
```

ATTENTION: GPU needed below

```bash
cd ${PATH_TO_REPO}/bin
bash finetune_rl.sh
```

For testing the model you can use the following script

```bash
cd ${PATH_TO_REPO}/bin
bash test_finetune_rl.sh
```


