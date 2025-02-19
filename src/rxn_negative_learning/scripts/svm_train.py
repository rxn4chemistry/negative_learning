import json
import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
import scipy
import torch
from rxn.utilities.logging import setup_console_logger
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel

from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.scripts.finetune_scorer import get_associated_max_len
from rxn_negative_learning.utils.smiles_utils import (
    get_reaction_from_precursors_and_product_smiles,
    load_jsonl_dataset_to_dataframe,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_hyperparams(grid_search_parameters_file: Path) -> Tuple[List[Any], List[Any], List[Any]]:
    with open(grid_search_parameters_file) as f:
        hyperparams = json.load(f)
        logger.info(f"Testing the following hyperparams: {hyperparams}")
    return hyperparams["c"], hyperparams["kernel"], hyperparams["gamma"]


def svm_train(
    embeddings_model_path: Path,
    training_datasets: Union[pd.DataFrame, List[pd.DataFrame]],
    validation_datasets: Union[pd.DataFrame, List[pd.DataFrame]],
    output_path: Path,
    grid_search_parameters_file: Optional[Path] = None,
    save_models: bool = False,
    k_fold_cross_validation: bool = True,
):
    device = "gpu" if torch.cuda.is_available() else "cpu"

    logger.info("Loading the embeddings model and tokenizer ...")
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=embeddings_model_path).to(
        device
    )
    model_max_length = get_associated_max_len(embeddings_model_path)
    logger.info(f"Model max sequence length: {model_max_length}")

    tokenizer = SmilesTokenizer.from_pretrained(embeddings_model_path)
    logger.info("Loading the embeddings model and tokenizer ... Done")

    if isinstance(training_datasets, pd.DataFrame):
        training_datasets = [training_datasets]
        validation_datasets = [validation_datasets]

    if len(training_datasets) < 2 and k_fold_cross_validation:
        logger.info("Performing k-fold stratified cross validation ...")
        # build a cross validation with k folds -> stratified version
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = training_datasets[0]["text"]
        y = training_datasets[0]["label"]
        training_datasets = []
        training_test_datasets = []
        logger.info("Folds statistics ...")
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            training_datasets.append(
                pd.DataFrame({"text": X_train, "label": y_train}).reset_index(drop=True)
            )
            training_test_datasets.append(
                pd.DataFrame({"text": X_test, "label": y_test}).reset_index(drop=True)
            )
            validation_datasets.append(validation_datasets[0])

            logger.info(
                f"Fold {i + 1}: lab 0->{len(training_datasets[i][training_datasets[i]['label'] == 0])} examples, lab 1->{len(training_datasets[i][training_datasets[i]['label'] == 1])} examples."
            )

    all_metrics = []
    n_splits = len(training_datasets)
    confidence = 0.95  # Change to your desired confidence level
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_splits - 1) if n_splits > 1 else None

    for i in range(n_splits):
        logger.info(f"Tokenizing training dataset {i} ...")
        tokenized_train = tokenizer(
            training_datasets[i]["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=model_max_length,
        )
        logger.info(f"Tokenizing training dataset {i} ... Done")
        logger.info(
            f"Training set statistics: pos -> {len(training_datasets[i].loc[training_datasets[i]['label'] == 1])} neg -> {len(training_datasets[i].loc[training_datasets[i]['label'] == 0])}"
        )

        logger.info(f"Tokenizing validation dataset {i}...")
        tokenized_valid = tokenizer(
            validation_datasets[i]["text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=model_max_length,
        )
        logger.info(f"Tokenizing validation dataset {i}... Done")
        logger.info(
            f"Validation set statistics: pos -> {len(validation_datasets[i].loc[validation_datasets[i]['label'] == 1])} neg -> {len(validation_datasets[i].loc[validation_datasets[i]['label'] == 0])}"
        )

        with torch.no_grad():
            logger.info("Getting train set embeddings ...")
            hidden_train = model(**tokenized_train)
            logger.info("Getting train set embeddings ... Done")

            logger.info("Extracting [CLS] token hidden state ...")
            training_datasets[i]["embeddings"] = (
                hidden_train.last_hidden_state[:, 0, :].numpy().tolist()
            )
            logger.info("Extracting [CLS] token hidden state ... Done")
            logger.info(f"Embeddings dim train: {len(training_datasets[i]['embeddings'].loc[0])}")

            logger.info("Getting validation set embeddings ...")
            hidden_valid = model(**tokenized_valid)
            logger.info("Getting validation set embeddings ... Done")

            logger.info("Extracting [CLS] token hidden state ...")
            validation_datasets[i]["embeddings"] = (
                hidden_valid.last_hidden_state[:, 0, :].numpy().tolist()
            )
            logger.info("Extracting [CLS] token hidden state ... Done")
            logger.info(f"Embeddings dim valid: {len(training_datasets[i]['embeddings'].loc[0])}")

    if grid_search_parameters_file is None:
        logger.info("No hyperparameters file provided ... using default values for svm model")
        c_values = [1.0]
        kernel_values = ["rbf"]
        gamma_values = ["scale"]
    else:
        c_values, kernel_values, gamma_values = load_hyperparams(grid_search_parameters_file)

    for c in c_values:
        for kernel in kernel_values:
            for gamma in gamma_values:
                pos_train_accuracy = []
                neg_train_accuracy = []
                f1_train_score = []
                auroc_train_score = []
                pos_valid_accuracy = []
                neg_valid_accuracy = []
                f1_valid_score = []
                auroc_valid_score = []
                precision_score_list = []

                for i in range(len(training_datasets)):
                    logger.info(
                        f"Training {i}th svm classifier : C -> {c}  kernel -> {kernel} gamma -> {gamma}..."
                    )
                    clsvm = svm.SVC(C=c, kernel=kernel, gamma=gamma, class_weight="balanced")
                    training_dataset = training_datasets[i]
                    validation_dataset = validation_datasets[i]
                    clsvm.fit(
                        np.array(training_dataset["embeddings"].tolist()),
                        training_dataset["label"].tolist(),
                    )

                    pos_train = training_dataset.loc[training_dataset["label"] == 1]
                    neg_train = training_dataset.loc[training_dataset["label"] == 0]
                    pos_valid = validation_dataset.loc[validation_dataset["label"] == 1]
                    neg_valid = validation_dataset.loc[validation_dataset["label"] == 0]

                    pos_train_accuracy.append(
                        round(
                            clsvm.score(
                                np.array(pos_train["embeddings"].tolist()),
                                pos_train["label"].tolist(),
                            ),
                            4,
                        )
                    )
                    neg_train_accuracy.append(
                        round(
                            clsvm.score(
                                np.array(neg_train["embeddings"].tolist()),
                                neg_train["label"].tolist(),
                            ),
                            4,
                        )
                    )
                    f1_train_score.append(
                        round(
                            f1_score(
                                training_dataset["label"].tolist(),
                                clsvm.predict(training_dataset["embeddings"].tolist()),
                            ),
                            4,
                        )
                    )
                    auroc_train_score.append(
                        round(
                            roc_auc_score(
                                training_dataset["label"].tolist(),
                                clsvm.predict(training_dataset["embeddings"].tolist()),
                            ),
                            4,
                        )
                    )
                    pos_valid_accuracy.append(
                        round(
                            clsvm.score(
                                np.array(pos_valid["embeddings"].tolist()),
                                pos_valid["label"].tolist(),
                            ),
                            4,
                        )
                    )
                    neg_valid_accuracy.append(
                        round(
                            clsvm.score(
                                np.array(neg_valid["embeddings"].tolist()),
                                neg_valid["label"].tolist(),
                            ),
                            4,
                        )
                    )
                    f1_valid_score.append(
                        round(
                            f1_score(
                                validation_dataset["label"].tolist(),
                                clsvm.predict(validation_dataset["embeddings"].tolist()),
                            ),
                            4,
                        )
                    )
                    auroc_valid_score.append(
                        round(
                            roc_auc_score(
                                validation_dataset["label"].tolist(),
                                clsvm.predict(validation_dataset["embeddings"].tolist()),
                            ),
                            4,
                        )
                    )
                    precision_score_list.append(
                        round(
                            precision_score(
                                validation_dataset["label"].tolist(),
                                clsvm.predict(validation_dataset["embeddings"].tolist()),
                            ),
                            4,
                        )
                    )

                    if save_models:
                        model_path = (
                            output_path / f"#{i}_svm_model_C{c}_kernel{kernel}_gamma{gamma}.pkl"
                        )
                        logger.info(f"Saving trained model to {model_path} ...")
                        with open(model_path, "wb") as f:
                            pickle.dump(clsvm, f)
                        logger.info(f"Saving trained model to {model_path} ... Done")

                        metrics_path = (
                            output_path / f"#{i}_svm_metrics_C{c}_kernel{kernel}_gamma{gamma}.json"
                        )
                        logger.info(f"Saving metrics to {metrics_path} ...")
                        with open(metrics_path, "w") as f:
                            m = {
                                "model": "svm-class",
                                "c": c,
                                "kernel": kernel,
                                "gamma": gamma,
                                "+train_acc": pos_train_accuracy[i],
                                "-train_acc": neg_train_accuracy[i],
                                "f1_train": f1_train_score[i],
                                "auroc_train": auroc_train_score[i],
                                "+valid_acc": pos_valid_accuracy[i],
                                "-valid_acc": neg_valid_accuracy[i],
                                "f1_valid": f1_valid_score[i],
                                "auroc_valid": auroc_valid_score[i],
                                "precision": precision_score_list[i],
                            }
                            json.dump(m, f, indent=4)
                        logger.info(f"Saving metrics to {metrics_path} ... Done")

                metrics = {
                    "model": "svm-class",
                    "c": c,
                    "kernel": kernel,
                    "gamma": gamma,
                    "+train_acc": np.mean(pos_train_accuracy),
                    "+train_acc_stderr": np.std(pos_train_accuracy, ddof=1) / np.sqrt(n_splits),
                    "-train_acc": np.mean(neg_train_accuracy),
                    "-train_acc_stderr": np.std(neg_train_accuracy, ddof=1) / np.sqrt(n_splits),
                    "f1_train_avg": np.mean(f1_train_score),
                    "f1_train_avg_stderr": np.std(f1_train_score, ddof=1) / np.sqrt(n_splits),
                    "auroc_train_avg": np.mean(auroc_train_score),
                    "auroc_train_avg_stderr": np.std(auroc_train_score, ddof=1) / np.sqrt(n_splits),
                    "+valid_acc": np.mean(pos_valid_accuracy),
                    "+valid_acc_stderr": np.std(pos_valid_accuracy, ddof=1) / np.sqrt(n_splits),
                    "-valid_acc": np.mean(neg_valid_accuracy),
                    "-valid_acc_stderr": np.std(neg_valid_accuracy, ddof=1) / np.sqrt(n_splits),
                    "f1_valid_avg": np.mean(f1_valid_score),
                    "f1_valid_avg_stderr": np.std(f1_valid_score, ddof=1) / np.sqrt(n_splits),
                    "auroc_valid_avg": np.mean(auroc_valid_score),
                    "auroc_valid_avg_stderr": np.std(auroc_valid_score, ddof=1) / np.sqrt(n_splits),
                    "precision": np.mean(precision_score_list),
                    "precision_stderr": np.std(precision_score_list, ddof=1) / np.sqrt(n_splits),
                }

                all_metrics.append(metrics)

                # logger.info(f"***Results svm classifier: C -> {c}  kernel -> {kernel} gamma -> {gamma} ***")
                # logger.info(f"SVM classifier score on positive train: {metrics['+train_acc']}")
                # logger.info(f"SVM classifier score on negative train: {metrics['-train_acc']}")
                # logger.info(f"SVM classifier score on positive valid: {metrics['+valid_acc']}")
                # logger.info(f"SVM classifier score on negative valid: {metrics['-valid_acc']}")

    metrics_df = pd.DataFrame(all_metrics).sort_values(by=["auroc_valid_avg"], ascending=False)
    logger.info("Performance table ...")
    logger.info(
        f"\n{metrics_df.loc[(metrics_df['+valid_acc'] >= 0.5) & (metrics_df['-valid_acc'] >= 0.5)].head(10)}"
    )

    all_metrics_path = output_path / "metrics_comparison.csv"
    metrics_df.to_csv(all_metrics_path, index=False)
    logger.info(f"Saved performance table to : {all_metrics_path}")


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--embeddings_model_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the embeddings model to fine-tune for classification .",
)
@click.option(
    "--training_dataset_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    multiple=True,
    help="Path to the training data. Can be given multiple times for a statistical analysis on different splits.",
)
@click.option(
    "--validation_dataset_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    multiple=True,
    help="Path to the validation data. Can be given multiple times for a statistical analysis on different splits.",
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output path to store metrics and model.",
)
@click.option(
    "--grid_search_parameters_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="Path to the json file containing a list of grid search hyperparameters to test.",
)
@click.option(
    "--save_models",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="Whether to save the trained models.",
)
@click.option(
    "--k_fold_cross_validation",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="Wheather to perfrom k-fold stratified cross validation. Works only if just one dataset is provided.",
)
def main(
    embeddings_model_path: Path,
    training_dataset_file: Union[Path, Tuple[Path]],
    validation_dataset_file: Union[Path, Tuple[Path]],
    output_path: Path,
    grid_search_parameters_file: Path,
    save_models: bool,
    k_fold_cross_validation: bool,
) -> None:
    """
    Fine-tune a pretrained model on a reaction classification task.
    """
    setup_console_logger()
    logger.info(f'Starting fine-tuning of model "{embeddings_model_path}".')

    logger.info(f'Loading the training data from "{training_dataset_file}"...')
    training_df = [load_jsonl_dataset_to_dataframe(f) for f in training_dataset_file]
    for i in range(len(training_df)):
        training_df[i]["text"] = training_df[i].apply(
            lambda x: get_reaction_from_precursors_and_product_smiles(x["source"], x["target"]),
            axis=1,
        )
        training_df[i] = training_df[i][["text", "score"]].rename({"score": "label"}, axis=1)
        print(training_df[i])
    logger.info(f'Loading the training data from "{training_dataset_file}"... Done.')

    logger.info(f'Loading the validation data from "{validation_dataset_file}"...')
    validation_df = [load_jsonl_dataset_to_dataframe(f) for f in validation_dataset_file]
    for i in range(len(validation_df)):
        validation_df[i]["text"] = validation_df[i].apply(
            lambda x: get_reaction_from_precursors_and_product_smiles(x["source"], x["target"]),
            axis=1,
        )
        validation_df[i] = validation_df[i][["text", "score"]].rename({"score": "label"}, axis=1)
    logger.info(f'Loading the validation data from "{validation_dataset_file}"... Done.')

    try:
        output_path.mkdir(parents=True, exist_ok=False)
        logger.info(f"Created output directory: {output_path}")
    except FileExistsError:
        logger.info("The output directory already exists and might not be empty ... Overwriting")

    svm_train(
        embeddings_model_path=embeddings_model_path,
        training_datasets=training_df,
        validation_datasets=validation_df,
        output_path=output_path,
        grid_search_parameters_file=grid_search_parameters_file,
        save_models=save_models,
        k_fold_cross_validation=k_fold_cross_validation,
    )


if __name__ == "__main__":
    main()
