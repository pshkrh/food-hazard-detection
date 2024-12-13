import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def compute_class_weights(labels):
    """
    Calculates class weights to handle class imbalance in the dataset.

    The weights are computed based on the frequency of each class, applying a smoothing factor to prevent extreme weights.

    Parameters:
        labels (array-like): An array of class labels.

    Returns:
        torch.FloatTensor: A tensor containing the weight for each class.
    """
    class_counts = np.bincount(labels)
    total = len(labels)
    smoothing_factor = 0.1
    class_weights = total / (
        len(class_counts) * (class_counts + smoothing_factor * total)
    )
    return torch.FloatTensor(class_weights)


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation performance stops improving.

    This class monitors a specified metric and stops the training process if there's no improvement
    after a certain number of consecutive epochs (patience). It helps prevent overfitting and saves computational resources.
    """

    def __init__(self, patience=3, min_delta=0):
        """
        Initializes the EarlyStopping instance.

        Parameters:
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 3.
            min_delta (float, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        """
        Checks if the validation score has improved and updates internal counters.

        Parameters:
            val_score (float): The current epoch's validation score.
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


class TrainingMetrics:
    """
    Tracks and records training and validation metrics over epochs.

    This class collects metrics such as loss and F1 scores for both training and validation phases,
    along with timestamps and epoch durations. It also identifies the best validation F1 score and the corresponding epoch.
    """

    def __init__(self):
        """
        Initializes the TrainingMetrics instance with empty lists and default best metrics.
        """
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.timestamps = []
        self.epoch_times = []
        self.best_val_f1 = 0
        self.best_epoch = 0

    def update(self, train_loss, val_loss, train_f1, val_f1, epoch_time=None):
        """
        Updates the metrics with the latest epoch's results.

        Parameters:
            train_loss (float): The average training loss for the epoch.
            val_loss (float): The average validation loss for the epoch.
            train_f1 (float): The training F1 score for the epoch.
            val_f1 (float): The validation F1 score for the epoch.
            epoch_time (float, optional): The duration of the epoch in seconds. Defaults to None.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_f1s.append(train_f1)
        self.val_f1s.append(val_f1)
        self.timestamps.append(datetime.now())
        if epoch_time:
            self.epoch_times.append(epoch_time)

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_epoch = len(self.train_losses) - 1

    def save_metrics(self, task_name, output_dir):
        """
        Saves the recorded metrics to CSV and pickle files, and logs the saving process.

        Parameters:
            task_name (str): The name of the task for which metrics are saved.
            output_dir (str): The directory where metrics and plots will be saved.
        """
        metrics_dir = os.path.join(output_dir, "metrics")
        plots_dir = os.path.join(output_dir, "plots")

        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        safe_task_name = sanitize_task_name(task_name)

        metrics_df = pd.DataFrame(
            {
                "epoch": range(1, len(self.train_losses) + 1),
                "timestamp": self.timestamps,
                "train_loss": self.train_losses,
                "val_loss": self.val_losses,
                "train_f1": self.train_f1s,
                "val_f1": self.val_f1s,
                "epoch_time": (
                    self.epoch_times
                    if self.epoch_times
                    else [None] * len(self.train_losses)
                ),
            }
        )

        summary_df = pd.DataFrame(
            {
                "metric": [
                    "best_val_f1",
                    "best_epoch",
                    "total_epochs",
                    "total_training_time",
                ],
                "value": [
                    self.best_val_f1,
                    self.best_epoch + 1,
                    len(self.train_losses),
                    sum(self.epoch_times) if self.epoch_times else None,
                ],
            }
        )

        metrics_filename = f"{safe_task_name}_metrics.csv"
        csv_path = os.path.join(metrics_dir, metrics_filename)
        metrics_df.to_csv(csv_path, index=False)

        summary_filename = f"{safe_task_name}_summary.csv"
        summary_path = os.path.join(metrics_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)

        pickle_filename = f"{safe_task_name}_metrics.pkl"
        pickle_path = os.path.join(metrics_dir, pickle_filename)
        with open(pickle_path, "wb") as f:
            pickle.dump(
                {
                    "task_name": task_name,
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "train_f1s": self.train_f1s,
                    "val_f1s": self.val_f1s,
                    "timestamps": self.timestamps,
                    "epoch_times": self.epoch_times,
                    "best_val_f1": self.best_val_f1,
                    "best_epoch": self.best_epoch,
                },
                f,
            )

        logging.info(f"Metrics saved for {task_name}:")
        logging.info(f"- Detailed metrics: {metrics_filename}")
        logging.info(f"- Summary metrics: {summary_filename}")
        logging.info(f"- Pickle format: {pickle_filename}")


def sanitize_task_name(task_name):
    """
    Sanitizes the task name by converting it to lowercase and replacing spaces and special characters with underscores.

    This function ensures that the task name is filesystem-friendly, making it suitable for use in filenames and directories.

    Parameters:
        task_name (str): The original task name.

    Returns:
        str: The sanitized task name.
    """
    return task_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


class SingleLabelDataset(Dataset):
    """
    A custom PyTorch Dataset for handling single-label text classification tasks.

    This dataset takes in texts and their corresponding labels, tokenizes the texts using a provided tokenizer,
    and returns input tensors suitable for model training and evaluation.
    """

    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        """
        Initializes the SingleLabelDataset with texts, labels, tokenizer, and maximum sequence length.

        Parameters:
            texts (list or array-like): A list of text samples.
            labels (list or array-like, optional): A list of labels corresponding to the texts. Defaults to None.
            tokenizer (transformers.PreTrainedTokenizer, optional): A tokenizer instance from HuggingFace Transformers.
                Defaults to None.
            max_length (int, optional): The maximum sequence length for tokenization. Defaults to 128.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of text samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized input and label for a given index.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels' (if available).
        """
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def generate_predictions(model, data_loader, device, task_head="hazard"):
    """
    Generates predictions for a given model and data loader.

    This function iterates over the data loader, performs a forward pass using the model,
    and collects the predicted class labels.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model for making predictions.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the data samples.
        device (torch.device): The device (CPU or GPU) on which computations are performed.
        task_head (str, optional): A string indicating the task type. Defaults to 'hazard'.

    Returns:
        numpy.ndarray: An array of predicted class labels.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.detach().cpu().numpy())
    return np.array(predictions)


def text_to_ids(input_text, vocabulary, max_len=128):
    """
    Converts input text into a list of token IDs based on the provided vocabulary.

    Words not found in the vocabulary are replaced with the ID for unknown tokens.
    The resulting list is padded or truncated to match the specified maximum length.

    Parameters:
        input_text (str): The text to be converted.
        vocabulary (dict): A dictionary mapping words to their corresponding IDs.
        max_len (int, optional): The desired length of the output list. Defaults to 128.

    Returns:
        list: A list of integer token IDs representing the input text.
    """
    word_list = input_text.lower().split()
    word_ids = [vocabulary.get(w, vocabulary["<unk>"]) for w in word_list]
    if len(word_ids) < max_len:
        word_ids += [vocabulary["<pad>"]] * (max_len - len(word_ids))
    else:
        word_ids = word_ids[:max_len]
    return word_ids


class SimpleTextDataset(Dataset):
    """
    A simple PyTorch Dataset for text classification tasks using a custom vocabulary.

    This dataset converts texts into sequences of token IDs based on a provided vocabulary
    and pairs them with their corresponding labels.
    """

    def __init__(self, input_texts, target_labels, vocabulary, max_len=128):
        """
        Initializes the SimpleTextDataset with input texts, target labels, vocabulary, and maximum sequence length.

        Parameters:
            input_texts (list or array-like): A list of text samples.
            target_labels (list or array-like): A list of labels corresponding to the texts.
            vocabulary (dict): A dictionary mapping words to unique integer IDs.
            max_len (int, optional): The maximum sequence length. Defaults to 128.
        """
        self.input_texts = input_texts
        self.target_labels = target_labels
        self.vocabulary = vocabulary
        self.max_len = max_len

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of text samples.
        """
        return len(self.input_texts)

    def __getitem__(self, index):
        """
        Retrieves the tokenized input and label for a given index.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids' and 'labels'.
        """
        token_id = text_to_ids(self.input_texts[index], self.vocabulary, self.max_len)
        return {
            "input_ids": torch.tensor(token_id, dtype=torch.long),
            "labels": torch.tensor(self.target_labels[index], dtype=torch.long),
        }
