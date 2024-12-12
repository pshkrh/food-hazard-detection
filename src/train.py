import nltk
import os
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import warnings
from sklearn.metrics import (
    f1_score,
    log_loss,
)
import pickle
from datetime import datetime
from nltk.corpus import stopwords
import zipfile
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from torch.optim import AdamW
import gc

from src.models import EnhancedClassifier, DNNClassifier, DANClassifier, CNNClassifier
from src.plotting_utils import plot_training_curves
from src.train_utils import compute_class_weights, EarlyStopping, TrainingMetrics, sanitize_task_name, \
    SingleLabelDataset, generate_predictions, text_to_ids, SimpleTextDataset

nltk.download("wordnet")
nltk.download("omw-1.4")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("training_debug.log"), logging.StreamHandler()],
)

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_title(input_title):
    """
    Cleans the input title by removing unwanted characters, patterns, and formatting to standardize it.

    This function performs multiple regex-based substitutions to eliminate dates, special characters, and specific patterns,
    ensuring that the title is in a clean and consistent format.

    Parameters:
        input_title (str): The title string to be cleaned.

    Returns:
        str: The cleaned title. If cleaning results in an empty string, the original title is returned.
    """
    orig_title = str(input_title)

    modified_title = re.sub(r"_", " ", input_title)
    modified_title = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "", modified_title)
    modified_title = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "", modified_title)
    modified_title = re.sub(
        r"\b\d{1,2}\s+\w+\s+\d{4}\b", "", modified_title, flags=re.IGNORECASE
    )
    modified_title = re.sub(r"\b\d{4}\b", "", modified_title)
    modified_title = re.sub(r"^\d{4}\s*", "", modified_title)
    modified_title = re.sub(r"\u00AE|®️", "", modified_title)
    modified_title = re.sub(r"&[a-zA-Z]+;", "", modified_title)
    modified_title = re.sub(
        r"^Recall Notification:?\s*", "", modified_title, flags=re.IGNORECASE
    )
    modified_title = re.sub(r"FSIS-\d+-\d+", "", modified_title)
    modified_title = re.sub(
        r"\bReport\s*\d+-?\s*", "", modified_title, flags=re.IGNORECASE
    )
    modified_title = re.sub(r"[()]", "", modified_title)
    modified_title = re.sub(r"-\s*$", "", modified_title)
    modified_title = re.sub(r"\s+-\s+", " ", modified_title)
    modified_title = re.sub(r"^:\s*", "", modified_title)
    modified_title = re.sub(r"\s*:\s*$", "", modified_title)
    modified_title = re.sub(r"\s+", " ", modified_title).strip()

    if modified_title.strip() == "":
        return orig_title
    else:
        return modified_title


def train_huggingface_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    task_name,
    num_epochs=10,
    learning_rate=2e-5,
    device="cuda",
    patience=3,
    gradient_accumulation_steps=1,
    output_dir="training_results",
):
    """
    Trains a PyTorch model using the provided training and validation data loaders.

    The function handles the training loop, including forward and backward passes, loss computation,
    gradient accumulation, optimizer steps, learning rate scheduling, and early stopping based on validation F1 score.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        class_weights (torch.FloatTensor): Tensor containing class weights for loss computation.
        task_name (str): The name of the task, used for logging and saving metrics.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
        device (str or torch.device, optional): Device to run the training on ('cuda' or 'cpu'). Defaults to 'cuda'.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 3.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before performing an optimizer step. Defaults to 1.
        output_dir (str, optional): Directory where training outputs (metrics, plots) will be saved. Defaults to 'training_results'.

    Returns:
        torch.nn.Module: The trained model with the best validation F1 score.
    """
    print(f"\nStarting training for: {task_name}")
    logging.info(f"\nStarting training for: {task_name}")

    model = model.to(device)
    metrics = TrainingMetrics()

    plots_dir = os.path.join(output_dir, "plots")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    total_steps = len(train_loader) // gradient_accumulation_steps * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    early_stopping = EarlyStopping(patience=patience)

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_val_f1 = 0
    best_state = None

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"{task_name} Epoch {epoch + 1}/{num_epochs}")
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps
            _, preds = torch.max(logits, 1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_preds.extend(preds.detach().cpu().numpy())
                val_labels_list.extend(labels.detach().cpu().numpy())

        tr_f1 = f1_score(train_labels, train_preds, average="macro")
        vl_f1 = f1_score(val_labels_list, val_preds, average="macro")
        tr_avg_loss = train_loss / len(train_loader)
        vl_avg_loss = val_loss / len(val_loader)

        epoch_time = (datetime.now() - epoch_start_time).total_seconds()

        metrics.update(tr_avg_loss, vl_avg_loss, tr_f1, vl_f1, epoch_time)

        print(
            f"{task_name} - Epoch {epoch + 1} | Training Loss: {tr_avg_loss:.4f} | Validation Loss: {vl_avg_loss:.4f} | Training F1: {tr_f1:.4f} | Validation F1: {vl_f1:.4f}"
        )
        logging.info(f"\n{task_name} - Epoch {epoch + 1}")
        logging.info(f"Training Loss: {tr_avg_loss:.4f}")
        logging.info(f"Validation Loss: {vl_avg_loss:.4f}")
        logging.info(f"Training F1: {tr_f1:.4f}")
        logging.info(f"Validation F1: {vl_f1:.4f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")

        plot_training_curves(metrics, task_name, plots_dir)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_state = model.state_dict().copy()

        early_stopping(vl_f1)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"\nSaving final metrics for: {task_name}")
    logging.info(f"\nSaving final metrics for: {task_name}")
    metrics.save_metrics(task_name, output_dir)

    if best_state is not None:
        model.load_state_dict(best_state)
        best_state = None

    return model


def train_sklearn_model(
    sk_model,
    train_X,
    train_y,
    val_X,
    val_y,
    classes_labels,
    task_label,
    out_dir,
    epochs_num=30,
):
    """
    Trains a scikit-learn model and tracks its performance over epochs.

    This function fits the model to the training data, evaluates it on validation data,
    updates training metrics, and saves the best-performing model based on validation F1 score.

    Parameters:
        sk_model (sklearn.base.BaseEstimator): The scikit-learn model to be trained.
        train_X (array-like): Feature matrix for training.
        train_y (array-like): Labels for training.
        val_X (array-like): Feature matrix for validation.
        val_y (array-like): Labels for validation.
        classes_labels (array-like): Array of class labels.
        task_label (str): The name of the task, used for logging and saving metrics.
        out_dir (str): Directory where training outputs (metrics, plots) will be saved.
        epochs_num (int, optional): Number of training epochs. Defaults to 30.

    Returns:
        sklearn.base.BaseEstimator: The trained scikit-learn model with the best validation F1 score.
    """
    print(f"Starting training for {task_label} model: {sk_model}")
    logging.info(f"Starting training for {task_label} model: {sk_model}")

    metrics = TrainingMetrics()
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    best_val_f1 = 0
    best_model = None

    for epoch in range(epochs_num):
        epoch_start_time = datetime.now()
        sk_model.fit(train_X, train_y)

        tr_preds = sk_model.predict(train_X)
        tr_f1 = f1_score(train_y, tr_preds, average="macro")
        tr_probs = sk_model.predict_proba(train_X)
        tr_loss = log_loss(train_y, tr_probs)

        vl_preds = sk_model.predict(val_X)
        vl_probs = sk_model.predict_proba(val_X)
        vl_f1 = f1_score(val_y, vl_preds, average="macro")
        vl_loss = log_loss(val_y, vl_probs)

        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        metrics.update(tr_loss, vl_loss, tr_f1, vl_f1, epoch_time)

        print(
            f"{task_label} - Epoch {epoch + 1} | Training Loss: {tr_loss:.4f} | Validation Loss: {vl_loss:.4f} | Training F1: {tr_f1:.4f} | Validation F1: {vl_f1:.4f}"
        )
        logging.info(f"\n{task_label} - Epoch {epoch + 1}")
        logging.info(f"Training Loss: {tr_loss:.4f}")
        logging.info(f"Validation Loss: {vl_loss:.4f}")
        logging.info(f"Training F1: {tr_f1:.4f}")
        logging.info(f"Validation F1: {vl_f1:.4f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")

        plot_training_curves(metrics, task_label, plots_dir)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_model = pickle.dumps(sk_model)

    if best_model is not None:
        sk_model = pickle.loads(best_model)
        best_model = None

    metrics.save_metrics(task_label, out_dir)
    return sk_model


def build_vocab(input_texts, vocab_max_size=20000):
    """
    Builds a vocabulary dictionary from the input texts based on word frequency.

    The vocabulary includes the most frequent words up to the specified maximum size,
    excluding stop words. Special tokens for padding and unknown words are also included.

    Parameters:
        input_texts (list or array-like): A list of text samples.
        vocab_max_size (int, optional): The maximum size of the vocabulary. Defaults to 20000.

    Returns:
        dict: A dictionary mapping words to unique integer IDs.
    """
    word_frequency = {}
    for txt in input_texts:
        for w in txt.lower().split():
            if w not in stop_words:
                word_frequency[w] = word_frequency.get(w, 0) + 1
    sorted_word_freq = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for i, (word, frequency) in enumerate(
        sorted_word_freq[: vocab_max_size - 2], start=2
    ):
        vocabulary[word] = i
    return vocabulary


def collate_fn_simple(batch_list):
    """
    Collates a list of samples into batched tensors.

    This function stacks the 'input_ids' and 'labels' from each sample into separate tensors.

    Parameters:
        batch_list (list): A list of dictionaries containing 'input_ids' and 'labels'.

    Returns:
        tuple: A tuple containing two tensors: (input_ids, labels).
    """
    input_ids = torch.stack([b["input_ids"] for b in batch_list])
    labels = torch.stack([b["labels"] for b in batch_list])
    return (input_ids, labels)


def train_pytorch_model(
    pt_model,
    train_loader_obj,
    val_loader_obj,
    task_label,
    dev,
    n_labels,
    epochs_num=30,
    lr_val=2e-5,
    patience_limit=7,
    out_dir="training_results",
):
    """
    Trains a PyTorch model using the provided training and validation data loaders.

    The function handles the training loop, including forward and backward passes, loss computation,
    gradient clipping, optimizer steps, and early stopping based on validation F1 score.

    Parameters:
        pt_model (torch.nn.Module): The PyTorch model to be trained.
        train_loader_obj (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader_obj (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        task_label (str): The name of the task, used for logging and saving metrics.
        dev (torch.device): The device (CPU or GPU) on which computations are performed.
        n_labels (int): The number of output labels/classes for classification.
        epochs_num (int, optional): Number of training epochs. Defaults to 30.
        lr_val (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
        patience_limit (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 7.
        out_dir (str, optional): Directory where training outputs (metrics, plots) will be saved. Defaults to 'training_results'.

    Returns:
        torch.nn.Module: The trained PyTorch model with the best validation F1 score.
    """
    print(f"Starting training for {task_label} model: {pt_model}")
    logging.info(f"Starting training for {task_label} model: {pt_model}")

    metrics = TrainingMetrics()
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    optimizer = AdamW(pt_model.parameters(), lr=lr_val)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience_limit)

    pt_model = pt_model.to(dev)

    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs_num):
        epoch_start_time = datetime.now()
        pt_model.train()
        tr_loss_sum = 0
        tr_preds = []
        tr_labels = []

        for batch in tqdm(
            train_loader_obj, desc=f"{task_label} Epoch {epoch + 1}/{epochs_num}"
        ):
            input_ids, labels = batch
            input_ids = input_ids.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            logits = pt_model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pt_model.parameters(), 1.0)
            optimizer.step()

            tr_loss_sum += loss.item()
            _, preds = torch.max(logits, 1)
            tr_preds.extend(preds.detach().cpu().numpy())
            tr_labels.extend(labels.detach().cpu().numpy())

        pt_model.eval()
        vl_loss_sum = 0
        vl_preds = []
        vl_labels_list = []

        with torch.no_grad():
            for batch in val_loader_obj:
                input_ids, labels = batch
                input_ids = input_ids.to(dev)
                labels = labels.to(dev)
                logits = pt_model(input_ids)
                loss = criterion(logits, labels)

                vl_loss_sum += loss.item()
                _, preds = torch.max(logits, 1)
                vl_preds.extend(preds.detach().cpu().numpy())
                vl_labels_list.extend(labels.detach().cpu().numpy())

        tr_f1 = f1_score(tr_labels, tr_preds, average="macro")
        vl_f1 = f1_score(vl_labels_list, vl_preds, average="macro")
        tr_avg_loss = tr_loss_sum / len(train_loader_obj)
        vl_avg_loss = vl_loss_sum / len(val_loader_obj)

        epoch_time = (datetime.now() - epoch_start_time).total_seconds()

        metrics.update(tr_avg_loss, vl_avg_loss, tr_f1, vl_f1, epoch_time)

        print(
            f"{task_label} - Epoch {epoch + 1} | Training Loss: {tr_avg_loss:.4f} | Validation Loss: {vl_avg_loss:.4f} | Training F1: {tr_f1:.4f} | Validation F1: {vl_f1:.4f}"
        )
        logging.info(f"\n{task_label} - Epoch {epoch + 1}")
        logging.info(f"Training Loss: {tr_avg_loss:.4f}")
        logging.info(f"Validation Loss: {vl_avg_loss:.4f}")
        logging.info(f"Training F1: {tr_f1:.4f}")
        logging.info(f"Validation F1: {vl_f1:.4f}")
        logging.info(f"Epoch Time: {epoch_time:.2f}s")

        plot_training_curves(metrics, task_label, plots_dir)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_state = pt_model.state_dict().copy()

        early_stopping(vl_f1)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"\nSaving final metrics for: {task_label}")
    logging.info(f"\nSaving final metrics for: {task_label}")
    metrics.save_metrics(task_label, out_dir)

    if best_state is not None:
        pt_model.load_state_dict(best_state)
        best_state = None

    return pt_model


def main(
    model_name="bert-base-uncased",
    subtask=1,
    epochs_num=20,
    batch_sz=16,
    lr_val=2e-5,
    max_len=32,
    grad_acc_steps=1,
    use_stratify=True,
    syn_data_suffix="25",
    freeze_until_layers=0,
    is_multi_task=False,
):
    """
    The main function orchestrates the training process for various models and subtasks.

    It handles data loading, preprocessing, model instantiation, training, evaluation, and submission file generation
    based on the specified model and subtask.

    Parameters:
        model_name (str, optional): The name of the model to train. Defaults to 'bert-base-uncased'.
        subtask (int, optional): The subtask identifier (1 or 2). Defaults to 1.
        epochs_num (int, optional): Number of training epochs. Defaults to 20.
        batch_sz (int, optional): Batch size for training and evaluation. Defaults to 16.
        lr_val (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
        max_len (int, optional): Maximum sequence length for tokenization. Defaults to 32.
        grad_acc_steps (int, optional): Number of gradient accumulation steps. Defaults to 1.
        use_stratify (bool, optional): Whether to use stratified splitting. Defaults to True.
        syn_data_suffix (str, optional): Suffix for synthetic data filenames. Defaults to '25'.
        freeze_until_layers (int, optional): Number of layers to freeze in the model. Defaults to 0.
        is_multi_task (bool, optional): Whether to perform multi-task learning. Defaults to False.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    print(f"Using device: {device}")

    model_safe = sanitize_task_name(model_name)
    subtask_safe = f"st{subtask}"
    run_output_dir = f"training-result-{model_safe}/{subtask_safe}"
    os.makedirs(run_output_dir, exist_ok=True)

    plots_dir = os.path.join(run_output_dir, "plots")
    metrics_dir = os.path.join(run_output_dir, "metrics")
    models_dir = os.path.join(run_output_dir, "models")
    submission_dir = os.path.join(run_output_dir, "submission")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    if subtask == 1:
        hazard_df = pd.read_csv(f"hazard_category_data_{syn_data_suffix}.csv")
        product_df = pd.read_csv(f"product_category_data_{syn_data_suffix}.csv")
        hazard_col = (
            "hazard-category" if "hazard-category" in hazard_df.columns else "hazard"
        )
        product_col = (
            "product-category"
            if "product-category" in product_df.columns
            else "product"
        )
        output_prefix = "st1"
        hazard_task_name = "ST1 Hazard Category Classification"
        product_task_name = "ST1 Product Category Classification"
    else:
        hazard_df = pd.read_csv(f"hazard_data_{syn_data_suffix}.csv")
        product_df = pd.read_csv(f"product_data_{syn_data_suffix}.csv")
        hazard_col = "hazard"
        product_col = "product"
        output_prefix = "st2"
        hazard_task_name = "ST2 Hazard Vector Classification"
        product_task_name = "ST2 Product Vector Classification"

    logging.info(
        f"\nStarting {output_prefix.upper()} training for model {model_name}..."
    )
    logging.info(f"Hazard task: {hazard_task_name}")
    logging.info(f"Product task: {product_task_name}")
    logging.info(f"Output directory: {run_output_dir}")
    print(f"\nStarting {output_prefix.upper()} training for model {model_name}...")
    print(f"Hazard task: {hazard_task_name}")
    print(f"Product task: {product_task_name}")
    print(f"Output directory: {run_output_dir}")

    haz_enc = LabelEncoder()
    prod_enc = LabelEncoder()

    hazard_df["label"] = haz_enc.fit_transform(hazard_df[hazard_col])
    product_df["label"] = prod_enc.fit_transform(product_df[product_col])

    encoder_file = os.path.join(metrics_dir, f"{output_prefix}_encoders.pkl")
    with open(encoder_file, "wb") as file_handle:
        pickle.dump({"haz_enc": haz_enc, "prod_enc": prod_enc}, file_handle)

    hazard_train, hazard_val = train_test_split(
        hazard_df,
        test_size=0.1,
        random_state=42,
        stratify=hazard_df[hazard_col] if use_stratify else None,
    )
    product_train, product_val = train_test_split(
        product_df,
        test_size=0.1,
        random_state=42,
        stratify=product_df[product_col] if use_stratify else None,
    )

    hazard_val = hazard_val.copy()
    hazard_val["cleaned_title"] = hazard_val["title"].apply(clean_title)

    product_val = product_val.copy()
    product_val["cleaned_title"] = product_val["title"].apply(clean_title)

    if model_name in [
        "facebook/bart-base",
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-base",
        "tuned_bert",
    ]:
        if model_name == "tuned_bert":
            base_model_name = "bert-base-uncased"
        else:
            base_model_name = model_name

        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased" if model_name == "tuned_bert" else model_name
        )

        hazard_train_dataset = SingleLabelDataset(
            texts=hazard_train["title"].values,
            labels=hazard_train["label"].values,
            tokenizer=tokenizer,
            max_length=max_len,
        )
        hazard_val_dataset = SingleLabelDataset(
            texts=hazard_val["cleaned_title"].values,
            labels=hazard_val["label"].values,
            tokenizer=tokenizer,
            max_length=max_len,
        )
        product_train_dataset = SingleLabelDataset(
            texts=product_train["title"].values,
            labels=product_train["label"].values,
            tokenizer=tokenizer,
            max_length=max_len,
        )
        product_val_dataset = SingleLabelDataset(
            texts=product_val["cleaned_title"].values,
            labels=product_val["label"].values,
            tokenizer=tokenizer,
            max_length=max_len,
        )

        hazard_train_loader = DataLoader(
            hazard_train_dataset, batch_size=batch_sz, shuffle=True
        )
        hazard_val_loader = DataLoader(hazard_val_dataset, batch_size=batch_sz)
        product_train_loader = DataLoader(
            product_train_dataset, batch_size=batch_sz, shuffle=True
        )
        product_val_loader = DataLoader(product_val_dataset, batch_size=batch_sz)

        weights_class_hazard = compute_class_weights(hazard_train["label"].values)
        weights_class_product = compute_class_weights(product_train["label"].values)

        hazard_model = EnhancedClassifier(
            base_model_name, num_labels=len(haz_enc.classes_)
        )
        product_model = EnhancedClassifier(
            base_model_name, num_labels=len(prod_enc.classes_)
        )

        logging.info(f"\nTraining {hazard_task_name}...")
        print(f"Training {hazard_task_name} now...")
        hazard_model = train_huggingface_model(
            model=hazard_model,
            train_loader=hazard_train_loader,
            val_loader=hazard_val_loader,
            class_weights=weights_class_hazard,
            task_name=hazard_task_name,
            num_epochs=epochs_num,
            learning_rate=lr_val,
            device=device,
            patience=7,
            gradient_accumulation_steps=grad_acc_steps,
            output_dir=run_output_dir,
        )

        logging.info(f"\nTraining {product_task_name}...")
        print(f"Training {product_task_name} now...")
        product_model = train_huggingface_model(
            model=product_model,
            train_loader=product_train_loader,
            val_loader=product_val_loader,
            class_weights=weights_class_product,
            task_name=product_task_name,
            num_epochs=epochs_num,
            learning_rate=lr_val,
            device=device,
            patience=7,
            gradient_accumulation_steps=grad_acc_steps,
            output_dir=run_output_dir,
        )

        torch.save(
            hazard_model.state_dict(),
            os.path.join(models_dir, f"best_{output_prefix}_hazard_model.pt"),
        )
        torch.save(
            product_model.state_dict(),
            os.path.join(models_dir, f"best_{output_prefix}_product_model.pt"),
        )

        if os.path.exists("incidents_unlabeled_val.csv"):
            val_df = pd.read_csv("incidents_unlabeled_val.csv")
            val_df["cleaned_title"] = val_df["title"].apply(clean_title)

            val_dataset = SingleLabelDataset(
                texts=val_df["cleaned_title"].values,
                labels=None,
                tokenizer=tokenizer,
                max_length=max_len,
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_sz)

            hazard_model = hazard_model.to(device)
            product_model = product_model.to(device)

            print("\nGenerating predictions...")
            logging.info("\nGenerating predictions...")
            hazard_preds = generate_predictions(
                hazard_model, val_loader, device, task_head="hazard"
            )
            product_preds = generate_predictions(
                product_model, val_loader, device, task_head="product"
            )

            hazard_preds_str = haz_enc.inverse_transform(hazard_preds)
            product_preds_str = prod_enc.inverse_transform(product_preds)

            submission_df = pd.DataFrame(
                {hazard_col: hazard_preds_str, product_col: product_preds_str}
            )

            submission_path = os.path.join(submission_dir, "submission.csv")
            submission_df.to_csv(submission_path, index=False)
            logging.info(f"\nSubmission file saved to {submission_path}")
            print(f"\nSubmission file saved to {submission_path}")

            zip_path = os.path.join(
                run_output_dir, f"submission-{model_safe}-{output_prefix}.zip"
            )
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(submission_path, arcname="submission.csv")

    elif model_name == "tfidf_logistic_regression":
        hazard_tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
        product_tfidf = TfidfVectorizer(max_features=20000, stop_words="english")

        hazard_train_texts = hazard_train["title"].apply(clean_title).values
        hazard_val_texts = hazard_val["cleaned_title"].values
        product_train_texts = product_train["title"].apply(clean_title).values
        product_val_texts = product_val["cleaned_title"].values

        X_hazard_train = hazard_tfidf.fit_transform(hazard_train_texts)
        X_hazard_val = hazard_tfidf.transform(hazard_val_texts)
        y_hazard_train = hazard_train["label"].values
        y_hazard_val = hazard_val["label"].values

        hazard_lr = LogisticRegression(max_iter=1000)
        hazard_lr = train_sklearn_model(
            hazard_lr,
            X_hazard_train,
            y_hazard_train,
            X_hazard_val,
            y_hazard_val,
            haz_enc.classes_,
            hazard_task_name,
            run_output_dir,
            epochs_num=1,
        )

        X_product_train = product_tfidf.fit_transform(product_train_texts)
        X_product_val = product_tfidf.transform(product_val_texts)
        y_product_train = product_train["label"].values
        y_product_val = product_val["label"].values

        product_lr = LogisticRegression(max_iter=1000)
        product_lr = train_sklearn_model(
            product_lr,
            X_product_train,
            y_product_train,
            X_product_val,
            y_product_val,
            prod_enc.classes_,
            product_task_name,
            run_output_dir,
            epochs_num=1,
        )

        if os.path.exists("incidents_unlabeled_val.csv"):
            val_df = pd.read_csv("incidents_unlabeled_val.csv")
            val_df["clean_title"] = val_df["title"].apply(clean_title)

            X_unlabeled_hazard = hazard_tfidf.transform(val_df["clean_title"].values)
            X_unlabeled_product = product_tfidf.transform(val_df["clean_title"].values)

            hazard_preds = hazard_lr.predict(X_unlabeled_hazard)
            product_preds = product_lr.predict(X_unlabeled_product)

            hazard_preds_str = haz_enc.inverse_transform(hazard_preds)
            product_preds_str = prod_enc.inverse_transform(product_preds)

            submission_df = pd.DataFrame(
                {hazard_col: hazard_preds_str, product_col: product_preds_str}
            )

            submission_path = os.path.join(submission_dir, "submission.csv")
            submission_df.to_csv(submission_path, index=False)

            zip_path = os.path.join(
                run_output_dir, f"submission-{model_safe}-{output_prefix}.zip"
            )
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(submission_path, arcname="submission.csv")

    elif model_name == "xgboost":
        hazard_tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
        product_tfidf = TfidfVectorizer(max_features=20000, stop_words="english")

        hazard_train_texts = hazard_train["title"].apply(clean_title).values
        hazard_val_texts = hazard_val["cleaned_title"].values
        product_train_texts = product_train["title"].apply(clean_title).values
        product_val_texts = product_val["cleaned_title"].values

        X_hazard_train = hazard_tfidf.fit_transform(hazard_train_texts)
        X_hazard_val = hazard_tfidf.transform(hazard_val_texts)
        y_hazard_train = hazard_train["label"].values
        y_hazard_val = hazard_val["label"].values

        hazard_xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        hazard_xgb = train_sklearn_model(
            hazard_xgb,
            X_hazard_train,
            y_hazard_train,
            X_hazard_val,
            y_hazard_val,
            haz_enc.classes_,
            hazard_task_name,
            run_output_dir,
            epochs_num=1,
        )

        X_product_train = product_tfidf.fit_transform(product_train_texts)
        X_product_val = product_tfidf.transform(product_val_texts)
        y_product_train = product_train["label"].values
        y_product_val = product_val["label"].values

        product_xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        product_xgb = train_sklearn_model(
            product_xgb,
            X_product_train,
            y_product_train,
            X_product_val,
            y_product_val,
            prod_enc.classes_,
            product_task_name,
            run_output_dir,
            epochs_num=1,
        )

        if os.path.exists("incidents_unlabeled_val.csv"):
            val_df = pd.read_csv("incidents_unlabeled_val.csv")
            val_df["clean_title"] = val_df["title"].apply(clean_title)

            X_unlabeled_hazard = hazard_tfidf.transform(val_df["clean_title"].values)
            X_unlabeled_product = product_tfidf.transform(val_df["clean_title"].values)

            hazard_preds = hazard_xgb.predict(X_unlabeled_hazard)
            product_preds = product_xgb.predict(X_unlabeled_product)

            hazard_preds_str = haz_enc.inverse_transform(hazard_preds)
            product_preds_str = prod_enc.inverse_transform(product_preds)

            submission_df = pd.DataFrame(
                {hazard_col: hazard_preds_str, product_col: product_preds_str}
            )

            submission_path = os.path.join(submission_dir, "submission.csv")
            submission_df.to_csv(submission_path, index=False)

            zip_path = os.path.join(
                run_output_dir, f"submission-{model_safe}-{output_prefix}.zip"
            )
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(submission_path, arcname="submission.csv")

    elif model_name == "random_forest":
        hazard_tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
        product_tfidf = TfidfVectorizer(max_features=20000, stop_words="english")

        hazard_train_texts = hazard_train["title"].apply(clean_title).values
        hazard_val_texts = hazard_val["cleaned_title"].values
        product_train_texts = product_train["title"].apply(clean_title).values
        product_val_texts = product_val["cleaned_title"].values

        X_hazard_train = hazard_tfidf.fit_transform(hazard_train_texts)
        X_hazard_val = hazard_tfidf.transform(hazard_val_texts)
        y_hazard_train = hazard_train["label"].values
        y_hazard_val = hazard_val["label"].values

        hazard_rf = RandomForestClassifier(n_estimators=100)
        hazard_rf = train_sklearn_model(
            hazard_rf,
            X_hazard_train,
            y_hazard_train,
            X_hazard_val,
            y_hazard_val,
            haz_enc.classes_,
            hazard_task_name,
            run_output_dir,
            epochs_num=1,
        )

        X_product_train = product_tfidf.fit_transform(product_train_texts)
        X_product_val = product_tfidf.transform(product_val_texts)
        y_product_train = product_train["label"].values
        y_product_val = product_val["label"].values

        product_rf = RandomForestClassifier(n_estimators=100)
        product_rf = train_sklearn_model(
            product_rf,
            X_product_train,
            y_product_train,
            X_product_val,
            y_product_val,
            prod_enc.classes_,
            product_task_name,
            run_output_dir,
            epochs_num=1,
        )

        if os.path.exists("incidents_unlabeled_val.csv"):
            val_df = pd.read_csv("incidents_unlabeled_val.csv")
            val_df["clean_title"] = val_df["title"].apply(clean_title)

            X_unlabeled_hazard = hazard_tfidf.transform(val_df["clean_title"].values)
            X_unlabeled_product = product_tfidf.transform(val_df["clean_title"].values)

            hazard_preds = hazard_rf.predict(X_unlabeled_hazard)
            product_preds = product_rf.predict(X_unlabeled_product)

            hazard_preds_str = haz_enc.inverse_transform(hazard_preds)
            product_preds_str = prod_enc.inverse_transform(product_preds)

            submission_df = pd.DataFrame(
                {hazard_col: hazard_preds_str, product_col: product_preds_str}
            )

            submission_path = os.path.join(submission_dir, "submission.csv")
            submission_df.to_csv(submission_path, index=False)

            zip_path = os.path.join(
                run_output_dir, f"submission-{model_safe}-{output_prefix}.zip"
            )
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(submission_path, arcname="submission.csv")

    elif model_name in ["dnn", "dan", "cnn"]:
        hazard_train_texts = hazard_train["title"].apply(clean_title).values
        hazard_val_texts = hazard_val["cleaned_title"].values
        product_train_texts = product_train["title"].apply(clean_title).values
        product_val_texts = product_val["cleaned_title"].values

        hazard_vocab = build_vocab(hazard_train_texts)
        product_vocab = build_vocab(product_train_texts)

        hazard_train_ds = SimpleTextDataset(
            hazard_train_texts, hazard_train["label"].values, hazard_vocab, max_len
        )
        hazard_val_ds = SimpleTextDataset(
            hazard_val_texts, hazard_val["label"].values, hazard_vocab, max_len
        )
        product_train_ds = SimpleTextDataset(
            product_train_texts, product_train["label"].values, product_vocab, max_len
        )
        product_val_ds = SimpleTextDataset(
            product_val_texts, product_val["label"].values, product_vocab, max_len
        )

        hazard_train_loader = DataLoader(
            hazard_train_ds,
            batch_size=batch_sz,
            shuffle=True,
            collate_fn=collate_fn_simple,
        )
        hazard_val_loader = DataLoader(
            hazard_val_ds, batch_size=batch_sz, collate_fn=collate_fn_simple
        )
        product_train_loader = DataLoader(
            product_train_ds,
            batch_size=batch_sz,
            shuffle=True,
            collate_fn=collate_fn_simple,
        )
        product_val_loader = DataLoader(
            product_val_ds, batch_size=batch_sz, collate_fn=collate_fn_simple
        )

        if model_name == "dnn":
            hazard_model = DNNClassifier(len(hazard_vocab), len(haz_enc.classes_))
            product_model = DNNClassifier(len(product_vocab), len(prod_enc.classes_))
        elif model_name == "dan":
            hazard_model = DANClassifier(len(hazard_vocab), len(haz_enc.classes_))
            product_model = DANClassifier(len(product_vocab), len(prod_enc.classes_))
        else:
            hazard_model = CNNClassifier(len(hazard_vocab), len(haz_enc.classes_))
            product_model = CNNClassifier(len(product_vocab), len(prod_enc.classes_))

        hazard_model = train_pytorch_model(
            hazard_model,
            hazard_train_loader,
            hazard_val_loader,
            hazard_task_name,
            device,
            len(haz_enc.classes_),
            num_epochs=epochs_num,
            lr_val=lr_val,
            patience_limit=7,
            out_dir=run_output_dir,
        )
        product_model = train_pytorch_model(
            product_model,
            product_train_loader,
            product_val_loader,
            product_task_name,
            device,
            len(prod_enc.classes_),
            num_epochs=epochs_num,
            lr_val=lr_val,
            patience_limit=7,
            out_dir=run_output_dir,
        )

        if os.path.exists("incidents_unlabeled_val.csv"):
            val_df = pd.read_csv("incidents_unlabeled_val.csv")
            val_df["clean_title"] = val_df["title"].apply(clean_title)

            X_unlabeled_h = [
                text_to_ids(t, hazard_vocab, max_len)
                for t in val_df["clean_title"].values
            ]
            hazard_model.eval()
            hazard_preds = []
            with torch.no_grad():
                for i in range(0, len(X_unlabeled_h), batch_sz):
                    batch_ids = torch.tensor(
                        X_unlabeled_h[i : i + batch_sz], dtype=torch.long
                    ).to(device)
                    logits = hazard_model(batch_ids)
                    _, preds = torch.max(logits, 1)
                    hazard_preds.extend(preds.cpu().numpy())
            hazard_preds_str = haz_enc.inverse_transform(hazard_preds)

            X_unlabeled_p = [
                text_to_ids(t, product_vocab, max_len)
                for t in val_df["clean_title"].values
            ]
            product_model.eval()
            product_preds = []
            with torch.no_grad():
                for i in range(0, len(X_unlabeled_p), batch_sz):
                    batch_ids = torch.tensor(
                        X_unlabeled_p[i : i + batch_sz], dtype=torch.long
                    ).to(device)
                    logits = product_model(batch_ids)
                    _, preds = torch.max(logits, 1)
                    product_preds.extend(preds.cpu().numpy())
            product_preds_str = prod_enc.inverse_transform(product_preds)

            submission_df = pd.DataFrame(
                {hazard_col: hazard_preds_str, product_col: product_preds_str}
            )

            submission_path = os.path.join(submission_dir, "submission.csv")
            submission_df.to_csv(submission_path, index=False)

            zip_path = os.path.join(
                run_output_dir, f"submission-{model_safe}-{output_prefix}.zip"
            )
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(submission_path, arcname="submission.csv")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    models = [
        "tuned_bert",
        "facebook/bart-base",
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-base",
        "tfidf_logistic_regression",
        "xgboost",
    ]
    subtasks = [1, 2]

    for model in models:
        for subtask in subtasks:
            main(
                model_name=model,
                subtask=subtask,
                epochs_num=2,
                batch_sz=32,
                lr_val=2e-5,
                max_len=64,
                grad_acc_steps=1,
                use_stratify=True,
                syn_data_suffix="25",
                freeze_until_layers=0,
                is_multi_task=False,
            )
