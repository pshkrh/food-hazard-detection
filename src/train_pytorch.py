import logging
import os
from datetime import datetime

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from plotting_utils import plot_training_curves
from train_utils import TrainingMetrics, EarlyStopping


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
