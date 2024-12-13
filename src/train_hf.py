import logging
import os
from datetime import datetime

import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from plotting_utils import plot_training_curves
from train_utils import TrainingMetrics, EarlyStopping


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
