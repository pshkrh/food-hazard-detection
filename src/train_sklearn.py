import logging
import os
import pickle
from datetime import datetime

from sklearn.metrics import f1_score, log_loss

from plotting_utils import plot_training_curves
from train_utils import TrainingMetrics


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
