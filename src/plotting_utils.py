import os

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

from train_utils import sanitize_task_name


def plot_training_curves(metrics, task_name, output_dir):
    """
    Generates and saves training and validation loss and F1 score plots.

    This function creates line plots for training and validation losses, as well as F1 scores, over the epochs.
    The plots are saved as PNG files in the specified output directory.

    Parameters:
        metrics (train_utils.TrainingMetrics): The instance containing recorded training metrics.
        task_name (str): The name of the task for labeling the plots.
        output_dir (str): The directory where the plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics.train_losses, label="Training Loss")
    plt.plot(metrics.val_losses, label="Validation Loss")
    plt.title(f"{task_name} - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{sanitize_task_name(task_name)}_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(metrics.train_f1s, label="Training F1")
    plt.plot(metrics.val_f1s, label="Validation F1")
    plt.title(f"{task_name} - Training and Validation F1 Scores")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{sanitize_task_name(task_name)}_f1.png"))
    plt.close()
