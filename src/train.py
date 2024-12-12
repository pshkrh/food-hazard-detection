import nltk
import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import pickle
import zipfile
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import gc
import argparse

from src.data_utils import clean_title, build_vocab, collate_fn_simple
from src.models import EnhancedClassifier, DNNClassifier, DANClassifier, CNNClassifier
from src.train_hf import train_huggingface_model
from src.train_pytorch import train_pytorch_model
from src.train_sklearn import train_sklearn_model
from src.train_utils import compute_class_weights, sanitize_task_name, \
    SingleLabelDataset, generate_predictions, text_to_ids, SimpleTextDataset

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords", quiet=True)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("training_debug.log"), logging.StreamHandler()],
)


def main(
    model_name="bert-base-uncased",
    subtask=1,
    epochs_num=20,
    batch_size=16,
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
        batch_size (int, optional): Batch size for training and evaluation. Defaults to 16.
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
            hazard_train_dataset, batch_size=batch_size, shuffle=True
        )
        hazard_val_loader = DataLoader(hazard_val_dataset, batch_size=batch_size)
        product_train_loader = DataLoader(
            product_train_dataset, batch_size=batch_size, shuffle=True
        )
        product_val_loader = DataLoader(product_val_dataset, batch_size=batch_size)

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
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

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
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_simple,
        )
        hazard_val_loader = DataLoader(
            hazard_val_ds, batch_size=batch_size, collate_fn=collate_fn_simple
        )
        product_train_loader = DataLoader(
            product_train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_simple,
        )
        product_val_loader = DataLoader(
            product_val_ds, batch_size=batch_size, collate_fn=collate_fn_simple
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
                for i in range(0, len(X_unlabeled_h), batch_size):
                    batch_ids = torch.tensor(
                        X_unlabeled_h[i : i + batch_size], dtype=torch.long
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
                for i in range(0, len(X_unlabeled_p), batch_size):
                    batch_ids = torch.tensor(
                        X_unlabeled_p[i : i + batch_size], dtype=torch.long
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
                batch_size=32,
                lr_val=2e-5,
                max_len=64,
                grad_acc_steps=1,
                use_stratify=True,
                syn_data_suffix="25",
                freeze_until_layers=0,
                is_multi_task=False,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train NLP model")

    parser.add_argument(
        "--model_name",
        type=str,
        default="tuned_bert",
        help="Name or path of the model to train (Possible choices: tuned_bert (default), facebook/bart-base, bert-base-uncased, roberta-base, microsoft/deberta-base, tfidf_logistic_regression, xgboost)."
    )
    parser.add_argument(
        "--subtask",
        type=int,
        default=1,
        help="Subtask identifier (1 or 2)."
    )
    parser.add_argument(
        "--epochs_num",
        type=int,
        default=20,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--lr_val",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=32,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps."
    )
    parser.add_argument(
        "--use_stratify",
        action="store_true",
        help="If set, use stratified splits during train/validation splitting."
    )
    parser.add_argument(
        "--syn_data_suffix",
        type=str,
        default="25",
        help="Suffix for synthetic data filenames."
    )
    parser.add_argument(
        "--freeze_until_layers",
        type=int,
        default=0,
        help="Number of layers to freeze in the model."
    )
    parser.add_argument(
        "--is_multi_task",
        action="store_true",
        help="If set, perform multi-task learning."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        subtask=args.subtask,
        epochs_num=args.epochs_num,
        batch_size=args.batch_size,
        lr_val=args.lr_val,
        max_len=args.max_len,
        grad_acc_steps=args.grad_acc_steps,
        use_stratify=args.use_stratify,
        syn_data_suffix=args.syn_data_suffix,
        freeze_until_layers=args.freeze_until_layers,
        is_multi_task=args.is_multi_task,
    )
