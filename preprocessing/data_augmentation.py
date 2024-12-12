import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    MarianMTModel,
    MarianTokenizer,
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import warnings
import logging
import multiprocessing
import matplotlib

# Set matplotlib to use a non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import wordnet, stopwords
import nltk
import random

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("augmentation_debug.log"), logging.StreamHandler()],
)

# NLTK downloads
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


############################################
# Domain-Specific Setup
############################################


def clean_title(title):
    # Step 1: Remove underscores
    title = re.sub(r"_", " ", title)

    # Step 2: Remove dates in various formats
    title = re.sub(
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "", title
    )  # Remove MM/DD/YYYY or similar
    title = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "", title)  # Remove YYYY-MM-DD
    title = re.sub(
        r"\b\d{1,2}\s+\w+\s+\d{4}\b", "", title, flags=re.IGNORECASE
    )  # Remove "12 July 2023"
    title = re.sub(r"\b\d{4}\b", "", title)  # Remove standalone years
    title = re.sub(r"^\d{4}\s*", "", title)  # Remove leading years

    # Step 3: Remove R trademark symbol and other symbols
    title = re.sub(r"\u00AE|®️", "", title)  # Remove R trademark symbol

    # Step 4: Remove HTML entities and other character encodings
    title = re.sub(r"&[a-zA-Z]+;", "", title)  # Remove entities like &rsquo; and &amp

    # Step 5: Remove unnecessary patterns
    title = re.sub(
        r"^Recall Notification:?\s*", "", title, flags=re.IGNORECASE
    )  # Remove specific prefixes
    title = re.sub(r"FSIS-\d+-\d+", "", title)  # Remove case numbers
    title = re.sub(
        r"\bReport\s*\d+-?\s*", "", title, flags=re.IGNORECASE
    )  # Remove "Report 064-2013"

    # Step 6: Remove parentheses
    title = re.sub(r"[()]", "", title)  # Remove parentheses

    # Step 7: Remove trailing hyphens
    title = re.sub(r"-\s*$", "", title)  # Remove strictly trailing hyphens

    # Step 8: Remove extra dashes and colons
    title = re.sub(r"\s+-\s+", " ", title)  # Remove dashes surrounded by spaces
    title = re.sub(r"^:\s*", "", title)  # Remove leading colons
    title = re.sub(r"\s*:\s*$", "", title)  # Remove trailing colons

    # Step 9: Normalize whitespace
    title = re.sub(r"\s+", " ", title).strip()

    # If cleaning results in an empty title, return an empty string
    return title


def filter_dataset(dataset):
    # Apply cleaning function in place
    dataset["title"] = dataset["title"].apply(clean_title)
    # Remove rows with empty cleaned titles
    dataset = dataset[dataset["title"] != ""]
    # Remove rows with fewer than 3 words in the cleaned title
    dataset = dataset[dataset["title"].str.split().str.len() >= 3].reset_index(
        drop=True
    )
    return dataset


############################################
# Paraphrasing and Back-Translation Models
############################################


def initialize_models(device="cpu"):
    """
    Initialize and return the augmentation models. This function is to be called once.
    """
    try:
        # Paraphrasing Model
        paraphrase_model_name = "ramsrigouthamg/t5_paraphraser"
        paraphrase_tokenizer = T5Tokenizer.from_pretrained(paraphrase_model_name)
        paraphrase_model = T5ForConditionalGeneration.from_pretrained(
            paraphrase_model_name
        )
        paraphrase_model = paraphrase_model.to(device)
        paraphrase_model.eval()

        # Back-Translation Models (English <-> German)
        en_de_model_name = "Helsinki-NLP/opus-mt-en-de"
        de_en_model_name = "Helsinki-NLP/opus-mt-de-en"

        en_de_tokenizer = MarianTokenizer.from_pretrained(en_de_model_name)
        en_de_model = MarianMTModel.from_pretrained(en_de_model_name).to(device).eval()

        de_en_tokenizer = MarianTokenizer.from_pretrained(de_en_model_name)
        de_en_model = MarianMTModel.from_pretrained(de_en_model_name).to(device).eval()

        return {
            "paraphrase_tokenizer": paraphrase_tokenizer,
            "paraphrase_model": paraphrase_model,
            "en_de_tokenizer": en_de_tokenizer,
            "en_de_model": en_de_model,
            "de_en_tokenizer": de_en_tokenizer,
            "de_en_model": de_en_model,
        }
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        return None


def paraphrase_text(text, models, device="cpu", num_return_sequences=1, num_beams=4):
    try:
        input_text = "paraphrase: " + text + " </s>"
        encoding = models["paraphrase_tokenizer"].encode_plus(
            input_text, return_tensors="pt", truncation=True
        )
        input_ids, attention_mask = encoding["input_ids"].to(device), encoding[
            "attention_mask"
        ].to(device)
        with torch.no_grad():
            outputs = models["paraphrase_model"].generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
            )
        paraphrased_sentences = [
            models["paraphrase_tokenizer"].decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in outputs
        ]
        return paraphrased_sentences[0].strip()
    except Exception as e:
        logging.error(f"Paraphrase error for text '{text}': {e}")
        return text + " notice"


def back_translate_text(text, models, device="cpu"):
    try:
        # English to German
        encoded = models["en_de_tokenizer"](
            [text], return_tensors="pt", truncation=True
        ).to(device)
        gen = models["en_de_model"].generate(**encoded, max_length=256, num_beams=4)
        german_text = models["en_de_tokenizer"].decode(gen[0], skip_special_tokens=True)

        # German back to English
        encoded_de = models["de_en_tokenizer"](
            [german_text], return_tensors="pt", truncation=True
        ).to(device)
        gen_en = models["de_en_model"].generate(
            **encoded_de, max_length=256, num_beams=4
        )
        back_translated = models["de_en_tokenizer"].decode(
            gen_en[0], skip_special_tokens=True
        )
        return back_translated.strip()
    except Exception as e:
        logging.error(f"Back translation error for text '{text}': {e}")
        return text + " notice"


############################################
# Synonym Augmentation
############################################


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").lower()
            synonym = "".join(
                [char for char in synonym if char.isalpha() or char == " "]
            )
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(text, n):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(
        set([word for word in words if word.lower() not in stop_words])
    )
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            if synonym != random_word and len(synonym) > 1:
                new_words = [
                    synonym if word == random_word else word for word in new_words
                ]
                num_replaced += 1
        if num_replaced >= n:
            break
    sentence = " ".join(new_words)
    return sentence if sentence.strip() != text.strip() else sentence + " notice"


############################################
# Augmentation Logic with CPU Parallelization and Visual Aids
############################################


def augment_text_line(text, models):
    method = random.choice(["synonyms", "paraphrase", "back_translation"])
    if method == "synonyms":
        n_sr = max(1, int(0.1 * len(text.split())))
        augmented = synonym_replacement(text, n_sr)
    elif method == "paraphrase":
        augmented = paraphrase_text(text, models, device="cpu")
    else:
        augmented = back_translate_text(text, models, device="cpu")

    if augmented.strip() == text.strip():
        # Try paraphrase as fallback
        augmented = paraphrase_text(text, models, device="cpu")
        if augmented.strip() == text.strip():
            augmented += " notice"
    return augmented


def plot_class_distribution(
    before_counts, after_counts, dataset_name, output_dir="plots"
):
    """
    Plots and saves the class distribution before and after augmentation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = sorted(before_counts.keys())
    before = [before_counts.get(label, 0) for label in classes]
    after = [after_counts.get(label, 0) for label in classes]

    x = np.arange(len(classes))  # label locations
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, before, width, label="Before Augmentation")
    rects2 = ax.bar(x + width / 2, after, width, label="After Augmentation")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Class Distribution Before and After Augmentation for {dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()

    # Attach a text label above each bar in rects1 and rects2, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_class_distribution.png"))
    plt.close()


def calculate_dynamic_target_count(class_counts, percentile_val=75):
    """
    Calculates the target count based on a specified percentile of the class distribution.
    This helps in preventing excessive augmentation for classes with very high counts.

    Parameters:
    - class_counts (Series): Series containing class counts.
    - percentile_val (int): The percentile to set as the target count.

    Returns:
    - target_count (int): The calculated target count.
    """
    # Debugging Statement: Verify the type of np.percentile
    values = list(class_counts.values)
    target_count = int(np.percentile(values, percentile_val))
    return target_count


def augment_dataset(
    df,
    label_col,
    text_col,
    models,
    percentile=75,
    max_aug_per_class=500,
    dataset_name="",
):
    """
    Augment the dataset by oversampling minority classes up to a dynamic target count,
    determined by a specified percentile of the class distribution. Limits the number
    of augmented samples per class to a maximum of 'max_aug_per_class' to prevent overfitting.
    Includes progress bars and logging for visual aid.
    Preserves all other features by duplicating them for augmented samples.

    Parameters:
    - df (DataFrame): The original dataset.
    - label_col (str): The name of the label column.
    - text_col (str): The name of the text column to augment.
    - models (dict): Dictionary containing initialized augmentation models.
    - percentile (int): The percentile of class counts to set as the target count.
    - max_aug_per_class (int): The maximum number of augmented samples per class.
    - dataset_name (str): The name of the dataset for logging and plotting purposes.

    Returns:
    - df (DataFrame): The augmented dataset.
    """
    # Get class counts before augmentation
    class_counts_before = df[label_col].value_counts().to_dict()

    # Get current class counts
    class_counts = df[label_col].value_counts()

    # Calculate dynamic target count based on percentile
    target_count = calculate_dynamic_target_count(
        class_counts, percentile_val=percentile
    )
    logging.info(
        f"Dynamic target count set to {target_count} based on the {percentile}th percentile."
    )

    augmented_texts = []
    augmented_labels = []
    augmented_other_features = []  # To store other features for augmented samples

    # Identify all other columns to preserve
    other_cols = [col for col in df.columns if col not in [text_col, label_col]]

    # Collect all original titles to ensure uniqueness
    original_titles = set(df[text_col].unique())

    # Progress bar for classes
    classes = class_counts.index.tolist()
    with tqdm(
        total=len(classes), desc=f"Augmenting classes in {dataset_name}", unit="class"
    ) as pbar_classes:
        for label in classes:
            current_count = class_counts[label]
            n_to_augment = min(target_count - current_count, max_aug_per_class)
            # Ensure n_to_augment is non-negative
            n_to_augment = max(n_to_augment, 0)
            if n_to_augment <= 0:
                pbar_classes.update(1)
                continue
            texts_to_augment = df[df[label_col] == label][text_col].tolist()
            if not texts_to_augment:
                pbar_classes.update(1)
                continue
            n_texts = len(texts_to_augment)
            if n_texts == 0:
                pbar_classes.update(1)
                continue
            n_aug_per_text = max(1, n_to_augment // n_texts)

            # Prepare tasks and keep track of original features
            tasks = []
            for text in texts_to_augment:
                for _ in range(n_aug_per_text):
                    tasks.append(text)
                    # Retrieve the original row to copy other features
                    original_row = df[df[text_col] == text].iloc[0]
                    other_features = original_row[other_cols].to_dict()
                    augmented_other_features.append(other_features)
                    if len(tasks) >= n_to_augment:
                        break
                if len(tasks) >= n_to_augment:
                    break

            # Use all available CPU cores
            num_processes = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Using starmap requires a list of tuples, hence (text, models) for each task
                # Wrap the pool.starmap with tqdm to visualize progress
                results = list(
                    tqdm(
                        pool.starmap(augment_text_line, [(t, models) for t in tasks]),
                        total=len(tasks),
                        desc=f'Augmenting label "{label}"',
                        leave=False,
                    )
                )

            augmented_texts.extend(results)
            augmented_labels.extend([label] * len(results))

            logging.info(f"Augmented class '{label}' with {len(results)} samples.")
            pbar_classes.update(1)

    # Ensure uniqueness of augmented titles
    filtered_augmented_texts = []
    filtered_augmented_labels = []
    filtered_augmented_other_features = []
    augmented_set = set()

    for text, label, features in zip(
        augmented_texts, augmented_labels, augmented_other_features
    ):
        if text not in original_titles and text not in augmented_set:
            filtered_augmented_texts.append(text)
            filtered_augmented_labels.append(label)
            filtered_augmented_other_features.append(features)
            augmented_set.add(text)

    # Log how many duplicates were removed
    duplicates_removed = len(augmented_texts) - len(filtered_augmented_texts)
    if duplicates_removed > 0:
        logging.info(f"Removed {duplicates_removed} duplicate augmented titles.")

    if filtered_augmented_texts:
        # Create a DataFrame for augmented samples
        augmented_df = pd.DataFrame(
            {text_col: filtered_augmented_texts, label_col: filtered_augmented_labels}
        )

        # Add other preserved features
        for col in other_cols:
            augmented_df[col] = [
                features[col] for features in filtered_augmented_other_features
            ]

        # Append augmented data to original DataFrame
        df = pd.concat([df, augmented_df], ignore_index=True)

    # Ensure no unwanted index columns exist
    unwanted_columns = ["Unnamed: 0"]
    for col in unwanted_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logging.info(f"Dropped unwanted column: {col}")

    # Reset the index to ensure it's clean
    df.reset_index(drop=True, inplace=True)

    # Get class counts after augmentation
    class_counts_after = df[label_col].value_counts().to_dict()

    # Plot class distribution
    plot_class_distribution(class_counts_before, class_counts_after, dataset_name)

    return df


def create_augmented_files():
    # Initialize augmentation models on CPU
    logging.info("Initializing augmentation models...")
    models = initialize_models(device="cpu")
    if models is None:
        logging.error(
            "Failed to initialize augmentation models. No augmentation performed."
        )
        return

    # Define dataset files and corresponding label columns
    datasets = [
        {
            "input_file": "hazard_category_data_25.csv",
            "output_file": "hazard_category_data_25_augmented.csv",
            "label_column": "hazard-category",
            "text_column": "title",
            "dataset_name": "Hazard Category Data",
        },
        {
            "input_file": "product_category_data_25.csv",
            "output_file": "product_category_data_25_augmented.csv",
            "label_column": "product-category",
            "text_column": "title",
            "dataset_name": "Product Category Data",
        },
        {
            "input_file": "hazard_data_25.csv",
            "output_file": "hazard_data_25_augmented.csv",
            "label_column": "hazard",
            "text_column": "title",
            "dataset_name": "Hazard Data",
        },
        {
            "input_file": "product_data_25.csv",
            "output_file": "product_data_25_augmented.csv",
            "label_column": "product",
            "text_column": "title",
            "dataset_name": "Product Data",
        },
    ]

    for dataset in datasets:
        input_path = dataset["input_file"]
        output_path = dataset["output_file"]
        label_col = dataset["label_column"]
        text_col = dataset["text_column"]
        dataset_name = dataset["dataset_name"]

        if os.path.exists(input_path):
            logging.info(f"Processing {input_path}...")

            # Check if the CSV has an index column by reading the first few lines
            sample_df = pd.read_csv(input_path, nrows=5)
            if "Unnamed: 0" in sample_df.columns:
                logging.info(
                    f"Detected 'Unnamed: 0' column in {input_path}. Setting it as index."
                )
                df = pd.read_csv(input_path, index_col=0)
                df.reset_index(drop=True, inplace=True)
            else:
                df = pd.read_csv(input_path)

            # Apply filter_dataset to clean and filter the data
            df = filter_dataset(df)
            logging.info(
                f"After filtering, {len(df)} samples remain in {dataset_name}."
            )

            # Augment dataset with visual aids
            df_augmented = augment_dataset(
                df=df,
                label_col=label_col,
                text_col=text_col,
                models=models,
                percentile=75,  # Set the target count based on the 75th percentile
                max_aug_per_class=500,  # Set maximum augmentation per class to 500
                dataset_name=dataset_name,
            )

            # Save augmented data without the index and without changing other features
            df_augmented.to_csv(output_path, index=False)
            logging.info(f"Saved {output_path}")
        else:
            logging.warning(f"File {input_path} does not exist. Skipping.")

    logging.info("Augmentation completed. The _augmented files are ready.")


# Run the augmentation process
if __name__ == "__main__":
    create_augmented_files()
