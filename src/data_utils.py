import re

import torch
from nltk.corpus import stopwords

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
