# Food Hazard Detection

This repository contains the final project for CS 6120: Natural Language Processing

Developed by Pushkar Kurhekar, Kerem Sahin, and Siddhartha Roy.

The project focuses on detecting food-related hazards by analyzing textual data using various machine learning 
and natural language processing models and techniques.

## Challenge Details

This project is an attempt to tackle the **Food Hazard Detection Challenge** as part of **SemEval 2025**.  
More details about the challenge can be found on the [official challenge website](https://food-hazard-detection-semeval-2025.github.io/).  
The challenge focuses on identifying potential hazards in food-related textual data, with a goal of improving food safety and awareness.

## Dataset

The dataset used for this challenge is provided by the organizers of the SemEval 2025 Food Hazard Detection Challenge.  
You can access the dataset and its details on the [official challenge website](https://food-hazard-detection-semeval-2025.github.io/).  
Ensure that the dataset is placed in the `data/` directory before running the training script.

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone https://github.com/pshkrh/food-hazard-detection.git
   cd food-hazard-detection
   ```

2. **Create a Virtual Environment**:
   ```
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

    If you do not have a GPU:
    ```
    pip install -r requirements.txt
    ```
   
    If you have a CUDA-enabled GPU:
    ```
    pip install -r requirements_gpu.txt
    ```
   
    You might also need to install nltk data (though this has been handled in the train code at the beginning):
    ```python
    import nltk
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("stopwords")
    ```

## How to Run the Synthetic Data Generation Pipeline

1. The notebook containing the synthetic data generation code is located in the `notebooks/` directory with the file name `data_synthesis_llama.ipynb`. This is best run on Colab with a GPU, or another similar GPU-based platform.

2. The `load_in_8bit` parameter should be set to `False` if running for 25 or lower minimum samples per class. For anything greater, we recommend enabling this since we ran out of memory for the LLama 3.1 8B model on our NVIDIA L4 GPU with 24GB of VRAM running on GCP.

3. The cells should be run sequentially. The data path by default is set to use a Google Cloud Storage bucket, but can be changed to a local file.

4. Running the code in the notebook will generate four CSV files: Hazard, Product each for both subtasks. This can then be used as input data for training the model. 

## How to Run Training

1. **Navigate to the `src/` Directory**:
   Change into the `src/` directory where the `train.py` script is located.
   ```bash
   cd src/
   ```

2. **Run the Training Script**:
   To run the training process, execute the `train.py` script. Running it without any arguments will use the best model we reported on, which is the tuned BERT model:
   ```bash
   python train.py
   ```

3. **Data Requirements**:
   Ensure that the required data files are stored in the `data/` folder with the following specific filenames:
   - For Subtask 1 (ST1):
     - `hazard_category_25.csv`
     - `product_category_25.csv`
   - For Subtask 2 (ST2):
     - `hazard_data_25.csv`
     - `product_data_25.csv`

   Here, the `25` suffix corresponds to the synthetic data generation suffix. Update the filenames if a different suffix is used.
   
   To generate the submission, you will also need the unlabeled validation set file, named `incidents_unlabeled_val.csv` in the `data` folder.


4. **Outputs**:
   The script will generate the following outputs in the `outputs/` directory:
   - Training results, including performance metrics.
   - Loss and F1 score plots to visualize the model’s training and evaluation performance.
   - A competition-ready submission file (if the unlabeled validation set file is present in the `data/` directory as mentioned above).

### Example Usage

To run the training script with the default configuration:
```bash
python train.py
```

### Command-Line Arguments for `train.py`

The `train.py` script includes several command-line arguments that allow you to configure the training process. Below is a brief explanation of each argument:

- `--model_name`: Name or path of the model to train.  
  **Default**: `tuned_bert`  
  **Possible Choices**: 
  - `tuned_bert` (default)
  - `facebook/bart-base`
  - `bert-base-uncased`
  - `roberta-base`
  - `microsoft/deberta-base`
  - `tfidf_logistic_regression`
  - `xgboost`  
  **Example**: `--model_name roberta-base`

- `--subtask`: Identifier for the subtask to solve (e.g., 1 or 2).  
  **Default**: `1`  
  **Example**: `--subtask 2`

- `--epochs_num`: Number of epochs for training.  
  **Default**: `20`  
  **Example**: `--epochs_num 30`

- `--batch_size`: Batch size for training and evaluation.  
  **Default**: `16`  
  **Example**: `--batch_size 32`

- `--lr_val`: Learning rate for the optimizer.  
  **Default**: `2e-5`  
  **Example**: `--lr_val 3e-5`

- `--max_len`: Maximum sequence length for tokenization. This controls the input length fed into the model.  
  **Default**: `100`  
  **Example**: `--max_len 128`

- `--grad_acc_steps`: Number of gradient accumulation steps. This helps simulate larger batch sizes on memory-constrained hardware.  
  **Default**: `1`  
  **Example**: `--grad_acc_steps 4`

- `--use_stratify`: Flag to enable stratified splits during train/validation splitting.  
  **Default**: Not set  
  **Example**: Add `--use_stratify` to enable.

- `--syn_data_suffix`: Suffix for synthetic data filenames. This is used to identify the relevant synthetic datasets.  
  **Default**: `25`  
  **Example**: `--syn_data_suffix 50`

### Example Usage

To train a `roberta-base` model on subtask 2 for 30 epochs with a batch size of 32 and a maximum token length of 128:
```bash
python train.py --model_name roberta-base --subtask 2 --epochs_num 30 --batch_size 32 --max_len 128
```

To enable stratified splits during train/validation splitting:
```bash
python train.py --use_stratify
```

## How to Run Training (Notebook Version)

If you prefer using Jupyter Notebooks instead of the `train.py` script, we provide a notebook in the repository for running the **tuned BERT model**.

In the `notebooks/` directory, the `bert_best_implementation.ipynb` notebook contains the code for training the tuned BERT model that we reported on.

Once the data is uploaded to the same folder as the notebook's location, running the cells sequentially will:
- Load the input data
- Preprocess it (title cleaning)
- Run the model training
- Save relevant metrics and plots
- Generate a submission csv and zip to upload to the competition leaderboard

## Repository Structure

- `data/`: Contains datasets used for training and evaluation.
- `notebooks/`: Jupyter notebooks for data exploration, model training, and analysis.
- `preprocessing/`: Scripts for data cleaning and preprocessing.
- `src/`: Source code for model definitions, training, and evaluation.
- `requirements.txt`: List of required Python packages.
