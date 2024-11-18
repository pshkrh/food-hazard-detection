import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import os
from shutil import make_archive


SAVE_MODELS = False


# load training data:
data = pd.read_csv('../data/incidents_train.csv', index_col=0)
train_df, dev_df = train_test_split(data, test_size=0.2, random_state=2024)

train_df.sample()
data.title.str.split().apply(len).describe()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(examples):
    return tokenizer(examples['title'], padding=True, truncation=True)


def prepare_data(label):
    # encode labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(data[label])

    train_df['label'] = label_encoder.transform(train_df[label])
    dev_df['label'] = label_encoder.transform(dev_df[label])

    # Convert DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    # Apply the tokenizer to the dataset
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)

    # Create DataCollator to handle padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=16)

    # Convert dataset to PyTorch format
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create DataLoader objects
    return (
        DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator),
        DataLoader(dev_dataset, batch_size=8, collate_fn=data_collator),
        label_encoder
    )


def compute_score(hazards_true, products_true, hazards_pred, products_pred):
    # compute f1 for hazards:
    f1_hazards = f1_score(
        hazards_true,
        hazards_pred,
        average='macro'
    )

    # compute f1 for products:
    f1_products = f1_score(
        products_true[hazards_pred == hazards_true],
        products_pred[hazards_pred == hazards_true],
        average='macro'
    )

    return (f1_hazards + f1_products) / 2.

label = 'hazard-category'

# Create DataLoader objects
train_dataloader, dev_dataloader, le_hazard_category = prepare_data(label)

model_hazard_category = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data[label].unique()))
model_hazard_category.to('cuda')  # Move model to GPU if available

optimizer = AdamW(model_hazard_category.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model_hazard_category.train()

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_hazard_category(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)



model_hazard_category.eval()
total_predictions = []
with torch.no_grad():
    for batch in dev_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_hazard_category(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_predictions.extend([p.item() for p in predictions])

predicted_labels = le_hazard_category.inverse_transform(total_predictions)
gold_labels = le_hazard_category.inverse_transform(dev_df.label.values)
print(classification_report(gold_labels, predicted_labels, zero_division=0))

dev_df['predictions-hazard-category'] = predicted_labels

label = 'product-category'

# Create DataLoader objects
train_dataloader, dev_dataloader, le_product_category = prepare_data(label)

model_product_category = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data[label].unique()))
model_product_category.to('cuda')  # Move model to GPU if available

optimizer = AdamW(model_product_category.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model_product_category.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_product_category(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model_product_category.eval()
total_predictions = []
with torch.no_grad():
    for batch in dev_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_product_category(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_predictions.extend([p.item() for p in predictions])

predicted_labels = le_product_category.inverse_transform(total_predictions)
gold_labels = le_product_category.inverse_transform(dev_df.label.values)
print(classification_report(gold_labels, predicted_labels, zero_division=0))

dev_df['predictions-product-category'] = predicted_labels

score = compute_score(
    dev_df['hazard-category'], dev_df['product-category'],
    dev_df['predictions-hazard-category'], dev_df['predictions-product-category']
)
print(f"Score Sub-Task 1: {score:.3f}")



label = 'hazard'

# Create DataLoader objects
train_dataloader, dev_dataloader, le_hazard = prepare_data(label)



model_hazard = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data[label].unique()))
model_hazard.to('cuda')  # Move model to GPU if available

optimizer = AdamW(model_hazard.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model_hazard.train()

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_hazard(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model_hazard.eval()
total_predictions = []
with torch.no_grad():
    for batch in dev_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_hazard(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_predictions.extend([p.item() for p in predictions])

predicted_labels = le_hazard.inverse_transform(total_predictions)
gold_labels = le_hazard.inverse_transform(dev_df.label.values)
print(classification_report(gold_labels, predicted_labels, zero_division=0))

dev_df['predictions-hazard'] = predicted_labels

label = 'product'

# Create DataLoader objects
train_dataloader, dev_dataloader, le_product = prepare_data(label)

model_product = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data[label].unique()))
model_product.to('cuda')  # Move model to GPU if available

optimizer = AdamW(model_product.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
model_product.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_product(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model_product.eval()
total_predictions = []
with torch.no_grad():
    for batch in dev_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model_product(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_predictions.extend([p.item() for p in predictions])

predicted_labels = le_product.inverse_transform(total_predictions)
gold_labels = le_product.inverse_transform(dev_df.label.values)
print(classification_report(gold_labels, predicted_labels, zero_division=0))

dev_df['predictions-product'] = predicted_labels

score = compute_score(
    dev_df['hazard'], dev_df['product'],
    dev_df['predictions-hazard'], dev_df['predictions-product']
)
print(f"Score Sub-Task 2: {score:.3f}")


if SAVE_MODELS:
    model_product_category.save_pretrained("bert_product_category")
    np.save("bert_product_category/label_encoder.npy", le_product_category.classes_)

    model_product.save_pretrained("bert_product")
    np.save("bert_product/label_encoder.npy", le_product.classes_)
    tokenizer.save_pretrained("bert_tokenizer")


#### PREDICT ON TEST SET

# load test data:
test_df = pd.read_csv('../data/incidents_unlabeled_test.csv', index_col=0)

test_df.sample()


def predict(texts, model_path, tokenizer_path="bert_tokenizer"):
    # Load the saved tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Load the saved label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(model_path + '/label_encoder.npy', allow_pickle=True)

    # Load the saved model
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Decode predictions back to string labels
    return label_encoder.inverse_transform(predictions.cpu().numpy().tolist())

predictions = pd.DataFrame()

for column in ['hazard-category', 'product-category', 'hazard', 'product']:
  predictions[column] = predict(test_df.title.to_list(), f"bert_{column.replace('-', '_')}")

predictions.sample()


# save predictions to a new folder:
os.makedirs('./submission/', exist_ok=True)
predictions.to_csv('./submission/submission.csv')

# zip the folder (zipfile can be directly uploaded to codalab):
make_archive('./submission', 'zip', './submission')