# Import Necessary Libraries
import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification, AdamW
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import pytesseract
from tqdm import tqdm
from PIL import Image
import difflib
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

# Hyperparameters
epochs = 10 # Number of training epochs
lr = 5e-5 # Learning Rate

# Set the device to GPU if available, else use CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define image directory
image_dir = r"/home/bhanu/ji/merged_images"

# Load the processed dataset JSON file
data_file = r"/home/bhanu/ji/combined_final_data.json"
with open(data_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data) # Convert the data into a pandas DataFrame

# Update image paths to include full directory
df["image_path"] = df["image_path"].apply(lambda x: os.path.join(image_dir, x))

# Verify image paths
print("Sample image paths:", df["image_path"].head())

# Define the target labels for token classification
tags = [
    "0", "Judge", "Judgment Date", "Case Number",
    "Petitioner/Accused", "Petitioner Advocate", "Respondent/Complainant", "Respondent Advocate"
]

# Print the target labels
print("Target labels:", tags, "\n")

# Create mappings for label-to-id and id-to-label
tags_to_ids = {v: k for k, v in enumerate(tags)}
ids_to_tags = {k: v for k, v in enumerate(tags)}

# Function to find the closest key value (to handle annotation errors)
def find_closest_key(input_key: str, actual_keys: List[str] = tags_to_ids.keys()) -> str:
    closest_match = difflib.get_close_matches(input_key, actual_keys, n=1, cutoff=0.8)
    return closest_match[0] if closest_match else print("--no-match-found--")

# Convert the labels to integer IDs using the tags_to_ids mapping
for k, row in enumerate(df.labels):
    df.labels[k] = [tags_to_ids[find_closest_key(x)] if not str(x).isdigit() else int(x) for x in row]

# Initialize feature extractor and tokenizer for LayoutLMv3
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

# Create a Hugging Face dataset from the pandas DataFrame
dataset = Dataset.from_pandas(df)

# Define the features of the dataset
features = Features({
    "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
    "input_ids": Sequence(Value(dtype="int64")),
    "attention_mask": Sequence(Value(dtype="int64")),
    "bbox": Array2D(dtype="int64", shape=(512, 4)),
    "labels": Sequence(ClassLabel(num_classes=len(tags), names=tags)),
})

# Function to process the dataset and convert images, tokens, and labels into the required format
def process_data(examples: Dict[str, any]) -> Dict[str, any]:
    """
    Process the dataset: extract images, tokenize text, and structure inputs for the model.
    """
    images = []
    for path in examples["image_path"]:
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            print(f"Warning: Missing image file {path}")
            images.append(Image.new("RGB", (224, 224))) # Use a blank image as a placeholder

    features = feature_extractor(images)
    labels = [label for label in examples["labels"]]
    words = [word[0] for word in examples["words"]]
    boxes = [box[0] for box in examples["boxes"]]
    
    encoded_inputs = tokenizer(words, boxes=boxes, word_labels=labels, max_length=512, padding="max_length", truncation=True)
    encoded_inputs["pixel_values"] = features["pixel_values"]
    
    return encoded_inputs

# Apply the processing function to the dataset
encoded_dataset = dataset.map(process_data, remove_columns=dataset.column_names, features=features, batched=True, batch_size=128, desc="Processing data...")
encoded_dataset.set_format(type="torch", device="cuda")

# Split the dataset into training and test sets (80% train, 20% test)
train_test_data = encoded_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_data["train"].shuffle()
test_dataset = train_test_data["test"]
print(f"\nData size: {len(encoded_dataset)}")
print(f"Train data: {len(train_dataset)}\nTest data: {len(test_dataset)}\n")

# Create DataLoader objects for batching and shuffling the training and test datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

# Load the pre-trained LayoutLMv3 model for token classification
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(tags),
    id2label=ids_to_tags,
    label2id=tags_to_ids
).to(device)

#----------------
# Model Training 
#----------------
print("\nModel training starts.\n")
optimizer = AdamW(model.parameters(), lr=lr)

# Define batch sizes for training and testing
train_batch_per_epoch = len(train_dataloader)
test_batch_per_epoch = len(test_dataloader)

# Training loop
for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train() # Set the model to training mode
    running_loss = 0.0
    running_correct = 0
    loop = tqdm(train_dataloader, leave=False, desc="Training")

    for batch in loop:
        batch = {key: value.to(device) for key, value in batch.items()} # Move batch to device
        output = model(**batch)
        
        loss = output.loss
        running_loss += loss.item()
        
        predictions = output.logits.argmax(-1)
        target_label_pos = np.where(batch["labels"].cpu() != -100)
        correct = (predictions[target_label_pos] == batch["labels"][target_label_pos]).float().sum() / len(target_label_pos[0])
        running_correct += correct

        loop.set_postfix(loss=loss.item(), accuracy=correct.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate average training accuracy and loss
    train_accuracy = round((100 * running_correct / train_batch_per_epoch).item(), 2)
    train_loss = running_loss / train_batch_per_epoch

    # Evaluation loop
    model.eval()
    running_loss = 0.0
    running_correct = 0
    
    loop = tqdm(test_dataloader, leave=False, desc="Evaluating")
    for batch in loop:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)
            loss = output.loss
            running_loss += loss.item()
            
            predictions = output.logits.argmax(-1)
            target_label_pos = np.where(batch["labels"].cpu() != -100)
            correct = (predictions[target_label_pos] == batch["labels"][target_label_pos]).float().sum() / len(target_label_pos[0])
            running_correct += correct

            loop.set_postfix(loss=loss.item(), accuracy=correct.item())

    val_accuracy = round((100 * running_correct / test_batch_per_epoch).item(), 2)
    val_loss = running_loss / test_batch_per_epoch

    print(f"\nEpoch {epoch+1}: Train Loss={train_loss}, Train Acc={train_accuracy}%, Test Loss={val_loss}, Test Acc={val_accuracy}%")

# Save the trained model
model.save_pretrained("model")
print("Model saved successfully!")
