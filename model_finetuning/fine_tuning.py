# Import necessary libraries
import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding


# Load data from CSV file
data_path = r"D:\AI & ML\Projects\TextOrigin AI vs Human Text Detection Examples\data\AI_Human.csv"
data = pd.read_csv(data_path) 
data0 = data[data['generated'] == 0][:500]
data1 = data[data['generated'] == 1][:500]
df = pd.concat([data0, data1], ignore_index=True)
print(df.columns)
df = df.rename(columns={'generated': 'labels'})
print(df.columns)
df['labels'] = df['labels'].astype(int)
print(df['labels'].unique()) 
print(df.head())
train_df, test_df = train_test_split(df, test_size=0.2)

from datasets import Dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Convert datasets to tokenized format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_data(examples):
    result = tokenizer(examples["text"], truncation=True)
    result["labels"] = examples["labels"]
    return result

tokenized_train = train_dataset.map(tokenize_data, batched=True)
tokenized_test = test_dataset.map(tokenize_data, batched=True)

# Load pre-trained DistilBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print("model loaded successfully.")

# Prepare data collator for padding sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch"
)
print("Training started.")
# Define Trainer object for training the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('model')
print("Model training complete and saved to 'model' directory.")