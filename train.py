import main
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Define IMDbDataset class
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Load a Pre-trained BERT model and Tokenizer:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Splitting the data
train_data, valid_data = train_test_split(main.labeled_train, test_size=0.1)  # 10% for validation

# Tokenize datasets:
train_encodings = tokenizer(list(train_data['review'].apply(' '.join)), truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(list(valid_data['review'].apply(' '.join)), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(list(main.test['review'].apply(' '.join)), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Add labels
train_encodings["labels"] = torch.tensor(train_data['sentiment'].tolist())
val_encodings["labels"] = torch.tensor(valid_data['sentiment'].tolist())

# Create IMDbDataset for train, validation, and test
train_dataset = IMDbDataset(train_encodings)
val_dataset = IMDbDataset(val_encodings)
test_dataset = IMDbDataset(test_encodings)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=500,
    do_train=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print(f"Evaluation Loss: {eval_results['eval_loss']}")

# Predict on the validation set for additional metrics
val_predictions = trainer.predict(val_dataset)
val_predicted_labels = np.argmax(val_predictions.predictions, axis=1)
true_labels = valid_data['sentiment'].tolist()

# Compute additional metrics
accuracy = accuracy_score(true_labels, val_predicted_labels)
precision = precision_score(true_labels, val_predicted_labels)
recall = recall_score(true_labels, val_predicted_labels)
f1 = f1_score(true_labels, val_predicted_labels)
probs = np.exp(val_predictions.predictions) / np.sum(np.exp(val_predictions.predictions), axis=1, keepdims=True)
roc_auc = roc_auc_score(true_labels, probs[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Predict on the test set
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Ensure the length of predicted labels matches the test set
assert len(predicted_labels) == len(main.test), "Mismatch between number of predictions and test samples."

# Ensure the test dataset has 25,000 rows
assert len(main.test) == 25000, "Test dataset does not have 25,000 rows."

# Create the submission DataFrame using the correct IDs from the test set
submission_df = pd.DataFrame({
    'id': main.test['id'].values,  # Use .values to ensure correct order
    'sentiment': predicted_labels
})

# Save the DataFrame to a CSV file with the correct format
submission_df.to_csv('submission.csv', index=False, sep=',')

