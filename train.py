import main
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load a Pre-trained BERT model and Tokenizer:
# For this task, we can use the basic BERT model, but there are many variants available.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize your dataset:
# Tokenize your reviews to convert them into a format suitable for BERT.
train_encodings = tokenizer(list(main.labeled_train['review'].apply(' '.join)), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(main.test['review'].apply(' '.join)), truncation=True, padding=True, max_length=512)

# Create a Dataset:
# Transformers library uses a custom dataset format.

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = IMDbDataset(train_encodings, main.labeled_train['sentiment'].tolist())
test_dataset = IMDbDataset(test_encodings)

# Train the Model:
# Use the Trainer class from the Transformers library to train the model.
training_args = TrainingArguments(
    output_dir='./results',  # specify a directory for outputs
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Evaluate and Predict:
# After training, we can use the trained model to predict sentiments of the test dataset.

predictions = trainer.predict(test_dataset)
predicted_labels = main.np.argmax(predictions.predictions, axis=1)

# Submission

# Create a DataFrame
submission_df = pd.DataFrame({
    'id': main.test['id'],  # Assuming the 'id' column is in your test DataFrame
    'sentiment': predicted_labels
})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
