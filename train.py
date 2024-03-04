import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json

seed_val = 1234
torch.manual_seed(seed_val)

train_data = pd.read_json('train.json')
train_texts = train_data['reviews'].tolist()
train_labels = train_data['sentiments'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_train_texts = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
train_labels = torch.tensor(train_labels)
train_dataset = TensorDataset(encoded_train_texts['input_ids'],
                              encoded_train_texts['attention_mask'],
                              train_labels)

num_epochs = 3
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    correct_predictions = 0
    total_samples = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%')

test_data = pd.read_json('test.json')
test_texts = test_data['reviews'].tolist()
encoded_test_texts = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')
test_dataset = TensorDataset(encoded_test_texts['input_ids'], encoded_test_texts['attention_mask'])
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
test_predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        test_predictions.extend(predicted_labels.cpu().numpy().tolist())

submission_df = pd.DataFrame({'predicted_sentiments': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print('predictions saved to submission.csv')

test_reviews = test_data['reviews'].tolist()
submission_data = []
for review, sentiment in zip(test_reviews, test_predictions):
    submission_data.append({'review': review, 'predicted_sentiment': sentiment})

with open('prediction.json', 'w') as json_file:
    json.dump(submission_data, json_file, indent=4)

print('predictions saved to prediction.json')
