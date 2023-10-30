import csv
import random

reviews = []
sentiments = []

with open('Review.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        sentiments.append(1 if row[0] == 'Positive' else 0)
        reviews.append(row[1])

# # Display a few random reviews
# for i in range(5):
#     index = random.randint(0, len(reviews) - 1)
#     print(f"Review: {reviews[index]}")
#     print(f"Sentiment: {'Positive' if sentiments[index] == 1 else 'Negative'}")
#     print("-----")





from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torch

# Tokenization
tokenizer = get_tokenizer('basic_english')
tokenized_reviews = [tokenizer(review) for review in reviews]

# Build vocabulary
counter = Counter()
for review in tokenized_reviews:
    counter.update(review)
vocab = Vocab(counter)
print(vocab.stoi)

# Numericalize, pad, and split the data
def numericalize(tokenized_review, vocab):
    return [vocab[token] for token in tokenized_review]


numericalized_reviews = [numericalize(review, vocab) for review in tokenized_reviews]
padded_reviews = torch.nn.utils.rnn.pad_sequence([torch.tensor(review) for review in numericalized_reviews],
                                                 batch_first=True)

# Split the data
train_reviews, val_reviews, train_sentiments, val_sentiments = train_test_split(padded_reviews, sentiments,
                                                                                test_size=0.2, random_state=42)





import torch.nn as nn


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output





# Hyperparameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
LEARNING_RATE = 0.001

assert all([all([idx < VOCAB_SIZE for idx in review]) for review in numericalized_reviews]), "Found an out-of-vocab index."
# Initialize the model
model = SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)





# Training parameters
EPOCHS = 5
BATCH_SIZE = 64

# Convert data to tensor format
train_reviews_tensor = torch.stack(list(train_reviews))
train_sentiments_tensor = torch.tensor(train_sentiments, dtype=torch.float32).view(-1, 1)
val_reviews_tensor = torch.stack(list(val_reviews))
val_sentiments_tensor = torch.tensor(val_sentiments, dtype=torch.float32).view(-1, 1)

# Training loop
for epoch in range(EPOCHS):
    for i in range(0, len(train_reviews), BATCH_SIZE):
        batch_reviews = train_reviews_tensor[i:i + BATCH_SIZE]
        batch_sentiments = train_sentiments_tensor[i:i + BATCH_SIZE]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_reviews)
        loss = criterion(outputs, batch_sentiments)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print loss for every epoch
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")





with torch.no_grad():
    val_outputs = model(val_reviews_tensor)
    val_loss = criterion(val_outputs, val_sentiments_tensor)
    val_predictions = torch.round(torch.sigmoid(val_outputs))
    accuracy = (val_predictions == val_sentiments_tensor).sum().float() / len(val_sentiments)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
