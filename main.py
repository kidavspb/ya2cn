# Importing the necessary libraries from scikit-learn
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data: list of reviews and their corresponding labels (Negative/Positive)
reviews = []
labels = []

with open('Review.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        reviews.append(row[1])
        labels.append(row[0])

# Splitting the dataset into training and testing sets
# 80% of data will be used for training, and 20% will be used for testing.
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2)

# Initializing the TF-IDF Vectorizer with English stop words
# Stop words are common words that do not contribute to the meaning of a sentence and are often removed to reduce noise.
vectorizer = TfidfVectorizer(stop_words='english')

# Transforming the training data: learning the vocabulary and converting text to vectors
X_train_vec = vectorizer.fit_transform(X_train)

# Transforming the testing data: using the vocabulary from training data and converting text to vectors
X_test_vec = vectorizer.transform(X_test)

# Initializing the Logistic Regression classifier
clf = LogisticRegression()

# Training the classifier using the vectorized training data
clf.fit(X_train_vec, y_train)

# Predicting the categories for the testing set
y_pred = clf.predict(X_test_vec)

# Printing a detailed classification report showing the performance metrics of the classifier
print(classification_report(y_test, y_pred))


import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

y_pred = clf.predict(X_test_vec)
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(set(labels)))
plt.xticks(tick_marks, set(labels), rotation=45)
plt.yticks(tick_marks, set(labels))

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




# For classifying a new review:
def analyze_comment(comment):
    new_review = vectorizer.transform([comment])
    prediction = clf.predict(new_review)
    return prediction[0]
