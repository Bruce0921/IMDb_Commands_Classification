import main

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Splitting the labeled data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(main.labeled_train_vectors, main.labeled_train['sentiment'], test_size=0.2, random_state=42)

# Training the model
lr = LogisticRegression(max_iter=1000)  # Increasing max_iter for convergence
lr.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred = lr.predict(X_val)

# Evaluating the model
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))
