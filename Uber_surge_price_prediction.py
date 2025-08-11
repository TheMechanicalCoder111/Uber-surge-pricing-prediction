# Uber Surge Pricing Prediction

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("C:/Users/91960/OneDrive/Desktop/Uber_surge_prediction/uber_surge_data.csv")


# Separate features and target
X = data.iloc[:,:-1]
y = data['surge_multiplier_class']



# Encode categorical variables
le = LabelEncoder()

X['location_type'] = le.fit_transform(X['location_type'])
X['weather'] = le.fit_transform(X['weather'])
X['time_of_day'] = le.fit_transform(X['time_of_day'])
X['demand_level'] = le.fit_transform(X['demand_level'])


# Handle class imbalance with Random Over sampling
ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)



# Build Logistic Regressor for classification
lr = LogisticRegression()
lr.fit(X_train, y_train)


# Accuray Score(%)
lr.score(X_test, y_test)*100

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {acc: }')
print(f'Weighted F1 Score: {f1: }')
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot = True)
plt.show()