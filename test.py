import mlflow
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("https://mlflow.docsystem.xyz")
model = mlflow.pyfunc.load_model(model_uri="models:/attackdetection/latest")

LABELS_NUM = ["BENIGN", "SUS"]


# Load data/features_cleaned.csv and data/labels_cleaned.csv
features = pd.read_csv("data/features_cleaned.csv")
features = features.astype(float)
labels = pd.read_csv("data/labels_cleaned.csv")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=25)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to DataFrame
X_train = pd.DataFrame(X_train, columns=features.columns)
X_test = pd.DataFrame(X_test, columns=features.columns)

good_predictions, bad_predictions = 0, 0

def predict_row(row: int):
    global good_predictions, bad_predictions
    # Get the row to predict
    df = X_train.iloc[[row]]

    prediction = [LABELS_NUM[i] for i in model.predict(df)][0]

    # Get its label
    # print(f"[{'OK' if LABELS_NUM[labels.iloc[row].values[0]] == prediction else '--'}] Expected label: {LABELS_NUM[labels.iloc[row].values[0]]}, got label: {prediction}")

    return LABELS_NUM[y_train.iloc[row].values[0]] == prediction

while True:
    for i in tqdm(range(10000)):
        j = random.randint(0, len(X_train) - 1)
        is_good_prediction = predict_row(j)
        if is_good_prediction:
            good_predictions += 1
        else:
            bad_predictions += 1

    print(f"Good predictions: {good_predictions}, Bad predictions: {bad_predictions}")
    print(f"Total predictions: {good_predictions + bad_predictions}")
    print(f"Accuracy: {good_predictions / (good_predictions + bad_predictions) * 100:.2f}%")

"""
response = requests.post('http://localhost:8000/reports/', json=data)

print(response.json())
"""
