import mlflow
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

mlflow.set_tracking_uri("https://mlflow.docsystem.xyz")
# load each models in the RandomSearch experiment
IS_RANDOM_TREE = True
model = mlflow.sklearn.load_model("runs:/c0147219bacd4e92bb09477869eb452d/model")

LABELS_NUM = ["BENIGN", "SUS"]


# Load data/features_cleaned.csv and data/labels_cleaned.csv
features = pd.read_csv("data/features_cleaned.csv")
labels = pd.read_csv("data/labels_cleaned.csv")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=25)

# Scale data
if not IS_RANDOM_TREE:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# convert to DataFrame
X_train = pd.DataFrame(X_train, columns=features.columns)
X_test = pd.DataFrame(X_test, columns=features.columns)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# evalData = X_test.copy()
# evalData['label'] = y_test

# calculate the model's metrics
def calculate_metrics():
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f1_score(y_test, y_pred))

calculate_metrics()

good_predictions, bad_predictions = 0, 0

def predict_row(row: int):
    global good_predictions, bad_predictions
    # Get the row to predict
    df = X_test.iloc[[row]]

    prediction = [LABELS_NUM[i] for i in model.predict(df)][0]

    # Get its label
    # print(f"[{'OK' if LABELS_NUM[labels.iloc[row].values[0]] == prediction else '--'}] Expected label: {LABELS_NUM[labels.iloc[row].values[0]]}, got label: {prediction}")

    return LABELS_NUM[y_test.iloc[row].values[0]] == prediction

while False:
    for i in tqdm(range(10000)):
        j = random.randint(0, len(X_test) - 1)
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
