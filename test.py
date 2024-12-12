try:
    import mlflow
except ImportError:
    print("You need to install mlflow")
    exit()

try:
    import pandas as pd
except ImportError:
    print("You need to install pandas")
    exit()

try:
    from tqdm import tqdm
except ImportError:
    print("You need to install tqdm")
    exit()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
except ImportError:
    print("You need to install scikit-learn")
    exit()

try:
    import seaborn as sns
except ImportError:
    print("You need to install seaborn")
    exit()

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("You need to install matplotlib")
    exit()

mlflow.set_tracking_uri("https://mlflow.docsystem.xyz")
# load each models in the RandomSearch experiment
IS_RANDOM_TREE = True
model = mlflow.sklearn.load_model("runs:/c0147219bacd4e92bb09477869eb452d/model")

LABELS_NUM = ["BENIGN", "DoS"]

COLUMN_MAPPINGS = {
    "flow_duration": "Flow Duration",
    "fwd_packet_length_std": "Fwd Packet Length Std",
    "bwd_packet_length_mean": "Bwd Packet Length Mean",
    "bwd_packet_length_std": "Bwd Packet Length Std",
    "flow_bytes_s": "Flow Bytes/s",
    "flow_packets_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "fwd_iat_total": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "bwd_iat_total": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "fwd_packets_s": "Fwd Packets/s",
    "bwd_packets_s": "Bwd Packets/s",
    "packet_length_mean": "Packet Length Mean",
    "packet_length_std": "Packet Length Std",
    "packet_length_variance": "Packet Length Variance",
    "average_packet_size": "Average Packet Size",
    "avg_bwd_segment_size": "Avg Bwd Segment Size",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min"
}

# revert the column mappings
COLUMN_MAPPINGS = {v: k for k, v in COLUMN_MAPPINGS.items()}

# Load data/features_cleaned.csv and data/labels_cleaned.csv
features = pd.read_csv("data/features_cleaned.csv")
labels = pd.read_csv("data/labels_cleaned.csv")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=25)

def create_sample_json():
    # export 3 benign and 3 DoS samples in json format
    # say in the name of the file if it's a benign or a DoS sample (benign_sample_1.json, dos_sample_1.json, benign_sample_2.json, etc.)
    benign_samples = features[labels['Label'] == 0].sample(3)
    dos_samples = features[labels['Label'] == 1].sample(3)

    for i in range(3):
        # use column mappings to get the original column names
        benign_samples.iloc[i].rename(COLUMN_MAPPINGS).to_json(f"data/sample/benign_sample_{i + 1}.json")
        dos_samples.iloc[i].rename(COLUMN_MAPPINGS).to_json(f"data/sample/dos_sample_{i + 1}.json")

# create_sample_json()

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

def save_confusion_matrix(filename="confusion_matrix.png"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_NUM, yticklabels=LABELS_NUM)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

save_confusion_matrix()