from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("https://mlflow.docsystem.xyz")

model_name = "attackdetection"

# Load the model
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

# Set labels
LABELS_NUM = ['BENIGN', 'DDoS']

# Set column names mapping
COLUMN_MAPPINGS = {
    "flow_duration": "Flow Duration",
    "fwd_packet_length_std": "Fwd Packet Length Std",
    "bwd_packet_length_mean": "Bwd Packet Length Mean",
    "flow_bytes_s": "Flow Bytes/s",
    "flow_packets_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "bwd_iat_total": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_packets_s": "Bwd Packets/s",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "idle_std": "Idle Std"
}

app = FastAPI()


class Report(BaseModel):
    flow_duration: float
    fwd_packet_length_std: float
    bwd_packet_length_mean: float
    flow_bytes_s: float
    flow_packets_s: float
    flow_iat_mean: float
    flow_iat_std: float
    bwd_iat_total: float
    bwd_iat_mean: float
    bwd_iat_std: float
    bwd_iat_max: float
    bwd_packets_s: float
    active_mean: float
    active_std: float
    active_max: float
    idle_std: float

    def to_df(self):
        df = pd.DataFrame([self.dict()])
        # Change column labels
        for column in df.columns:
            df = df.rename(columns={column: COLUMN_MAPPINGS[column]})
        return df

    def scale_df(self):
        scaler = StandardScaler()
        return scaler.transform(self.to_df())


@app.post('/reports/')
async def create_report(report: Report):
    # Make predictions
    predictions = model.predict(report.scale_df())
    print(predictions)
    predictions = [LABELS_NUM[prediction] for prediction in predictions.tolist()]
    return {"predictions": predictions}
