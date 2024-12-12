from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("https://mlflow.docsystem.xyz")

model_name = "attackdetection"

# Load the model
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

# Set labels
LABELS_NUM = ['BENIGN', 'DoS']

# Set column names mapping
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

app = FastAPI()


class Report(BaseModel):
    flow_duration: int
    fwd_packet_length_std: float
    bwd_packet_length_mean: float
    bwd_packet_length_std: float
    flow_bytes_s: float
    flow_packets_s: float
    flow_iat_mean: float
    flow_iat_std: float
    flow_iat_max: int
    fwd_iat_total: int
    fwd_iat_mean: float
    fwd_iat_std: float
    fwd_iat_max: int
    bwd_iat_total: int
    bwd_iat_mean: float
    bwd_iat_std: float
    bwd_iat_max: int
    fwd_packets_s: float
    bwd_packets_s: float
    packet_length_mean: float
    packet_length_std: float
    packet_length_variance: float
    average_packet_size: float
    avg_bwd_segment_size: float
    active_mean: float
    active_std: float
    active_max: int
    active_min: int
    idle_mean: float
    idle_std: float
    idle_max: int
    idle_min: int

    def to_df(self):
        df = pd.DataFrame([self.dict()])
        # Change column labels
        for column in df.columns:
            df = df.rename(columns={column: COLUMN_MAPPINGS[column]})
        return df


@app.post('/reports/')
async def create_report(report: Report):
    # Make predictions
    predictions = model.predict(report.to_df())
    predictions = [LABELS_NUM[prediction] for prediction in predictions.tolist()]
    return {"predictions": predictions}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
