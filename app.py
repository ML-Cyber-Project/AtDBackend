from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Report(BaseModel):
    flow_duration: int
    fwd_packet_length_std: float
    bwd_packet_length_mean: float
    flow_bytes_s: float
    flow_packets_s: float
    flow_iat_mean: float
    flow_iat_std: float
    bwd_iat_total: int
    bwd_iat_mean: float
    bwd_iat_std: float
    bwd_iat_max: int
    bwd_packets_s: float
    active_mean: float
    active_std: float
    active_max: int
    idle_std: float


@app.post('/reports/')
async def create_report(report: Report):
    return report
