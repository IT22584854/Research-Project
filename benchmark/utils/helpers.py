import time
from datetime import datetime

def calculate_latency(start_ts, end_ts):

    try:
        if isinstance(start_ts, str):
            start = datetime.fromisoformat(start_ts)
            end = datetime.fromisoformat(end_ts)
            return (end - start).total_seconds()
        else:
            return float(end_ts) - float(start_ts)
    except Exception:
        return None