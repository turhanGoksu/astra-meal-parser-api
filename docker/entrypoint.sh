#!/bin/sh
# Start the API: fetch the model into the /models volume if it is missing
# (first start only), then hand PID 1 to uvicorn so it receives signals.
set -e
python -m scripts.download_model
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
