#!/bin/bash
# Application start script for AWS CodeDeploy

cd /var/www/battery-prediction

echo "Starting Flask API..."
export PYTHONPATH=/var/www/battery-prediction:$PYTHONPATH

# Use systemd or supervisor for production
# For now, run in background
nohup python3 api/app.py > /var/log/battery-api.log 2>&1 &

echo "API started. PID: $!"

