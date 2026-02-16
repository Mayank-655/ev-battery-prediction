#!/bin/bash
# After install script for AWS CodeDeploy

cd /var/www/battery-prediction

echo "Installing Python dependencies..."
pip3 install --user -r api/requirements.txt

echo "Setting up environment..."
export PYTHONPATH=/var/www/battery-prediction:$PYTHONPATH

