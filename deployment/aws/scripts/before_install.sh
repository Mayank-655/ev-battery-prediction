#!/bin/bash
# Before install script for AWS CodeDeploy

echo "Installing system dependencies..."
sudo yum update -y
sudo yum install -y python3 python3-pip gcc

echo "Creating application directory..."
sudo mkdir -p /var/www/battery-prediction
sudo chown ec2-user:ec2-user /var/www/battery-prediction

