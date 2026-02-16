#!/bin/bash
# EC2 User Data Script for Auto-Setup

# Update system
yum update -y

# Install Docker
yum install -y docker
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Python and dependencies
yum install -y python3 python3-pip gcc

# Clone repository (replace with your repo URL)
# git clone https://github.com/Mayank-655/ev-battery-prediction.git /opt/battery-prediction

# Set up application
# cd /opt/battery-prediction
# pip3 install -r api/requirements.txt

# Start application with Docker Compose
# cd /opt/battery-prediction/deployment
# docker-compose up -d

