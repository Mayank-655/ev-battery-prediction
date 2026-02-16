# 🚀 Deployment Guide

This guide covers deploying the EV Battery Degradation Prediction system in three ways:

1. **Flask REST API** - Local or server deployment
2. **Streamlit Dashboard** - Interactive web interface
3. **AWS Deployment** - Cloud deployment with Docker

---

## 1. Flask REST API

### Local Deployment

```bash
# Install dependencies
cd api
pip install -r requirements.txt

# Make sure you have a trained model
cd ..
python main.py train

# Start the API
cd api
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /model_info` - Model information
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [3.7, -2.0, 25.0, 1, 2.0],
      [3.6, -2.0, 25.5, 2, 1.99],
      ...
    ]
  }'
```

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
```

---

## 2. Streamlit Dashboard

### Local Deployment

```bash
# Install dependencies
cd dashboard
pip install -r requirements.txt

# Make sure you have a trained model
cd ..
python main.py train

# Start the dashboard
cd dashboard
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Features

- **Data Overview**: Explore battery datasets
- **Predictions**: Interactive prediction interface
- **Model Performance**: View evaluation metrics
- **Settings**: Model configuration

### Production Deployment

Use Streamlit Cloud or deploy with:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 3. AWS Deployment

### Option A: Docker Deployment

#### Build and Run Locally

```bash
# Build Docker image
docker build -f deployment/Dockerfile -t battery-prediction-api .

# Run container
docker run -p 5000:5000 battery-prediction-api
```

#### Deploy to AWS ECS/Fargate

1. Push image to ECR:
```bash
aws ecr create-repository --repository-name battery-prediction
docker tag battery-prediction-api:latest <account>.dkr.ecr.<region>.amazonaws.com/battery-prediction:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/battery-prediction:latest
```

2. Create ECS task definition and service
3. Configure load balancer and auto-scaling

### Option B: EC2 Deployment

1. **Launch EC2 Instance**
   - AMI: Amazon Linux 2
   - Instance type: t3.medium or larger
   - Security group: Allow port 5000

2. **SSH into instance and run:**
```bash
# Clone repository
git clone https://github.com/Mayank-655/ev-battery-prediction.git
cd ev-battery-prediction

# Install dependencies
pip3 install -r api/requirements.txt

# Train model (or upload pre-trained model)
python3 main.py train

# Start API
cd api
python3 app.py
```

3. **Use systemd for service management:**
```bash
sudo nano /etc/systemd/system/battery-api.service
```

```ini
[Unit]
Description=Battery Prediction API
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/ev-battery-prediction/api
ExecStart=/usr/bin/python3 app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable battery-api
sudo systemctl start battery-api
```

### Option C: AWS Lambda (Serverless)

For serverless deployment, use AWS Lambda with API Gateway:

1. Package model and dependencies
2. Create Lambda function
3. Configure API Gateway
4. Set up environment variables

### Option D: AWS Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB:
```bash
eb init -p python-3.11 battery-prediction
eb create battery-prediction-env
```

3. Deploy:
```bash
eb deploy
```

---

## Environment Variables

Set these for production:

```bash
export FLASK_ENV=production
export MODEL_PATH=/path/to/model
export PORT=5000
```

---

## Monitoring & Logging

### Health Checks

The API includes a health endpoint:
```bash
curl http://localhost:5000/health
```

### Logging

Configure logging in `api/app.py`:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Security Considerations

1. **API Keys**: Add authentication for production
2. **HTTPS**: Use SSL/TLS certificates
3. **Rate Limiting**: Implement rate limiting
4. **Input Validation**: Validate all inputs
5. **CORS**: Configure CORS properly

---

## Scaling

- **Horizontal Scaling**: Use load balancer with multiple instances
- **Vertical Scaling**: Increase instance size
- **Caching**: Cache model predictions
- **CDN**: Use CloudFront for static assets

---

## Cost Optimization

- Use spot instances for training
- Auto-scale based on demand
- Use S3 for model storage
- Implement request caching

---

## Troubleshooting

### Model Not Found
- Ensure model is trained: `python main.py train`
- Check model path in config

### Port Already in Use
- Change port: `app.run(port=5001)`
- Kill existing process: `lsof -ti:5000 | xargs kill`

### Memory Issues
- Reduce batch size
- Use smaller model
- Enable model quantization

---

For more details, see the main README.md

