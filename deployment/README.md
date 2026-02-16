# 🚀 Quick Deployment Guide

## Flask REST API

```bash
# 1. Install dependencies
cd api
pip install -r requirements.txt

# 2. Train model (if not already done)
cd ..
python main.py train

# 3. Start API
cd api
python app.py
```

**Test the API:**
```bash
# In another terminal
python api/test_api.py
```

## Streamlit Dashboard

```bash
# 1. Install dependencies
cd dashboard
pip install -r requirements.txt

# 2. Start dashboard
streamlit run app.py
```

Visit: `http://localhost:8501`

## Docker Deployment

```bash
# Build image
docker build -f deployment/Dockerfile -t battery-api .

# Run container
docker run -p 5000:5000 battery-api
```

## Docker Compose

```bash
cd deployment
docker-compose up -d
```

For detailed AWS deployment instructions, see `DEPLOYMENT.md`

