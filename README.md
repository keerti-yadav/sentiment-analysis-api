# Sentiment Analysis API

AI-powered sentiment analysis with full DevOps pipeline.

## Features
- ✅ 4 REST API endpoints
- ✅ ML model (85%+ accuracy)
- ✅ Docker containerization
- ✅ CI/CD pipeline
- ✅ Automated testing
- ✅ Prometheus metrics

## Quick Start

### Local Development
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\train.py
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

### Docker
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## API Endpoints

- `POST /predict` - Single text sentiment
- `POST /batch-predict` - Multiple texts
- `GET /health` - Health check
- `GET /metrics` - Model metrics

## Tech Stack
- FastAPI
- scikit-learn
- Docker
- GitHub Actions
- Prometheus