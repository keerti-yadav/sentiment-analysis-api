from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

from app.model import SentimentModel
from app.schemas import (
    TextInput, BatchTextInput, SentimentOutput, 
    BatchSentimentOutput, HealthResponse, MetricsResponse
)

# Initialize FastAPI
app = FastAPI(
    title="Sentiment Analysis API",
    description="AI-powered sentiment analysis with DevOps",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter('sentiment_predictions_total', 'Total predictions')
prediction_duration = Histogram('sentiment_prediction_duration_seconds', 'Prediction time')

# Load model
print("\n🚀 Starting Sentiment Analysis API...")
sentiment_model = SentimentModel()
print("🎉 API Ready!\n")

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/batch-predict", "/health", "/metrics"],
        "docs": "/docs"
    }

@app.post("/predict", response_model=SentimentOutput)
async def predict_sentiment(input_data: TextInput):
    """Analyze sentiment of single text"""
    try:
        start_time = time.time()
        
        label, confidence, sentiment = sentiment_model.predict(input_data.text)
        
        prediction_counter.inc()
        prediction_duration.observe(time.time() - start_time)
        
        return SentimentOutput(
            text=input_data.text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            label=label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=BatchSentimentOutput)
async def batch_predict_sentiment(input_data: BatchTextInput):
    """Analyze sentiment of multiple texts"""
    try:
        start_time = time.time()
        
        results = sentiment_model.batch_predict(input_data.texts)
        
        outputs = [
            SentimentOutput(
                text=text,
                sentiment=sentiment,
                confidence=round(confidence, 4),
                label=label
            )
            for text, (label, confidence, sentiment) in zip(input_data.texts, results)
        ]
        
        prediction_counter.inc(len(input_data.texts))
        prediction_duration.observe(time.time() - start_time)
        
        return BatchSentimentOutput(results=outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=sentiment_model.is_loaded()
    )

@app.get("/metrics")
async def get_metrics():
    """Get model metrics"""
    model_metrics = sentiment_model.get_metrics()
    return MetricsResponse(**model_metrics)

@app.get("/prometheus-metrics")
async def prometheus_metrics():
    """Prometheus metrics"""
    return Response(content=generate_latest(), media_type="text/plain")