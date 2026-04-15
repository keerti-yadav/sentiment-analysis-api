from pydantic import BaseModel, Field
from typing import List

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic!"
            }
        }

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class SentimentOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float
    label: int

class BatchSentimentOutput(BaseModel):
    results: List[SentimentOutput]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class MetricsResponse(BaseModel):
    accuracy: float
    train_samples: int
    test_samples: int
    total_predictions: int