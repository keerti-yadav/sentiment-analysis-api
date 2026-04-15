import joblib
import os
from typing import List, Tuple
import numpy as np

class SentimentModel:
    def __init__(self, model_path: str = "models/sentiment_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.metrics = None
        self.prediction_count = 0
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        print(f"🔄 Loading model from {self.model_path}...")
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            
            metrics_path = "models/metrics.joblib"
            if os.path.exists(metrics_path):
                self.metrics = joblib.load(metrics_path)
            
            print(f"✅ Model loaded successfully!")
        else:
            raise FileNotFoundError(f"❌ Model not found at {self.model_path}")
    
    def predict(self, text: str) -> Tuple[int, float, str]:
        """Predict sentiment for single text"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = float(np.max(probabilities))
        
        sentiment_text = "positive" if prediction == 1 else "negative"
        
        self.prediction_count += 1
        
        return int(prediction), confidence, sentiment_text
    
    def batch_predict(self, texts: List[str]) -> List[Tuple[int, float, str]]:
        """Predict sentiment for multiple texts"""
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            confidence = float(np.max(probs))
            sentiment_text = "positive" if pred == 1 else "negative"
            results.append((int(pred), confidence, sentiment_text))
        
        self.prediction_count += len(texts)
        return results
    
    def get_metrics(self) -> dict:
        """Get model metrics"""
        base_metrics = self.metrics if self.metrics else {}
        base_metrics['total_predictions'] = self.prediction_count
        return base_metrics
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None