import pandas as pd
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data():
    """Load movie reviews dataset"""
    print("📚 Loading IMDB movie reviews...")
    from nltk.corpus import movie_reviews
    
    documents = []
    labels = []
    
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(movie_reviews.raw(fileid))
            labels.append(1 if category == 'pos' else 0)
    
    df = pd.DataFrame({
        'text': documents,
        'sentiment': labels
    })
    
    print(f"✅ Loaded {len(df)} reviews")
    print(f"   Positive: {sum(df['sentiment'] == 1)}")
    print(f"   Negative: {sum(df['sentiment'] == 0)}")
    
    return df

def train_model():
    """Train sentiment analysis model"""
    print("\n🚀 Starting model training...\n")
    
    # Load data
    df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['sentiment']
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Create model pipeline
    print("\n🔧 Building model pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # Train
    print("🎯 Training model (this takes 1-2 minutes)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("📈 Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✨ MODEL TRAINED SUCCESSFULLY!")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"\n📋 Detailed Performance:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Positive']))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/sentiment_model.joblib')
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    joblib.dump(metrics, 'models/metrics.joblib')
    
    print(f"\n💾 Model saved to: models/sentiment_model.joblib")
    print(f"   Size: {os.path.getsize('models/sentiment_model.joblib') / 1024 / 1024:.2f} MB")
    
    # Test predictions
    print("\n🧪 Testing predictions:")
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time and money.",
        "It was okay, nothing special."
    ]
    
    for text in test_texts:
        pred = pipeline.predict([text])[0]
        prob = pipeline.predict_proba([text])[0]
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = max(prob) * 100
        print(f"   '{text[:50]}...'")
        print(f"   → {sentiment} ({confidence:.1f}% confident)\n")
    
    return pipeline, metrics

if __name__ == "__main__":
    train_model()