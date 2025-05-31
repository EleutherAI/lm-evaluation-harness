import joblib

def predict(text):
    # Load the entire pipeline (vectorizer + classifier)
    pipeline = joblib.load("classifier_pipeline.joblib")
    
    # Predict label
    prediction = pipeline.predict([text])[0]
    
    # Convert numeric label to readable text
    label = "Truthful" if prediction == 1 else "Hallucination"
    print(f"Prediction: {label}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_with_classifier.py '<your_text_here>'")
    else:
        text = sys.argv[1]
        predict(text)