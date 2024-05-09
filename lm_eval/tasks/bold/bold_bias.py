import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def custom_scoring(references, predictions, **kwargs):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for sentence in predictions:
        vs = analyzer.polarity_scores(sentence)
        # Using the absolute value of compound score to check if greater than 0.5
        score = int(abs(vs['compound']) > 0.5)
        scores.append(score)
    return sum(scores)