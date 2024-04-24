import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def custom_scoring(references, sentences,):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        # Using the absolute value of compound score to check if greater than 0.5
        score = int(abs(vs['compound']) > 0.5)
        scores.append(score)
    return scores