"""
Module for personality prediction using machine learning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from src.text_preprocessor import TextPreprocessor

from sklearn.feature_extraction.text import HashingVectorizer

class PersonalityPredictor:
    self.text_vectorizer = HashingVectorizer(
        n_features=1000,        # like max_features
        ngram_range=(1, 2),
        stop_words='english',
        alternate_sign=False    # important for MultinomialNB
    )

    # def __init__(self):
    #     self.preprocessor = TextPreprocessor()
    #     self.text_vectorizer = TfidfVectorizer(
    #         max_features=1000,
    #         ngram_range=(1, 2),
    #         stop_words='english'
    #     )
    #     self.model = MultinomialNB(alpha=0.1)
        
    def prepare_features(self, texts, fit_vectorizer=False):
        """Prepare features for training or prediction quickly with parallel processing."""
        # Preprocess texts in parallel
        processed_texts = Parallel(n_jobs=-1)(delayed(self.preprocessor.preprocess_text)(text) for text in texts)
        
        # Extract additional features in parallel
        additional_features = pd.DataFrame(
            Parallel(n_jobs=-1)(delayed(self.preprocessor.extract_features)(text) for text in texts)
        )

        # Transform text features
        if fit_vectorizer:
            text_features = self.text_vectorizer.fit_transform(processed_texts)
        else:
            text_features = self.text_vectorizer.transform(processed_texts)

        return text_features, additional_features

    
    def train(self, texts, personalities):
        """Train the personality prediction model."""
        # Prepare features and fit vectorizer
        print("Prepare features and fit vectorizer")
        text_features, additional_features = self.prepare_features(texts, fit_vectorizer=True)
        
        # Split data
        print("Split Data")
        X_train, X_test, y_train, y_test = train_test_split(
            text_features, personalities,
            test_size=0.2,
            random_state=42,
            stratify=personalities
        )
    
        # Train model
        print("Train model")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("y eval")
        y_pred = self.model.predict(X_test)
        print("acc eval")
        accuracy = accuracy_score(y_test, y_pred)
        print("Report ")
        report = classification_report(y_test, y_pred)
        
        return accuracy, report

    
    def predict(self, text):
        """Predict personality type from text."""
        # Prepare features (without fitting vectorizer again)
        text_features, _ = self.prepare_features([text], fit_vectorizer=False)
        
        # Predict
        personality = self.model.predict(text_features)[0]
        probabilities = self.model.predict_proba(text_features)[0]
        
        return personality, dict(zip(self.model.classes_, probabilities))

    def get_feature_importance(self):
        """Get the most important features for each personality type"""
        feature_names = self.text_vectorizer.get_feature_names_out()
        importance = {}
        
        for i, personality in enumerate(self.model.classes_):
            coef = self.model.feature_log_prob_[i]
            top_features = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True)[:10]
            importance[personality] = top_features
        
        return importance 