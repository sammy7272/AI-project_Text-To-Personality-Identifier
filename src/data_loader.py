import pandas as pd
import re

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def clean_text(self, text):
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        # Remove "|||" separators
        text = text.replace("|||", " ")
        # Lowercase
        text = text.lower()
        # Remove non-alphabetic characters (optional)
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        # Clean posts
        df['cleaned_posts'] = df['posts'].apply(self.clean_text)
        X = df['cleaned_posts']
        y = df['type']
        return X, y
