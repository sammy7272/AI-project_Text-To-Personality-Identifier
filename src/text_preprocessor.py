"""
Module for text preprocessing and feature extraction
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class TextPreprocessor:
    def __init__(self):

        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('punkt_tab')

        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('averaged_perceptron_tagger_eng')   

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                   "N": wordnet.NOUN,
                   "V": wordnet.VERB,
                   "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @ references and '#' from hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        return text
    
    def preprocess_text(self, text):
        """Full text preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                pos = self.get_wordnet_pos(token)
                lemma = self.lemmatizer.lemmatize(token, pos)
                processed_tokens.append(lemma)
        
        return ' '.join(processed_tokens)
    
    def extract_features(self, text):
        """Extract additional features from text"""
        processed_text = self.preprocess_text(text)
        tokens = word_tokenize(processed_text)
        
        features = {
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'sentence_count': len(nltk.sent_tokenize(processed_text)),

        }
        
        return features 