"""
Module for decision tree-based career suggestions
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class CareerDecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.feature_names = ['analytical_skills', 'communication_skills',
                            'leadership_skills', 'creative_skills']
    
    def train(self, X, y):
        """Train the decision tree model"""
        self.model.fit(X, y)
    
    def predict_career(self, skills):
        """Predict suitable career based on skills"""
        # Ensure skills are in the correct format
        if len(skills) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} skills, got {len(skills)}")
        
        # Convert skills to numpy array and reshape for prediction
        skills_array = np.array(skills).reshape(1, -1)
        return self.model.predict(skills_array)[0]
    
    def get_feature_importance(self):
        """Get the importance of each skill in career prediction"""
        return dict(zip(self.feature_names, self.model.feature_importances_)) 