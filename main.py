'''
Main script for the personality-based career recommendation system
'''
import pandas as pd
from src.data_loader import DataLoader
from src.personality import PersonalityPredictor
from src.role_mapping import RoleMapper
from src.graph import CareerGraph
from src.decision_tree import CareerDecisionTree


def main():
    # Step 1: Load and preprocess data
    # print("Loading and preprocessing data...")
    data_loader = DataLoader('data/mbti_1.csv')
    X_texts, y = data_loader.load_and_preprocess()

    # Step 2: Train personality predictor
    # print("Training PersonalityPredictor...")
    personality_predictor = PersonalityPredictor()
    # print("accuracy, report = personality_predictor.train(X_texts, y)")
    accuracy, report = personality_predictor.train(X_texts, y)
    print(f"Personality prediction accuracy: {accuracy:.2%}\n")
    print("Classification report:\n", report)

    # Step 3: Prepare feature input for decision tree
    # Here we derive skill features from personality (example mapping)
    # You will refine this mapping based on real feature extraction logic
    skill_map = {
        'Analyst': [5, 3, 2, 1],
        'Strategist': [4, 2, 5, 3],
        'Advocate': [3, 5, 2, 4],
        # add mapping for all role names
    }

    # Initialize other components
    
    role_mapper = RoleMapper()
    
    career_graph = CareerGraph()
    
    career_tree = CareerDecisionTree()

    # Train decision tree with dummy skill data for demonstration
    # In practice, build X_skills matrix and y_roles vector from your data
    
    X_skills = pd.DataFrame(list(skill_map.values()), columns=career_tree.feature_names)
    
    y_roles = list(skill_map.keys())
    
    career_tree.train(X_skills, y_roles)

    # Step 4: Example of full pipeline for a new user input
    sample_text = X_texts.iloc[0]
    print(f"\nSample user text: {sample_text[:100]}...")

    predicted_personality, probs = personality_predictor.predict(sample_text)
    print(f"Predicted personality: {predicted_personality}")

    role = role_mapper.get_role(predicted_personality)
    print(f"Mapped role: {role}")

    # Decision tree suggests a career based on skill vector
    skills = skill_map.get(role, [3, 3, 3, 3])  # fallback average skills
    suggested_career = career_tree.predict_career(skills)
    print(f"Decision tree career suggestion: {suggested_career}")

    # Graph-based additional recommendations
    more_careers = career_graph.get_recommendations(suggested_career)
    print(f"Graph-based related careers: {more_careers}")

    print("\nCareer Recommendation Pipeline completed.")


if __name__ == "__main__":
    main()
