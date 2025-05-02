"""
Test script to verify package imports
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import networkx as nx
from constraint import *

def main():
    print("Testing package imports...")
    
    # Test numpy
    arr = np.array([1, 2, 3])
    print(f"NumPy test: {arr}")
    
    # Test pandas
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"Pandas test:\n{df}")
    
    # Test scikit-learn
    X = [[1], [2], [3]]
    y = [0, 1, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Scikit-learn test: Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Test networkx
    G = nx.Graph()
    G.add_edge(1, 2)
    print(f"NetworkX test: Graph has {G.number_of_edges()} edges")
    
    # Test nltk
    nltk.download('punkt')
    print("NLTK test: Successfully downloaded 'punkt'")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    main() 