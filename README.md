# Personality-Based Career Recommendation System

A machine learning-based system that predicts personality types from text and suggests suitable careers and career paths based on personality traits and skills.

## 🎯 Features

- **Personality Prediction**: Predicts MBTI personality types from text using NLP and machine learning
- **Career Mapping**: Maps personality types to suitable careers
- **Career Path Recommendations**: Suggests career paths and transitions using graph-based algorithms
- **Skills-Based Suggestions**: Recommends careers based on skills using decision trees

## 🛠️ Tech Stack

- **Python 3.11+**
- **Machine Learning**:
  - scikit-learn
  - numpy
  - pandas
- **NLP**:
  - NLTK
  - spaCy
- **Graph Processing**:
  - networkx
- **Other**:
  - python-constraint
  - Jupyter Notebooks

## 📁 Project Structure

```
project/
├── data/                 # Dataset storage
├── src/                  # Source code
│   ├── personality.py    # Personality prediction
│   ├── role_mapping.py   # Career mapping
│   ├── graph.py          # Career path recommendations
│   ├── decision_tree.py  # Skills-based suggestions
│   └── text_preprocessor.py  # Text processing
├── notebooks/            # Jupyter notebooks
└── main.py              # Main application
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personality-career-recommender.git
   cd personality-career-recommender
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Prepare your dataset:
   - Place your text data in the `data/` directory
   - Format should be CSV with columns for text and personality type

2. Run the main application:
   ```bash
   python main.py
   ```

3. For development and testing:
   - Use Jupyter notebooks in the `notebooks/` directory
   - Run individual modules for specific functionality

## 📊 Model Architecture

1. **Text Preprocessing**:
   - Text cleaning and normalization
   - Tokenization and lemmatization
   - Feature extraction

2. **Personality Prediction**:
   - TF-IDF vectorization
   - Naive Bayes classification
   - Probability-based predictions

3. **Career Mapping**:
   - Rule-based personality to career mapping
   - Graph-based career path recommendations
   - Skills-based decision tree

## 🙏 Acknowledgments

- MBTI personality type system
- Open-source machine learning libraries
- Career development research