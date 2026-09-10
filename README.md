# NLP-Legal-Case-Categorization
A natural language processing (NLP) and machine learning pipeline designed to automatically classify Indian legal cases based on their judgment texts.

## Overview
This project processes raw, unstructured legal judgment documents and categorizes them into specific legal domains using traditional machine learning classifiers. It demonstrates a complete data science workflow from text preprocessing to model evaluation.

## Features & Pipeline
* **Data Preprocessing:** Utilizes `pandas` to clean the `Indian_law_NLP` dataset, handling missing values and dropping extraneous columns.
* **Text Normalization:** Implements a text preprocessing pipeline using `re` and `nltk` to lowercase text, remove non-alphabetic characters, strip stopwords, and apply WordNet lemmatization[cite: 2].
* **Feature Engineering:** Extracts features using `TfidfVectorizer` with bigrams (n-gram range 1-2) and a 5,000-word vocabulary limit[cite: 2].
* **Model Training:** Trains and evaluates three separate classification models using `scikit-learn`: `LogisticRegression` (with balanced class weights), `RandomForestClassifier`, and `DecisionTreeClassifier`[cite: 2].
* **Evaluation:** Outputs accuracy scores, classification reports, and confusion matrices to compare model performance[cite: 2].

## Tech Stack
* Python
* scikit-learn
* NLTK (Natural Language Toolkit)
* Pandas & NumPy
