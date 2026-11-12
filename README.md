# Assignment 17: Text Cleaning, Preprocessing & NLP Pipeline

**Student Name:** Patan Khalilulla Khan

## Overview
This project demonstrates a complete NLP preprocessing pipeline starting from raw text to model-ready text.

## Dataset
- **Source:** Created a small custom dataset of 25 product reviews.
- **File:** `dataset.csv`

## Pipeline Tasks
1.  **Basic Text Cleaning:** Lowercasing, removing punctuation, removing numbers, removing extra whitespace.
2.  **Advanced Text Cleaning:** Removing URLs, emails, HTML tags, special characters.
3.  **Handling Stopwords:** Using NLTK's list of English stopwords.
4.  **Tokenization:** Word and sentence tokenization.
5.  **Stemming & Lemmatization:** Porter Stemmer and WordNet Lemmatizer.

## Observations & Insights
1.  **Difference between basic and advanced cleaning:** Basic cleaning focuses on structural issues like casing, punctuation, and white spaces. Advanced cleaning addresses noise like URLs, HTML tags, and email addresses that do not contribute to the semantic meaning of the text.
2.  **Why lemmatization is preferred over stemming:** Stemming reduces words to their base form by cutting off suffixes, which can result in non-words (e.g., "running" -> "run", but "universally" -> "univers"). Lemmatization uses vocabulary and morphological analysis to return a valid word (lemma).
3.  **Importance of preprocessing:** Proper preprocessing removes noise and reduces the complexity of the data, which significantly improves the performance and accuracy of NLP models.

## How to Run
Ensure you have the required dependencies installed:
`pip install pandas nltk`

Then, run the main script:
`python pipeline.py`
