import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import spacy

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text, stop_words):
    """
    Lowercases, removes punctuation and stopwords, and tokenizes text.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text, language='portuguese')
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(tokens)

def get_sentiment(score):
    """
    Maps 1-5 review score to Negative (0), Neutral (1), Positive (2).
    """
    if score <= 2:
        return 0 # Negative
    elif score == 3:
        return 1 # Neutral
    else:
        return 2 # Positive

def main():
    print("Loading Dataset...")
    df = pd.read_csv('/home/ayush-wase/E Commerce ML/NLP/olist_order_reviews_dataset.csv')
    
    print(f"Original shape: {df.shape}")
    # Drop rows without text
    df = df.dropna(subset=['review_comment_message']).copy()
    print(f"Shape after dropping null text reviews: {df.shape}")
    
    # We will sample to speed up demonstration if the dataset is huge, else take all.
    # The Brazilian dataset can be large (100k reviews), taking a workable chunk for NLP metrics
    sample_size = min(10000, len(df))
    df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    print("\n--- 1. Text Preparation ---")
    stop_words = set(stopwords.words('portuguese'))
    
    print("Cleaning text (lowercasing, punctuation removal, stopword removal)...")
    df['clean_text'] = df['review_comment_message'].apply(lambda x: clean_text(x, stop_words))
    
    print("Sample Output:")
    print(df[['review_comment_message', 'clean_text']].head(3))
    
    print("\n--- 2. TF-IDF Representation ---")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    print(f"TF-IDF Matrix Shape (Reviews, Features): {X_tfidf.shape}")
    
    print("\n--- 3. Sentiment Analysis ---")
    df['sentiment'] = df['review_score'].apply(get_sentiment)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['sentiment'], test_size=0.2, random_state=42)
    
    clf = MultinomialNB()
    print("Training Multinomial Naive Bayes Classifier on TF-IDF features...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    target_names = ['Negative (1-2)', 'Neutral (3)', 'Positive (4-5)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\n--- 4. Named Entity Recognition (NER) ---")
    print("Loading SpaCy Portuguese Model...")
    try:
        nlp = spacy.load("pt_core_news_sm")
    except OSError:
        print("SpaCy model 'pt_core_news_sm' not found. Please install using: python -m spacy download pt_core_news_sm")
        return
        
    print("Extracting Entities from a sample of reviews...")
    # Take a small sub-sample that likely contains entities (longer reviews)
    df['text_len'] = df['review_comment_message'].str.len()
    long_reviews = df.sort_values(by='text_len', ascending=False).head(20)['review_comment_message']
    
    entities_found = False
    for text in long_reviews:
        doc = nlp(text)
        if doc.ents:
            entities_found = True
            print(f"\nReview: '{text[:100]}...'")
            for ent in doc.ents:
                print(f"  -> Entity: {ent.text} | Label: {ent.label_}")
                
    if not entities_found:
         print("No prominent named entities (ORG, LOC, PER) found in the considered sample.")
         
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
