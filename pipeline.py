import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Task 1: Load and understand
df = pd.read_csv('dataset.csv')
print("--- Task 1: Raw Data ---")
print(df.head())
df['length'] = df['text'].apply(len)
print(df[['text', 'length']].head())

# Task 2: Basic Cleaning
def basic_clean(text):
    text = text.lower() # 1. Lowercase
    text = re.sub(r'[^\w\s]', '', text) # 2. Remove punctuation
    text = re.sub(r'\d+', '', text) # 3. Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # 4. Remove extra whitespace
    return text

df['clean_text_basic'] = df['text'].apply(basic_clean)
print("\n--- Task 2: Basic Cleaning ---")
print(df[['text', 'clean_text_basic']].head())

# Task 3: Advanced Cleaning
def advanced_clean(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URLs
    text = re.sub(r'\S+@\S+', '', text) # Emails
    text = re.sub(r'<.*?>', '', text) # HTML
    text = re.sub(r'[^\w\s]', '', text) # Special chars (this also removes emojis)
    return text.strip()

df['clean_text_advanced'] = df['text'].apply(advanced_clean)

# Task 4: Stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w.lower() not in stop_words])

df['text_no_stopwords'] = df['clean_text_advanced'].apply(remove_stopwords)

# Task 5: Repeated Characters & Slang
def normalize_slang_and_chars(text):
    # Repeated chars (simple regex)
    text = re.sub(r'(.)\1{2,}', r'\1', text) 
    
    # Slang dict
    slang_dict = {'u': 'you', 'gr8': 'great'}
    words = text.split()
    new_words = [slang_dict.get(w.lower(), w) for w in words]
    return ' '.join(new_words)

df['text_normalized'] = df['text_no_stopwords'].apply(normalize_slang_and_chars)

# Task 6: Tokenization
print("\n--- Task 6: Tokenization (First 3 samples) ---")
for i in range(3):
    text = df['text'][i]
    print(f"Sample {i+1}:")
    print("Words:", word_tokenize(text))
    print("Sentences:", sent_tokenize(text))

# Task 7: Stemming
stemmer = PorterStemmer()
print("\n--- Task 7: Stemming ---")
sample_text = "running runs ran"
print(f"Original: {sample_text}")
print("Stemmed:", [stemmer.stem(w) for w in sample_text.split()])

# Task 8: Lemmatization
lemmatizer = WordNetLemmatizer()
print("\n--- Task 8: Lemmatization ---")
print(f"Original: {sample_text}")
print("Lemmatized:", [lemmatizer.lemmatize(w) for w in sample_text.split()])

# Task 9: Pipeline Function
def nlp_preprocess(text):
    text = text.lower()
    text = advanced_clean(text)
    words = text.split()
    text = ' '.join([w for w in words if w.lower() not in stop_words])
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(lemmatized)

df['final_clean_text'] = df['text'].apply(nlp_preprocess)
print("\n--- Task 9: Final Clean Text ---")
print(df[['text', 'final_clean_text']].head())
