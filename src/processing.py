import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure necessary NLTK data
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

setup_nltk()

# Basic Cleaning
def basic_clean(text):
    text = text.lower() # 1. Lowercase
    text = re.sub(r'[^\w\s]', '', text) # 2. Remove punctuation
    text = re.sub(r'\d+', '', text) # 3. Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # 4. Remove extra whitespace
    return text

# Advanced Cleaning
def advanced_clean(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URLs
    text = re.sub(r'\S+@\S+', '', text) # Emails
    text = re.sub(r'<.*?>', '', text) # HTML
    text = re.sub(r'[^\w\s]', '', text) # Special chars
    return text.strip()

# Handling Stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w.lower() not in stop_words])

# Handling Repeated Characters & Slang
def normalize_slang_and_chars(text):
    text = re.sub(r'(.)\1{2,}', r'\1', text) 
    slang_dict = {'u': 'you', 'gr8': 'great'}
    words = text.split()
    new_words = [slang_dict.get(w.lower(), w) for w in words]
    return ' '.join(new_words)

# Pipeline Function
lemmatizer = WordNetLemmatizer()
def nlp_preprocess(text):
    text = text.lower()
    text = advanced_clean(text)
    words = text.split()
    text = ' '.join([w for w in words if w.lower() not in stop_words])
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(lemmatized)
