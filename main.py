import pandas as pd
from src.processing import nlp_preprocess

def main():
    # Load dataset
    df = pd.read_csv('data/dataset.csv')
    
    print("--- Original Data ---")
    print(df.head())
    
    # Apply pipeline
    df['final_clean_text'] = df['text'].apply(nlp_preprocess)
    
    print("\n--- Processed Data ---")
    print(df[['text', 'final_clean_text']].head())
    
    # Save the result
    df.to_csv('data/processed_dataset.csv', index=False)
    print("\nProcessed data saved to data/processed_dataset.csv")

if __name__ == "__main__":
    main()
