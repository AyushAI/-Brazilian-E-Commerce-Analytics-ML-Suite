import pandas as pd
import os

def main():
    print("Loading original dataset to generate a test sample...")
    df = pd.read_csv('final_outlierTreated.csv')
    
    # We want a mix of churn and non-churn
    print("Sampling rows...")
    churn_samples = df[df['Churn'] == 1].sample(10, random_state=42)
    non_churn_samples = df[df['Churn'] == 0].sample(10, random_state=42)
    
    # Combine and shuffle
    test_sample = pd.concat([churn_samples, non_churn_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Define columns to keep for testing (must match our pipeline logic)
    drop_cols = ['customer_unique_id', 'customer_city', 'customer_state', 'geolocation_lat', 'geolocation_lng']
    test_sample = test_sample.drop(columns=[col for col in drop_cols if col in test_sample.columns], errors='ignore')

    output_file = 'test_sample.csv'
    test_sample.to_csv(output_file, index=False)
    print(f"Test sample successfully saved to {output_file} with {len(test_sample)} rows.")
    
    print("\nSample Preview:")
    print(test_sample.head())

if __name__ == "__main__":
    main()
