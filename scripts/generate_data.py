import argparse
import pandas as pd
import numpy as np
import random
from datetime import timedelta

def generate_synthetic_data(input_file, output_file, num_samples=None):
    # the line of code below loads the original data
    original_data = pd.read_csv(input_file)
    
    # If num_samples is not provided, the same number of rows as the original dataset will be used
    if num_samples is None:
        num_samples = len(original_data)
    
    synthetic_data = pd.DataFrame()

    # the code below generates synthetic numeric data
    numeric_columns = original_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        mean = original_data[col].mean()
        std = original_data[col].std()
        min_val = original_data[col].min()
        max_val = original_data[col].max()
        synthetic_data[col] = np.clip(
            np.random.normal(loc=mean, scale=std, size=num_samples),
            a_min=min_val,
            a_max=max_val
        )
    
    # the code below generates synthetic categorical data
    categorical_columns = original_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        values = original_data[col].dropna().unique()
        probabilities = original_data[col].value_counts(normalize=True, dropna=True).values
        synthetic_data[col] = np.random.choice(values, size=num_samples, p=probabilities)
    
    # the code below generates synthetic date data if present
    if 'FinancialsDate' in original_data.columns:
        date_col = pd.to_datetime(original_data['FinancialsDate'], errors='coerce')
        min_date = date_col.min()
        max_date = date_col.max()
        synthetic_data['FinancialsDate'] = [
            min_date + timedelta(days=random.randint(0, (max_date - min_date).days))
            for _ in range(num_samples)
        ]
    
    # Saving the synthetic dataset to a CSV file
    synthetic_data.to_csv(output_file, index=False)
    print(f"Synthetic dataset saved to: {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset from an input CSV file.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the synthetic dataset.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples in the synthetic dataset (default is the same as the input dataset).")
    
    args = parser.parse_args()

    # Call the function to generate synthetic data
    generate_synthetic_data(args.input_file, args.output_file, args.num_samples)

if __name__ == "__main__":
    main()
