#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Analysis and Visualization Project
---------------------------------------------
This script performs data loading, exploration, analysis, and visualization
on the Iris dataset, demonstrating key data science techniques.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def main():
    """Main function to execute the data analysis workflow."""
    print("="*80)
    print("DATA ANALYSIS AND VISUALIZATION PROJECT".center(80))
    print("="*80)
    
    # Task 1: Load and Explore the Dataset
    print("\n\n" + "TASK 1: LOAD AND EXPLORE THE DATASET".center(80))
    print("-"*80)
    
    # Load the Iris dataset
    print("Loading the Iris dataset...")
    try:
        # Load the Iris dataset from sklearn
        iris = load_iris()
        
        # Create a DataFrame
        column_names = iris.feature_names
        df = pd.DataFrame(data=iris.data, columns=column_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        # Add datetime index for time series visualization
        # Let's create a fake date range to demonstrate time series
        dates = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        df['date'] = dates
        df['measurement_value'] = np.random.normal(loc=5, scale=1, size=len(df))
        
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Display the first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Display dataset information
    print("\nDataset information:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())
    
    # Intentionally introduce some missing values to demonstrate cleaning
    print("\nIntroducing some missing values for demonstration...")
    df.loc[10:15, 'sepal length (cm)'] = np.nan
    
    print("Missing values after introduction:")
    print(df.isnull().sum())
    
    # Clean the dataset
    print("\nCleaning the dataset...")
    # Fill missing values with median
    df['sepal length (cm)'] = df['sepal length (cm)'].fillna(df['sepal length (cm)'].median())
    
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    
    # Task 2: Basic Data Analysis
    print("\n\n" + "TASK 2: BASIC DATA ANALYSIS".center(80))
    print("-"*80)
    
    # Compute basic statistics
    print("\nBasic statistics of numerical columns:")
    print(df.describe())
    
    # Group by species and compute mean for each group
    species_grouped = df.groupby('species').mean()
    print("\nMean values for each species:")
    print(species_grouped)
    
    # Correlation analysis
    print("\nCorrelation matrix:")
    print(df.select_dtypes(include=[np.number]).corr().round(2))
    
    # Additional analysis: Find min/max values per species
    print("\nMinimum and maximum sepal length by species:")
    min_max = df.groupby('species')['sepal length (cm)'].agg(['min', 'max'])
    print(min_max)
    
    # Task 3: Data Visualization
    print("\n\n" + "TASK 3: DATA VISUALIZATION".center(80))
    print("-"*80)
    
    # Create a directory for saving figures
    print("\nCreating visualizations...")
    
    # 1. Line chart showing trends over time
    plt.figure(figsize=(10, 6))
    
    # Group by date and species for the time series
    time_series = df.groupby(['date', 'species'])['measurement_value'].mean().unstack()
    
    for species in time_series.columns:
        plt.plot(time_series.index[:30], time_series[species][:30], label=species)
    
    plt.title('Measurement Values Over Time by Species', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Measurement Value', fontsize=12)
    plt.legend(title='Species')
    plt.tight_layout()
    plt.show()
    
    # 2. Bar chart comparing sepal length across species
    plt.figure(figsize=(10, 6))
    
    # Calculate mean sepal length by species
    sepal_by_species = df.groupby('species')['sepal length (cm)'].mean().sort_values()
    
    # Create bar chart
    sns.barplot(x=sepal_by_species.index, y=sepal_by_species.values)
    
    plt.title('Average Sepal Length by Species', fontsize=16)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Average Sepal Length (cm)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # 3. Histogram of sepal width distribution
    plt.figure(figsize=(10, 6))
    
    # Create histograms for each species
    for species in df['species'].unique():
        sns.histplot(df[df['species'] == species]['sepal width (cm)'], 
                    kde=True, label=species, alpha=0.6)
    
    plt.title('Distribution of Sepal Width by Species', fontsize=16)
    plt.xlabel('Sepal Width (cm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Species')
    plt.tight_layout()
    plt.show()
    
    # 4. Scatter plot of sepal length vs petal length
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot with different colors for each species
    for species, color in zip(df['species'].unique(), ['blue', 'green', 'red']):
        subset = df[df['species'] == species]
        plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                   label=species, color=color, alpha=0.7)
    
    plt.title('Sepal Length vs Petal Length by Species', fontsize=16)
    plt.xlabel('Sepal Length (cm)', fontsize=12)
    plt.ylabel('Petal Length (cm)', fontsize=12)
    plt.legend(title='Species')
    plt.tight_layout()
    plt.show()
    
    # 5. Bonus visualization: Pairplot for all features
    plt.figure(figsize=(12, 10))
    
    # Create a pairplot
    pair_plot = sns.pairplot(df, hue='species', height=2.5, 
                           vars=[col for col in df.columns if '(cm)' in col])
    pair_plot.fig.suptitle('Pairwise Relationships in Iris Dataset', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 6. Bonus visualization: Box plots
    plt.figure(figsize=(12, 6))
    
    # Create box plots for all numerical features
    feature_cols = [col for col in df.columns if '(cm)' in col]
    melted_df = pd.melt(df, id_vars=['species'], value_vars=feature_cols, 
                      var_name='feature', value_name='value')
    
    sns.boxplot(x='feature', y='value', hue='species', data=melted_df)
    plt.title('Distribution of Features by Species', fontsize=16)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Value (cm)', fontsize=12)
    plt.legend(title='Species')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Summary of findings
    print("\n\n" + "SUMMARY OF FINDINGS".center(80))
    print("-"*80)
    print("\nKey observations from the analysis:")
    print("1. The Iris dataset contains three species: setosa, versicolor, and virginica.")
    print("2. There is a clear distinction between setosa and the other two species based on petal dimensions.")
    print("3. Versicolor and virginica have some overlap but can generally be distinguished by feature combinations.")
    print("4. Petal length and width show the strongest correlation.")
    print("5. Setosa has the smallest petals but wider sepals compared to other species.")
    
    print("\nThank you for reviewing this data analysis project!")
    

if __name__ == "__main__":
    main()