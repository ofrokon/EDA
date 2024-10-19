# Exploratory Data Analysis (EDA) Example Script
# This script demonstrates various EDA techniques using the Iris dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set the style for our plots
plt.style.use('seaborn-v0_8')

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Function to save plots
def save_plot(fig, filename):
    plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 1. Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())
print("\nValue Counts for Species:")
print(df['species'].value_counts())

# 2. Data Visualization

# 2.1 Histogram
def create_histogram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['sepal length (cm)'], bins=20, edgecolor='black')
    ax.set_title('Histogram of Sepal Length')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Frequency')
    save_plot(fig, 'histogram_sepal_length.png')

create_histogram()

# 2.2 Box Plot
def create_boxplot():
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='species', y='petal length (cm)', data=df, ax=ax)
    ax.set_title('Box Plot of Petal Length by Species')
    save_plot(fig, 'boxplot_petal_length.png')

create_boxplot()

# 2.3 Scatter Plot
def create_scatterplot():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df, ax=ax)
    ax.set_title('Scatter Plot: Sepal Length vs Sepal Width')
    save_plot(fig, 'scatterplot_sepal.png')

create_scatterplot()

# 3. Correlation Analysis
def create_correlation_heatmap():
    corr_matrix = df.drop('species', axis=1).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Iris Features')
    save_plot(fig, 'correlation_heatmap.png')

create_correlation_heatmap()

# 4. Pair Plot
# def create_pairplot():
#     fig = sns.pairplot(df, hue='species', height=2.5)
#     fig.fig.suptitle('Pair Plot of Iris Dataset', y=1.02)
#     save_plot(fig, 'pairplot.png')
# create_pairplot()

def create_pairplot():
    pair_grid = sns.pairplot(df, hue='species', height=2.5)
    pair_grid.fig.suptitle('Pair Plot of Iris Dataset', y=1.02)
    plt.tight_layout()
    plt.show()
    pair_grid.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.close(pair_grid.fig)

create_pairplot()

# 5. Dimensionality Reduction with PCA
def perform_pca():
    # Standardize the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.drop('species', axis=1))

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['species'] = df['species']

    # Plot PCA results
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, ax=ax)
    ax.set_title('PCA of Iris Dataset')
    ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%})')
    save_plot(fig, 'pca_visualization.png')

    print("\nPCA Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)

perform_pca()

# 6. Additional Analysis: Distribution of Features by Species
def plot_feature_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Distribution of Features by Species', fontsize=16)
    
    for i, feature in enumerate(iris.feature_names):
        row = i // 2
        col = i % 2
        sns.histplot(data=df, x=feature, hue='species', kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    save_plot(fig, 'feature_distributions.png')

plot_feature_distributions()

print("EDA completed. All visualizations have been saved as PNG files.")