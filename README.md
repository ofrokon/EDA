# Iris Dataset Exploratory Data Analysis (EDA)

This repository contains a Python script that performs Exploratory Data Analysis (EDA) on the famous Iris dataset. It demonstrates various EDA techniques and visualizations commonly used in data science projects.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Visualizations](#visualizations)
5. [Additional Analysis](#additional-analysis)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview

This project showcases a comprehensive EDA process using the Iris dataset. It includes:

- Descriptive statistics
- Data visualization techniques
- Correlation analysis
- Dimensionality reduction using PCA
- Distribution analysis of features by species

The main goal is to provide a clear example of how to approach EDA in a data science project, using Python and popular data science libraries.

## Installation

To run this project, you need Python 3.6 or later. Follow these steps to set up your environment:

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/iris-eda-example.git
   cd iris-eda-example
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the EDA script, simply execute:

```
python iris_eda.py
```

This will generate several visualization files in the `visualizations/` directory and print some statistical information to the console.

## Visualizations

The script produces the following visualizations:

1. `histogram_sepal_length.png`: Distribution of sepal length across all species.
2. `boxplot_petal_length.png`: Comparison of petal length across different species.
3. `scatterplot_sepal.png`: Relationship between sepal length and sepal width, colored by species.
4. `correlation_heatmap.png`: Correlation matrix of all numeric features.
5. `pairplot.png`: Pairwise relationships between all features, separated by species.
6. `pca_visualization.png`: Projection of the dataset onto its first two principal components.
7. `feature_distributions.png`: Distribution of each feature, separated by species.

## Additional Analysis

The script also performs Principal Component Analysis (PCA) to reduce the dimensionality of the dataset. The results of this analysis are visualized in `pca_visualization.png`, and the explained variance ratios are printed to the console.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your suggested changes.

## License

This project is open source and available under the [MIT License](LICENSE).

---

For more information on the Iris dataset, visit the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

For questions or feedback, please open an issue in this repository.
