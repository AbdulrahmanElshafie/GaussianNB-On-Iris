# Naive Bayes Classifier for Iris Dataset

This Python script utilizes the Scikit-learn library to build a Naive Bayes classifier for the Iris dataset. It employs the Gaussian Naive Bayes algorithm to classify iris flowers based on their sepal length and petal length features.

## Overview

- **Data Loading**: The code loads the Iris dataset, focusing on the first and third features (sepal length and petal length) for classification.
- **Data Preparation**: The dataset is shuffled for randomness in the data order.
- **Model Creation**: It initializes a Gaussian Naive Bayes classifier from Scikit-learn.
- **Data Splitting**: The script splits the data into training, validation, and test sets.
- **Training**: The model is trained using the training set.
- **Validation**: A validation process aims to enhance model accuracy.
- **Testing**: The trained model is tested on the test dataset for predictions.
- **Accuracy Calculation**: The accuracy of the model is calculated based on the test predictions.

## Functions

- `split()`: Divides the dataset into training, validation, and test sets.
- `train()`: Trains the Naive Bayes classifier using the training data.
- `validate()`: Validates the model to improve its accuracy using the validation data.
- `test()`: Tests the trained model on the test data and predicts labels.
- `calc_accuracy()`: Computes the accuracy of the model based on predicted and original labels.
- `train_validate_test_split()`: Orchestrates the entire process of training, validation, testing, and accuracy calculation.

## Usage

1. Ensure you have the necessary libraries installed (`Scikit-learn`, `mlxtend`, `matplotlib`).
2. Run the script.
3. The accuracy of the Naive Bayes classifier will be displayed.
4. The decision boundaries of the trained model will be visualized using the `plot_decision_regions()` function.

## Visualization

The code includes a visualization of the decision boundaries generated by the trained model using `plot_decision_regions()`. The regions represent the classification boundaries for Iris-setosa, Iris-versicolor, and Iris-virginica flowers based on their sepal and petal measurements.
