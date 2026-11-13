# Titanic Survival Prediction using C4.5-Style Decision Tree

## Project Overview

This project implements a **C4.5-style Decision Tree Classifier from scratch using Python** to predict passenger survival on the Titanic dataset.

The implementation uses entropy, information gain, gain ratio, and threshold-based splitting to construct a classification tree. Unlike using a pre-built Decision Tree classifier, the splitting and prediction logic are implemented manually.

The project also includes decision tree visualization and classification performance evaluation.

## Objectives

- Implement a C4.5-style Decision Tree algorithm from scratch.
- Understand entropy, information gain, and gain ratio.
- Handle categorical and numerical attributes.
- Determine optimal splitting thresholds for numerical features.
- Visualize the generated decision tree.
- Evaluate the model using standard classification metrics.

## Dataset

The project uses the Titanic passenger dataset to predict whether a passenger survived.

**Target Variable:** `Survived`

- `0` = Not Survived
- `1` = Survived

**Input Features:**

| Feature | Description |
|---|---|
| Pclass | Passenger ticket class |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Passenger ticket fare |
| Embarked | Port of embarkation |

## Methodology

### 1. Data Preprocessing

- Convert relevant columns into numeric values.
- Remove records with missing target labels.
- Replace missing Age and Fare values using median imputation.
- Replace missing categorical values using mode imputation.
- Treat passenger class as a categorical feature.
- Split the dataset into 80% training and 20% testing data using stratified sampling.

### 2. Entropy Calculation

Entropy measures the uncertainty or impurity of a dataset.

**Formula:**

H(S) = -Σ pᵢ log₂(pᵢ)

A lower entropy indicates a more homogeneous dataset.

### 3. Information Gain

Information Gain measures how much uncertainty is reduced after splitting the dataset using an attribute.

**Formula:**

Gain(S, A) = H(S) - Σ (|Sᵥ| / |S|) H(Sᵥ)

### 4. Gain Ratio

Gain Ratio normalizes Information Gain using Split Information.

**Formula:**

Gain Ratio = Information Gain / Split Information

This helps reduce the preference of information gain toward attributes with many distinct values.

### 5. Decision Tree Construction

The decision tree is constructed recursively using the following approach:

1. Calculate entropy for the current dataset.
2. Evaluate possible splits for available features.
3. Calculate information gain and gain ratio.
4. Consider attributes meeting the average information gain criterion.
5. Select the attribute with the highest gain ratio among those candidates.
6. Split the dataset and recursively construct child nodes.
7. Stop when the stopping conditions are reached.

**Tree Configuration:**

- Maximum Depth: 4
- Minimum Samples: 20
- Splitting Criterion: Gain Ratio
- Numerical Splitting: Binary threshold-based
- Categorical Splitting: Multiway

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Model Evaluation

The model is evaluated on unseen test data using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics help assess the classification performance and identify incorrect survival predictions.

## Visualizations

The project generates:

**1. Decision Tree Visualization**

Displays the hierarchical structure of the trained decision tree, including splitting attributes, gain ratios, branches, and predicted classes.

**2. Confusion Matrix**

Visualizes the model's correct and incorrect predictions for survived and non-survived passengers.

## How to Run

1. Clone or download the repository.
2. Install the required Python libraries using `pip install -r requirements.txt`.
3. Run the program using `python c45_titanic.py`.
4. Examine the root-node gain ratio calculations and evaluation metrics in the console.
5. View the decision tree and confusion matrix visualizations.

An internet connection may be required to load the dataset from Google Drive.

## Implementation Notes

This project is an educational **C4.5-style implementation**, not a complete reproduction of the original C4.5 algorithm.

The implementation supports gain ratio selection, numerical threshold splits, and categorical multiway splits. However, it does not implement C4.5 pruning or its complete missing-value handling procedure.

Additionally, missing-value imputation is currently performed before the train-test split. For more rigorous evaluation, imputation statistics should be calculated using training data only.

## Conclusion

This project demonstrates how a decision tree classifier can be implemented without relying on a pre-built classification model.

It provides practical understanding of entropy, information gain, gain ratio, recursive tree construction, decision boundaries, and classification evaluation using a real-world dataset.
