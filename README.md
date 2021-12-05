# Predicting whether an instance of breast cancer is malignant or benign

The project aims at analysing data on breast cancer obtained from Wisconsin Breast Cancer Database (WBCD) in order to predict whether a given instance of cancer is malignant or benign.

This repository contains the source code as well as the visualisations and models created as a part of the Final Project for the Data Analytics course (UE19CS312) at PES University.

The Final Report for the document can be found [here](https://drive.google.com/file/d/1bvMVRW-fg9vHtm-FSPI9T7EkJz2u46Wi/view?usp=sharing). <br>
The Video Presentation can be viewed [here](https://drive.google.com/file/d/1YQ_X-XuokOeZ9NOVt42OK88ZiW2G3xQ-/view?usp=drivesdk).

## Team Members

[Smruthi Gowtham](https://github.com/smruthig) <br>
[Snigdha Sinha](https://github.com/Snigdha-Sinha) <br>
[Vridhi Goyal](https://github.com/Vridhi-Goyal) <br>
[Yashi Chawla](https://github.com/Yashi-Chawla) <br>

## Directory Structure

```
kepler-exoplanet-analysis
├── data
    ├── WBCD.xls
    ├── train.csv
    └── test.csv


├── docs
    ├── Project Guidelines and Requirements Documents

├── models
    ├── classifier.py
    ├── classifier.sh
    ├── model_results.csv
    ├── neuralnetwork.py
    ├── neuralnetwork.sh
    └── parse_results.py

├── notebook
    └── Notebooks containing data preprocessing, visualisation and analysis

├── plots
    └── All plots based on visualisation

├── presentation
    └── Presentation and Video

├── report
    ├── Final Report
    ├── Final Report Plagiarism Check
    ├── Literature Review
    └── Literature Review Plagiarism Check

└── results
    └── Results of all six models with all permutations of preprocessing


```

## How to run the code?

1. Clone this repository

```bash
git clone https://github.com/smruthig/Cancer_Data_Analytics
```

2. Classification models can be trained using the classifier.py

```
python3 classifier.py [-c number of splits -n normalize boolean -s standardize boolean -pca n_components]
```

3. In order to try out all preprocessing pipelines on the classifier, can use the bash script.

```
./classifier.sh
```

4. Neural network can be trained using

```
python3 neuralnetwork.py
```

5. All preprocessing pipelines can be tried on the neuralnetwork using the bash script.

```
./neuralnetwork.sh
```

6. Run any notebook by executing all the code cells.

## About 
Breast cancer is a prevalent incursive cancer among women. It is one among the primary causes of death related to cancer,
which can be classified as Malignant or Benign. Breast cancer diagnosis is time consuming and due to its gravity, it is imperative to design a solution to automate the process of identification of the same in its early stages so that it can be treated efficiently. Breast Cancer prediction aims at extracting features from the given samples and predicting it as Benign or Malignant. The dataset chosen is extracted from the Wisconsin Breast Cancer Dataset. This implementation compares six basic models namely, Logistic Regression, SVC, AdaBoost, Neural Network Nearest Neighbours and Random Forest through performance metrics like, F1-score and F1-stratified score. The results reveal that the highest performing model for this problem statement is Logistic Regression with an F1-score of 0.9788 and F1-stratified score of 0.9822.

## Dataset
The dataset that has been used in our project is The Wisconsin Breast Cancer Dataset (WBCD),  acquired from the repository of UCI Machine learning, is a benchmark dataset. The dataset is distributed over 37.25\% cancerous samples and 62.75\% non-cancerous samples.
The Wisconsin Breast cancer Dataset contains 569 instances and 32 features that provide accurate information regarding the
diagnosis of breast cancer. The columns of the dataset represent the features of the cell nuclei found in the digitised image of fine needle aspirates (FNA) of breast mass.

## Predictive Modelling

This study's primary aim is binary classification of the given samples of cell nuclei as benign or malignant.

Grid Search was performed on all models to find out the best parameters, with various combinations of preprocessing.
PCA was performed to reduce the dimensionality of the dataset. 
Various preprocessing combinations were tried out with every model to determine the best results. 
Logistic Regression with preprocessing as standardization and PCA gave the best results with F1 score as 98% on the dataset. 

The six models used are

1. Support Vector Machine
2. Random Forest
3. AdaBoost
4. Neural Network.
5. Logistic Regression
6. KNN

## Evaluation of Model Performance

Considering the size of the dataset, we implemented cross validation on 15 splits. 
Grid search was implemented to get the best set of hyperparameters. 
Accuracy,  Roc-Auc score, Mean Squared Error were calculated based on the best parameters. 

Since the dataset is small, we use both -

- A non-stratified split
- A stratified split

This is to ensure that within each fold the number of positive and negative examples are equal. We measure our classifier’s performance across each split and finally take the mean of the performance achieved.

## Model Results
|                      | Accuracy | F1 score | RocAucScore | MSE      |
|----------------------|----------|----------|-------------|----------|
| Logisitic Regression | 0.982456 | 0.974359 | 0.980513    | 0.132453 |
| SVC                  | 0.991228 | 0.987342 | 0.9875      | 0.093689 |
| Adaboost             | 0.964912 | 0.95     | 0.956565    | 0.187317 |
| Random Forest        | 0.973684 | 0.962025 | 0.968243    | 0.16221  |
| KNN                  | 0.973684 | 0.961039 | 0.973684    | 0.162221 |