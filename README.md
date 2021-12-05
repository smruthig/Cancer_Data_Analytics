# Predicting whether an instance of breast cancer is malignant or benign

The project aims at analysing data on breast cancer obtained from Wisconsin Breast Cancer Database (WBCD) in order to predict whether a given instance of cancer is malignant or benign.

This repository contains the source code as well as the visualisations and models created as a part of the Final Project for the Data Analytics course (UE19CS312) at PES University.

The Final Report for the document can be found [here](). <br>
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

## About the problem statement

## Dataset

## Predictive Modelling

The six models used are

1. Support Vector Machine
2. Random Forest
3. AdaBoost
4. Neural Network.
5. Logistic Regression
6. KNN

## Evaluation of Model Performance

Since the dataset is small, we again use both -

- A non-stratified split
- A stratified split

This is to ensure that within each fold the number of positive and negative examples are equal. We measure our classifier’s performance across each split and finally take the mean of the performance achieved.

## Model Results

| Model          | Stratified F-1 Score | Non-Stratified F-1 Score |
| -------------- | -------------------- | ------------------------ |
| SVM            | 98.28%               | 98.31%                   |
| Random Forest  | 97.68%               | 97.61%                   |
| AdaBoost       | 98.01%               | 98.17%                   |
| Neural Network | 98.16%               | 98.27%                   |

## Conclusions
