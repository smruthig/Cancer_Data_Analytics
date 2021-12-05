import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA

df = pd.read_excel("cancer_dataset.xls")
df.drop(columns=["id"], inplace=True)
df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == 'M' else 0)

X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values
X.shape, y.shape

parser = argparse.ArgumentParser()
parser.add_argument("--cv_split", '-c', help="Number of cross validation splits", type=int, default=15)
parser.add_argument("--normalize", '-n', help="Normalize data", type=bool)
parser.add_argument("--standardize", '-s', help="Standardize data", type=bool)
parser.add_argument("--pca", '-pca', help="Use PCA", type=int)
args = parser.parse_args()
print(args)

CV_SPLIT = args.cv_split
PREPROCESSING_COMPONENTS = list()
if args.standardize:
    scaler = StandardScaler()
    PREPROCESSING_COMPONENTS.append(("scaler", scaler))
if args.normalize:
    normalizer = Normalizer()
    PREPROCESSING_COMPONENTS.append(("normalizer", normalizer))
if args.pca is not None:
    pca = PCA(n_components=args.pca)
    PREPROCESSING_COMPONENTS.append(("pca", pca))

pipe = Pipeline(PREPROCESSING_COMPONENTS)
pipe.fit(X)
X_transformed = pipe.transform(X)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def trainNeuralNetwork(X_train, X_test, y_train, y_test):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.00015), loss='binary_crossentropy', metrics=['accuracy', f1_m])
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    predictions = model.predict(X_test)
    predictions = (predictions >= 0.5).astype(int)
    return f1_score(y_test, predictions)



def getCrossValidationScore():
    cv_stratified = StratifiedKFold(n_splits=CV_SPLIT, shuffle=True)
    cv_scores_stratified = list()
    for train_index, test_index in cv_stratified.split(X_transformed, y):
        X_train, X_test = X_transformed[train_index], X_transformed[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cv_scores_stratified.append(trainNeuralNetwork(X_train, X_test, y_train, y_test))

    cv_kfold = KFold(n_splits=CV_SPLIT, shuffle=True)
    cv_scores_kfold = list()
    for train_index, test_index in cv_kfold.split(X_transformed, y):
        X_train, X_test = X_transformed[train_index], X_transformed[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cv_scores_kfold.append(trainNeuralNetwork(X_train, X_test, y_train, y_test))
    
    return cv_scores_stratified, cv_scores_kfold

start_time = time.time()
cv_scores_stratified, cv_scores_kfold = getCrossValidationScore()
end_time = time.time()

data = {
    "cv_scores_stratified": np.mean(cv_scores_stratified),
    "cv_scores_kfold": np.mean(cv_scores_kfold),
    "runtime_mins": (end_time - start_time) / 60
}
with open(f"neuralnetwork_{args}.json", 'w') as f:
    json.dump(data, f, indent=4)