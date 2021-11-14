import json
import time
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from yellowbrick.model_selection import CVScores
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

df = pd.read_excel("cancer_dataset.xls")
df.drop(columns=["id"], inplace=True)
df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == 'M' else 0)

X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values
X.shape, y.shape

scaler = StandardScaler()
normalizer = Normalizer()
pipe = Pipeline([("scaler", scaler), ("normalizer", normalizer)])
pipe.fit(X)
X_transformed = pipe.transform(X)

CV_SPLIT = 15


def getModelParameterSearch(model):
    if isinstance(model, LogisticRegression):
        param_grid = {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": [0.5, 0.4, 0.3, 0.2, 0.1, 1, 10],
            "tol": [1e-4, 1e-3, 1e-2, 1e-1, 1],
            "solver": ["liblinear", "lbfgs"],
            "dual": [True, False]
        }
    elif isinstance(model, SVC):
        param_grid = {
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "C": [0.5, 0.4, 0.3, 0.2, 0.1, 1, 10],
            "degree": [2, 3, 4, 5],
            "gamma": ["scaled", "auto"],
            "shrinking": [True, False],
            "tol": [1e-4, 1e-3, 1e-2, 1e-1, 1],
            "coef0": [0.5, 0.4, 0.3, 0.2, 0, 0.1, 0.5, 1, 10]

        }
    elif isinstance(model, RandomForestClassifier):
        param_grid = {
            "n_estimators": [10, 50, 100, 250, 500, 1000],
            # "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "min_samples_split": [1, 2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "max_features": [None, "auto", "log2"],
            "bootstrap": [True, False],
            # "min_impurity_decrease": [0.1, 0.2, 0.4, 0.6, 0.8, 1],
            "criterion": ["gini", "entropy"],
            # "oob_score": [True, False],
            # "class_weight": ["balanced", "balanced_subsample"]
        }
    elif isinstance(model, AdaBoostClassifier):
        param_grid = {
            "n_estimators": [10, 50, 100, 250, 500, 1000],
            "learning_rate": [0.1, 0.3, 0.5, 1, 5, 10],
            "algorithm": ["SAMME", "SAMME.R"],
            "base_estimator": [None, DecisionTreeClassifier(max_depth=None), RandomForestClassifier]
        }
    else:
        raise Exception("Invalid model type")
    return param_grid


def getCrossValidationScores(model):
    cv1 = KFold(n_splits=CV_SPLIT, shuffle=True, random_state=0)
    cv2 = StratifiedKFold(n_splits=CV_SPLIT, shuffle=True, random_state=0)
    visualizer = CVScores(model, cv=cv1, scoring='f1_weighted')
    visualizer.fit(X_transformed, y)
    score1 = visualizer.cv_scores_mean_
    visualizer = CVScores(model, cv=cv2, scoring='f1_weighted')
    visualizer.fit(X_transformed, y)
    score2 = visualizer.cv_scores_mean_
    return score1, score2


def getBestParameters(model):
    parameter_grid = getModelParameterSearch(model)
    search = GridSearchCV(model, parameter_grid, cv=CV_SPLIT,
                          scoring='f1_weighted', n_jobs=-1, error_score=0, verbose=2)
    search.fit(X_transformed, y)
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_model_scores = getCrossValidationScores(best_model)
    return best_model, best_params, best_model_scores


models = [
    LogisticRegression(random_state=0, max_iter=500, n_jobs=-1),
    SVC(random_state=0, max_iter=500),
    RandomForestClassifier(max_leaf_nodes=None,
                           max_depth=None, random_state=0, n_jobs=-1),
    AdaBoostClassifier(random_state=0)
]

for model in tqdm(models):
    model_name = model.__class__.__name__
    print(model_name)
    try:
        start_time = time.time()
        best_model, best_params, best_model_scores = getBestParameters(model)
        data = {
            "model": model_name,
            "best_params": best_params,
            "best_model_scores": best_model_scores,
            "runtime_mins": (time.time() - start_time) / 60
        }
        with open(f"{model_name}.json", 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(e)
