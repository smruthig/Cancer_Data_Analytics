import json
import pandas as pd
from pathlib import Path

filenames = [Path(f) for f in Path(".").glob("*.json")]
df = pd.DataFrame(columns=["Model", "Standardize", "Normalize", "PCA", "F1-Stratified", "F1", "Runtime"])

for filename in filenames:
    data = json.load(open(filename))
    model = filename.stem
    model_components = model.split("_", 1)
    model_name, namespace = model_components
    components = namespace[10:-1]
    components = components.split(",")
    
    normalize = False
    standardize = False
    pca = False
    for c in components:
        variable, value = c.split("=")
        variable = variable.strip()
        value = value.strip()
        if variable == "normalize" and value != "None":
            normalize = value
        elif variable == "standardize" and value != "None":
            standardize = value
        elif variable == "pca" and value != "None":
            pca = value

    runtime = data["runtime_mins"]
    if 'best_model_scores' in data:
        f1, f1_stratified = data['best_model_scores']
    else:    
        f1 = data["cv_scores_kfold"]
        f1_stratified = data["cv_scores_stratified"]
    df.loc[len(df)] = [model_name, standardize, normalize, pca, f1_stratified, f1, runtime]

df.sort_values(by=["F1-Stratified"], ascending=False, inplace=True, ignore_index=True)
df.to_csv("model_results.csv", index=False)