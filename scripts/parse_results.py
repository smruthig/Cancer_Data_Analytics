import json
import pandas as pd
from pathlib import Path

filenames = [Path(f) for f in Path(".").glob("../models/*.json")]
df = pd.DataFrame(columns=["Model", "F1-Stratified", "F1", "Runtime"])

for filename in filenames:
    data = json.load(open(filename))
    model = filename.stem
    runtime = data["runtime_mins"]
    if 'best_model_scores' in data:
        f1, f1_stratified = data['best_model_scores']
    else:
        f1 = data["cv_scores_kfold"]
        f1_stratified = data["cv_scores_stratified"]
    df.loc[len(df)] = [model, f1_stratified, f1, runtime]

df.sort_values(by=["F1-Stratified"], ascending=False,
               inplace=True, ignore_index=True)
print(df)
df.to_csv("models_results.csv")
