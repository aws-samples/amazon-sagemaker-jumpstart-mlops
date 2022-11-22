"""Feature engineers the abalone dataset."""
import pandas as pd
import numpy as np

import io
import os
import sys
import time
import json

from time import strftime, gmtime
from sklearn import preprocessing


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    #Read Data
    churn = pd.read_csv(
        f"{base_dir}/input/churn.txt"
    )
    # Drop Column
    churn = churn.drop("Phone", axis=1)
    churn["Area Code"] = churn["Area Code"].astype(object)
    churn["target"] = churn["Churn?"].map({"True.": 1, "False.": 0})
    churn.drop(["Churn?"], axis=1, inplace=True)
    churn = churn[["target"] + churn.columns.tolist()[:-1]]
    cat_columns = [
        "State",
        "Account Length",
        "Area Code",
        "Phone",
        "Int'l Plan",
        "VMail Plan",
        "VMail Message",
        "Day Calls",
        "Eve Calls",
        "Night Calls",
        "Intl Calls",
        "CustServ Calls",
    ]

    cat_idx = []
    for idx, col_name in enumerate(churn.columns.tolist()):
        if col_name in cat_columns:
            cat_idx.append(idx)
            
    with open("cat_idx.json", "w") as outfile:
        json.dump({"cat_idx": cat_idx}, outfile)
    
    for idx, col_name in enumerate(churn.columns.tolist()):
        if col_name in cat_columns:
            le = preprocessing.LabelEncoder()
            churn[col_name] = le.fit_transform(churn[col_name])
        
    from sklearn.model_selection import train_test_split
    
    # Split in Train, Test and Validation Datasets
    train, val_n_test = train_test_split(
        churn, test_size=0.3, random_state=42, stratify=churn["target"]
    )
    
    validation, test = train_test_split(
        val_n_test, test_size=0.3, random_state=42, stratify=val_n_test["target"]
    )
    
    # Save the Dataframes as csv files
    train.to_csv(f"{base_dir}/train/data.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/data.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/data.csv", header=False, index=False)
