import pandas as pd
import numpy as np

print("loading pred_bert")

df = pd.read_csv("pred5/result_bert52.csv", delimiter=",")
temp_result = {"ids": df[["ID"]].values, "0": df[["0"]].values, "1": df[["1"]].values, "2": df[["2"]].values}

# map_back = {3: "unknown", 0: "neutral", 1: "entailment", 2: "contradiction"}
map_back = {3: "unknown", 0: "contradiction", 1: "neutral", 2: "entailment"}


def process_file(file):
    ans = []
    for _ in file["ids"]:
        ids = _[0]
        prob = [file["0"][ids][0], file["1"][ids][0], file["2"][ids][0]]
        p = np.argmax(prob)
        ans.append(map_back[p])
    return ans


predicts = process_file(temp_result)
name = ['label']
out = pd.DataFrame(columns=name, data=predicts)
out.to_csv("pred5/predict_bert_52.csv", encoding='utf-8')

