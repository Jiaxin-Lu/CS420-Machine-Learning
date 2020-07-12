import pandas as pd
import numpy as np

print("loading data\n")
df = pd.read_csv("label.csv", delimiter=",")

temp_test_label = {"ids": df[["ID"]].values, "label": df[["label"]].values}


map_targets = {"unknown": 3, "neutral": 1, "entailment": 2, "contradiction": 0}

labels = []
for _ in temp_test_label["ids"]:
    ids = _[0]
    labels.append(map_targets[temp_test_label["label"][ids][0]])

labels = np.array(labels)
np.savetxt("y_test.txt", labels, delimiter=',')
