import pandas as pd
import numpy as np

pred_0 = np.loadtxt("./pred/predicts_51_2.txt", delimiter=',')
pred_1 = np.loadtxt("./pred/predicts_52.txt", delimiter=',')
pred_2 = np.loadtxt("./pred/predicts_53.txt", delimiter=',')
# pred_3 = np.loadtxt("./pred/predicts_7.txt", delimiter=',')

ans = []
# map_back = {3: "unknown", 0: "neutral", 1: "entailment", 2: "contradiction"}
map_back = {3: "unknown", 0: "contradiction", 1: "neutral", 2: "entailment"}

for i in range(1000):
    ls = np.zeros(3)
    ls[int(pred_0[i])] += 0.82
    ls[int(pred_1[i])] += 0.817889
    ls[int(pred_2[i])] += 0.817778
    # ls[int(pred_3[i])] += 0.874
    ans_ = np.argmax(ls)
    ans.append(map_back[ans_])

name = ['label']

out = pd.DataFrame(columns=name, data=ans)
out.to_csv("predict_merge.csv", encoding='utf-8')
