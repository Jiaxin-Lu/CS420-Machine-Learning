import numpy as np
import os
import sys
import pandas as pd

print("loading data\n")
df = pd.read_csv("train.csv", delimiter=",")
temp_train = {"ids": df[["ID"]].values, "premise1": df[["premise1"]].values,
              "premise2": df[["premise2"]].values, "premise3": df[["premise3"]].values,
              "premise4": df[["premise4"]].values, "hypothesis": df[["hypothesis"]].values,
              "label": df[["label"]].values}

df = pd.read_csv("test.csv", delimiter=",")
temp_test = {"ids": df[["ID"]].values, "premise1": df[["premise1"]].values,
             "premise2": df[["premise2"]].values, "premise3": df[["premise3"]].values,
             "premise4": df[["premise4"]].values, "hypothesis": df[["hypothesis"]].values}

# map_targets = {"unknown": 3, "neutral": 0, "entailment": 1, "contradiction": 2}
map_targets = {"unknown": 3, "neutral": 1, "entailment": 2, "contradiction": 0}

target_list = ["contradiction", "entailment", "neutral"]


def process_file(file, is_train=True):
    labels, texts = [], []
    for _ in file["ids"]:
        ids = _[0]
        text = file["premise1"][ids][0] + " " + file["premise2"][ids][0] + " " + \
               file["premise3"][ids][0] + " " + file["premise4"][ids][0] + " ||| " + \
               file["hypothesis"][ids][0]
        texts.append(text)
        if is_train:
            label = map_targets[temp_train["label"][ids][0]]
            labels.append(label)
        else:
            labels.append(map_targets["contradiction"])
    return labels, texts


def process_file2(file, is_train=True):
    labels, texts, hypothesises = [], [], []
    for _ in file["ids"]:
        ids = _[0]
        text = file["premise1"][ids][0] + " | " + file["premise2"][ids][0] + " | " + \
               file["premise3"][ids][0] + " | " + file["premise4"][ids][0]
        hypothesis = file["hypothesis"][ids][0]
        hypothesises.append(hypothesis)
        texts.append(text)
        if is_train:
            label = map_targets[temp_train["label"][ids][0]]
            labels.append(label)
        else:
            labels.append(map_targets["contradiction"])
    return labels, texts, hypothesises


"""for bert vector"""
train_labels, train_texts = process_file(temp_train, is_train=True)
train_df = pd.DataFrame({'label': train_labels, 'text': train_texts})

test_labels, test_texts = process_file(temp_test, is_train=False)
test_df = pd.DataFrame({'label': test_labels, 'text': test_texts})

"""for bert"""
# labels, texts, hypothesis = process_file2(temp_train, is_train=True)
# train_df = pd.DataFrame({'label': labels, 'text': texts, 'hypothesis': hypothesis})
#
# labels, texts, hypothesis = process_file2(temp_test, is_train=False)
# test_df = pd.DataFrame({'label': labels, 'text': texts, 'hypothesis': hypothesis})
# test_df.to_csv("testData2.csv")
# train_df.to_csv("trainData2.csv")

"""print info"""

print(train_df.head())
print(test_df.head())

train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
print(train_df.describe())

test_df['text_len'] = test_df['text'].apply(lambda x: len(x))
print(test_df.describe())
