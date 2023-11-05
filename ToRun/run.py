# %%
import pandas as pd
import numpy as np

# %%
attributes = ["VoW", "SoW", "CoW", "Entropy", "Label"]
bank_note_training = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/bank-note/train.csv", header=None, names=attributes)
bank_note_testing = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/bank-note/test.csv", header=None, names=attributes)

# %%
def add_bias_term_to_df(df):
    rows = df.shape[0]
    columns = df.shape[1]
    ones = np.ones(rows)
    #Need to check if actually need to insert
    if("Bias Term" not in df.columns):
        df.insert(columns-1, "Bias Term", ones, False)

# %%
def sgn(x):
    return np.where(x >= 0, 1, -1)

# %%
def train_perceptron(training, r, t):
    add_bias_term_to_df(training)
    weights = np.zeros(training.shape[1]-1) #Minus one from labels
    for epoch in range(t):
        print(epoch)
        for index, row in training.iterrows():
            label = row.iloc[-1]
            data = row.iloc[:len(row)-1]
            #print("Label:", label)
            #print("Data:", data)
            prediction = np.dot(weights, data)
            if((2*label-1)*prediction <= 0):
                weights = weights + r*((2*label-1)*data)
    return weights

# %%
def evaluate_perceptron(testing, weights):
    correct = 0
    add_bias_term_to_df(testing)
    for index, row in testing.iterrows():
        label = row.iloc[-1]
        data = row.iloc[:len(row)-1]
        prediction = np.dot(weights, data)
        if((2*label-1)*prediction > 0):
            correct += 1
    return correct/testing.shape[0]

# %%
def train_voted_perceptron(training, r, t):
    add_bias_term_to_df(training)
    weights = [np.zeros(training.shape[1]-1)] #Minus one from labels
    C_m = [0]
    m = 0
    for epoch in range(t):
        print(epoch)
        for index, row in training.iterrows():
            label = row.iloc[-1]
            data = row.iloc[:len(row)-1]
            prediction = np.dot(weights[m], data)
            if((2*label-1)*prediction <= 0):
                weights.append(weights[m] + r*((2*label-1)*data))
                m += 1
                C_m.append(1)
            else:
                C_m[m] = C_m[m]+1
    return weights, C_m

# %%
def evaluate_voted_perceptron(testing, weights, counts):
    correct = 0
    add_bias_term_to_df(testing)
    for index, row in testing.iterrows():
        label = row.iloc[-1]
        data = row.iloc[:len(row)-1]
        prediction = np.dot(counts, sgn(np.matmul(weights, data)))
        if((2*label-1)*prediction > 0):
            correct += 1
    return correct/testing.shape[0]

# %%
def train_average_perceptron(training, r, t):
    add_bias_term_to_df(training)
    weights = np.zeros(training.shape[1]-1) #Minus one from labels
    av = np.zeros(training.shape[1]-1)
    for epoch in range(t):
        print(epoch)
        for index, row in training.iterrows():
            label = row.iloc[-1]
            data = row.iloc[:len(row)-1]
            #print("Label:", label)
            #print("Data:", data)
            prediction = np.dot(weights, data)
            if((2*label-1)*prediction <= 0):
                weights = weights + r*((2*label-1)*data)
            av = av + weights
    return av

# %%
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Normal Perceptron")
weights = train_perceptron(bank_note_training, 0.1, 10)
print("Weights:", weights)
print("Test Accuracy:", evaluate_perceptron(bank_note_testing, weights))

# %%
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Voted Perceptron")
weights, counts = train_voted_perceptron(bank_note_training, 0.1, 10)
print("Weights:", weights)
print("Counts:", counts)
print("Test Accuracy:", evaluate_voted_perceptron(bank_note_testing, weights, counts))

# %%
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Average Perceptron")
weights = train_average_perceptron(bank_note_training, 0.1, 10)
print("Weights:", weights)
print("Test Accuracy:", evaluate_perceptron(bank_note_testing, weights))


