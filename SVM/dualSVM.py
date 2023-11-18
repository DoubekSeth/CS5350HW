import numpy as np
import pandas as pd
from scipy import optimize

attributes = ["VoW", "SoW", "CoW", "Entropy", "Label"]
bank_note_training = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/bank-note/train.csv", header=None, names=attributes)
bank_note_testing = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/bank-note/test.csv", header=None, names=attributes)

def add_bias_term_to_df(df):
    rows = df.shape[0]
    columns = df.shape[1]
    ones = np.ones(rows)
    #Need to check if actually need to insert
    if("Bias Term" not in df.columns):
        df.insert(columns-1, "Bias Term", ones, False)


def dualSVM(training, epochs, kernel, C):
    #Add bias term to dataframe
    add_bias_term_to_df(training)
    #Initialize weights to 0
    w = np.zeros(training.shape[1]-1)
    y_vec = 2*training["Label"]-1
    #For training examples, find optimal values
    kernelMatrix = np.fromfunction(np.vectorize(lambda i, j: kernel(training.iloc[int(i)], training.iloc[int(j)])), (training.shape[0], training.shape[0]))

    def objective_function(X):
        return 0.5*np.sum(np.dot(kernelMatrix * np.array((y_vec * X))[:, np.newaxis], y_vec*X))-np.sum(X)

    bounds = optimize.Bounds(lb=0, ub=C)

    def mutualSlack(X):
            return np.dot(X, y_vec)-0
        
    constraint_dict = {'type': 'eq', 'fun':mutualSlack}

    initial_guess = np.zeros(len(y_vec))

    result = optimize.minimize(objective_function, initial_guess, method="SLSQP", constraints=constraint_dict, bounds=bounds)["x"]
    print(result)

    return result

def noKernel(a, b):
    return np.dot(a, b)

def evaluate_dualSVM(testing, training, alphas):
    correct = 0
    add_bias_term_to_df(testing)
    for index, row in testing.iterrows():
        label = row.iloc[-1]
        data = row.iloc[:len(row)-1]
        prediction = sgn(alphas*training*np.fromfunction(lambda i, j: kernel(training)))
        if((2*label-1)*prediction > 0):
            correct += 1
    return correct/testing.shape[0]

def sgn(x):
    return 1 if x >= 0 else -1

print(dualSVM(bank_note_training, 1, noKernel, C=10))
# for i in range(0, 3):
#     if(i == 0):
#         C = 100/873
#     if(i == 1):
#         C = 500/873
#     if(i == 2):
#         C = 700/873
#     print(i)
#     trained_weights = primalSVM(bank_note_training, 100, C=C)
#     print(evaluate_perceptron(bank_note_training, trained_weights))
#     acc = evaluate_perceptron(bank_note_testing, trained_weights)
#     print(acc)
#     print(trained_weights)
