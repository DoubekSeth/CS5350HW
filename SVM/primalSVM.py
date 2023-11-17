import numpy as np
import pandas as pd

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

def primalSVM(training, epochs):
    gamma_0 = 0.001
    C = 700/873
    N = training.shape[0]
    #Add bias term to dataframe
    add_bias_term_to_df(training)
    #Initialize weights to 0
    w = np.zeros(training.shape[1]-1)
    for epoch in range(0, epochs):
        print(epoch)
        gamma = secondSchedule(gamma_0=gamma_0, t=epoch)
        #First, shuffle training
        shuffled_training = training.sample(frac=1)
        #For training examples
        for index, row in shuffled_training.iterrows():
            x = row.iloc[:len(row)-1]
            y = 2*row.iloc[-1]-1
            w_bias_0 = np.copy(w)
            w_bias_0[-1] = 0
            #print("X", x)
            #print("Y", y)
            #print("W", w)
            #print("Pred", y*np.dot(w, x))
            if(y*np.dot(w, x)<=1):
                w = w - gamma*(w_bias_0) + gamma*C*N*y*x
            else:
                w = w-gamma*w_bias_0
        #print("Training acc:", evaluate_perceptron(training, w))
    return w

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

def firstSchedule(gamma_0, alpha, t):
    return gamma_0/(1+gamma_0/alpha*t)

def secondSchedule(gamma_0, t):
    return gamma_0/(1+t)
trained_weights = primalSVM(bank_note_training, 100)
print(evaluate_perceptron(bank_note_training, trained_weights))
acc = evaluate_perceptron(bank_note_testing, trained_weights)
print(acc)
#print(trained_weights)
