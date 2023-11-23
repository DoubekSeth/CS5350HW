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

def primalSVM(training, epochs, C):
    gamma_0 = 0.01
    N = training.shape[0]
    #Add bias term to dataframe
    add_bias_term_to_df(training)
    #Initialize weights to 0
    w = np.zeros(training.shape[1]-1)
    for epoch in range(0, epochs):
        #print(epoch)
        gamma = firstSchedule(gamma_0=gamma_0, alpha=0.05, t=epoch)
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

print("Starting primal SVM")
C = 500/873
trained_weights = primalSVM(bank_note_training, 20, C=C)
print("Training Accuracy:", evaluate_perceptron(bank_note_training, trained_weights))
acc = evaluate_perceptron(bank_note_testing, trained_weights)
print("Testing Accuracy:", acc)
print("Weights", trained_weights)

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


def dualSVM(training, kernel, C, gamma=1):
    #Add bias term to dataframe
    add_bias_term_to_df(training)
    #Initialize weights to 0
    y_vec = 2*training["Label"]-1
    #For training examples, find optimal values
    kernelMatrix = np.fromfunction(np.vectorize(lambda i, j: kernel(training.iloc[int(i), :training.shape[1]-1], training.iloc[int(j), :training.shape[1]-1])), (training.shape[0], training.shape[0]))

    def objective_function(X):
        return 0.5*np.sum(np.dot(kernelMatrix * np.array((y_vec * X))[:, np.newaxis], y_vec*X))-np.sum(X)

    bounds = optimize.Bounds(lb=0, ub=C)

    def mutualSlack(X):
        return np.dot(X, y_vec)-0
        
    constraint_dict = {'type': 'eq', 'fun':mutualSlack}

    initial_guess = np.zeros(len(y_vec))

    result = optimize.minimize(objective_function, initial_guess, method="SLSQP", constraints=constraint_dict, bounds=bounds)["x"]
    #Set all alphas close to 0 to 0
    epsilon = 0.00000001
    result[result < epsilon] = 0
    return result

def noKernel(a, b):
    return np.dot(a, b)

def gaussiankernel(a, b, c):
    return np.e**(-(np.linalg.norm(a-b)**2)/c)

def evaluate_dualSVM(testing, training, alphas, kernel, gamma=1):
    correct = 0
    add_bias_term_to_df(testing)
    add_bias_term_to_df(training)
    test_y = 2*testing["Label"]-1
    #Create kernel matrix
    kernelMatrix = np.fromfunction(np.vectorize(lambda i, j: kernel(training.iloc[int(i), :training.shape[1]-1], testing.iloc[int(j), :testing.shape[1]-1])), (training.shape[0], testing.shape[0]))
    alphayis = alphas*(2*training["Label"]-1)
    vecsgn = np.vectorize(sgn)
    predictions = vecsgn(np.dot(alphayis, kernelMatrix))
    errs = 0.5*np.sum(np.abs(predictions-test_y))
    return (testing.shape[0]-errs)/testing.shape[0]

def sgn(x):
    return 1 if x >= 0 else -1

def recover_weights(alphas, training):
    y_vec = 2*training["Label"]-1
    x_vec = training[["VoW", "SoW", "CoW", "Entropy", "Bias Term"]]
    w = np.sum(x_vec * np.array((y_vec * alphas))[:, np.newaxis], axis=0)
    return w


C = 700/873
gamma = 100
print("Starting Dual SVM, this might take 3-5 mins to run")
alphas = dualSVM(bank_note_training, noKernel, C=C, gamma=gamma)
print("Testing Error", evaluate_dualSVM(testing=bank_note_testing, training=bank_note_training, alphas=alphas, kernel=noKernel, gamma=gamma))
print("Training Error", evaluate_dualSVM(testing=bank_note_training, training=bank_note_training, alphas=alphas, kernel=noKernel, gamma=gamma))
print("Number of support vectors", np.count_nonzero(alphas))
