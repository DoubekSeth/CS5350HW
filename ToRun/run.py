# %% [markdown]
# All Imports, including data

# %%
import numpy as np
import pandas as pd

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
def createWeights0(hiddenLayer1, hiddenLayer2, data):
    add_bias_term_to_df(data)
    columns = data.shape[1]-1#Minus 1 from y labels
    weights = {}
    for i in range(1, hiddenLayer1):
        weights["0"+str(i)] = np.zeros(columns)
    for j in range(1, hiddenLayer2):
        weights["1"+str(j)] = np.zeros(hiddenLayer1)
    weights["21"] = np.zeros(hiddenLayer2)
    return weights

# %%
def createWeightsNormal(hiddenLayer1, hiddenLayer2, data):
    add_bias_term_to_df(data)
    columns = data.shape[1]-1#Minus 1 from y labels
    weights = {}
    for i in range(1, hiddenLayer1):
        weights["0"+str(i)] = np.random.randn(columns)
    for j in range(1, hiddenLayer2):
        weights["1"+str(j)] = np.random.randn(hiddenLayer1)
    weights["21"] = np.random.randn(hiddenLayer2)
    return weights

# %%
def sigmoid(x):
    return 1/(1+np.power(np.e, -x))

# %%
def backProp(pred, label, weights, firstLayerNodes, secondLayerNodes, hiddenLayer1, hiddenLayer2, data):
    columns = data.shape[0]
    wGrads = {}
    nodeGrads = {}
    #First pass
    nodeGrads["y"] = (pred - label) #dL/dy
    wGrads["21"] = nodeGrads["y"]*secondLayerNodes #dL/dw2
    #Second pass
    nodeGrads["2"] = nodeGrads["y"]*weights["21"] #dL/dZ2
    for i in range(1, hiddenLayer2):
        wGrads["1"+str(i)] = nodeGrads["2"][i]*secondLayerNodes[i]*(1-secondLayerNodes[i])*firstLayerNodes #dL/dw1
    #Third pass
    for j in range(1, hiddenLayer1):
        nodeGrads["1"+str(j)] = np.dot(np.multiply(nodeGrads["2"], np.multiply(secondLayerNodes, 1-secondLayerNodes))[1:], [weights["1"+str(a)][j] for a in range(1, hiddenLayer2)]) #dL/dZ1
    for k in range(1, hiddenLayer1):
        wGrads["0"+str(k)] = nodeGrads["1"+str(k)]*firstLayerNodes[k]*(1-firstLayerNodes[k])*np.array(data)
    return wGrads

# %%
def learningRate(gamma_0, alpha, t):
    return gamma_0/(1+gamma_0/alpha*t)

# %%
def evaluateNN(DF, weights, hiddenLayer1, hiddenLayer2):
    add_bias_term_to_df(DF)
    count = 0
    for index, row in DF.iterrows():
        label = row.iloc[-1]
        data = row.iloc[:len(row)-1]
        #Now, start the passes
        firstLayerNodes = np.ones(1)
        for i in range(1, hiddenLayer1):
            w = weights["0"+str(i)]
            firstLayerNodes = np.append(firstLayerNodes, sigmoid(np.dot(w, data)))
        #Second Pass
        secondLayerNodes = np.ones(1)
        for j in range(1, hiddenLayer2):
            w = weights["1"+str(j)]
            secondLayerNodes = np.append(secondLayerNodes, sigmoid(np.dot(w, firstLayerNodes)))
        #Final pass
        pred = np.dot(weights["21"], secondLayerNodes)
        if(pred < 0.5 and label == 0 or pred >= 0 and label == 1):
            count += 1
    return count/DF.shape[0]


# %%
def evaluateLossNN(DF, weights, hiddenLayer1, hiddenLayer2):
    add_bias_term_to_df(DF)
    losses = 0
    for index, row in DF.iterrows():
        label = row.iloc[-1]
        data = row.iloc[:len(row)-1]
        #Now, start the passes
        firstLayerNodes = np.ones(1)
        for i in range(1, hiddenLayer1):
            w = weights["0"+str(i)]
            firstLayerNodes = np.append(firstLayerNodes, sigmoid(np.dot(w, data)))
        #Second Pass
        secondLayerNodes = np.ones(1)
        for j in range(1, hiddenLayer2):
            w = weights["1"+str(j)]
            secondLayerNodes = np.append(secondLayerNodes, sigmoid(np.dot(w, firstLayerNodes)))
        #Final pass
        pred = np.dot(weights["21"], secondLayerNodes)
        l = 0.5*(pred-label)**2
        losses += l
    return losses

# %%
def forwardPass(DF, weights, T, gamma_0, alpha, hiddenLayer1, hiddenLayer2, updateWeights=True, learningRate=learningRate):
    add_bias_term_to_df(DF)
    for epoch in range(T):
        #print(epoch)
        shuffled_data = DF.sample(frac=1)
        for index, row in shuffled_data.iterrows():
            label = row.iloc[-1]
            data = row.iloc[:len(row)-1]
            #Now, start the passes
            firstLayerNodes = np.ones(1)
            for i in range(1, hiddenLayer1):
                w = weights["0"+str(i)]
                firstLayerNodes = np.append(firstLayerNodes, sigmoid(np.dot(w, data)))
            #Second Pass
            secondLayerNodes = np.ones(1)
            for j in range(1, hiddenLayer2):
                w = weights["1"+str(j)]
                secondLayerNodes = np.append(secondLayerNodes, sigmoid(np.dot(w, firstLayerNodes)))
            #Final pass
            pred = np.dot(weights["21"], secondLayerNodes)
            if(updateWeights):
                gradient = backProp(pred, label, weights, firstLayerNodes, secondLayerNodes, hiddenLayer1, hiddenLayer2, data)
                #update weights
                for key in weights:
                    weights[key] = weights[key] - learningRate(gamma_0, alpha, index)*gradient[key]
        #print(evaluateLossNN(DF, weights, hiddenLayer1, hiddenLayer2))
    return weights

# %%
data = bank_note_training
for i in range(0, 3):
    if(i == 0):
        hiddenLayer1 = 5
        hiddenLayer2 = 5
    if(i == 1):
        hiddenLayer1 = 10
        hiddenLayer2 = 10
    if(i == 2):
        hiddenLayer1 = 25
        hiddenLayer2 = 25
    print("Neural Network Width:", hiddenLayer1)

    weights = createWeightsNormal(hiddenLayer1, hiddenLayer2, data)
    learnedWeights = forwardPass(data, weights, 10, 0.1, 25, hiddenLayer1, hiddenLayer2)
    print("Training:", evaluateNN(bank_note_training, learnedWeights, hiddenLayer1, hiddenLayer2))
    print("Testing:", evaluateNN(bank_note_testing, learnedWeights, hiddenLayer1, hiddenLayer2))


