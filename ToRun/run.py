# %% [markdown]
# <a href="https://colab.research.google.com/github/DoubekSeth/CS5350HW/blob/main/DecisionTree/CS5350HW1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Import libraries and data sets

# %% [markdown]
# ## Libraries

# %%
from collections import Counter
import queue
import math
import statistics
import copy
import numpy as np
import pandas as pd

# %% [markdown]
# ## Datasets

# %% [markdown]
# ### Tennis

# %%
s_tennis = [["sunny", "hot", "high", "weak", "-"],
     ["sunny", "hot", "high", "strong", "-"],
     ["overcast", "hot", "high", "weak", "+"],
     ["rainy", "mild", "high", "weak", "+"],
     ["rainy", "cool", "normal", "weak", "+"],
     ["rainy", "cool", "normal", "strong", "-"],
     ["overcast", "cool", "normal", "strong", "+"],
     ["sunny", "mild", "high", "weak", "-"],
     ["sunny", "cool", "normal", "weak", "+"],
     ["rainy", "mild", "normal", "weak", "+"],
     ["sunny", "mild", "normal", "strong", "+"],
     ["overcast", "mild", "high", "strong", "+"],
     ["overcast", "hot", "normal", "weak", "+"],
     ["rainy", "mild", "high", "strong", "-"]]
full_attributes_tennis = {"outlook":["sunny", "overcast", "rainy"], "temperature":["hot", "mild", "cool"], "humidity":["high", "normal", "low"], "wind":["strong", "weak"]}
remaining_attributes_tennis = ["outlook", "temperature", "humidity", "wind"]

# %% [markdown]
# ### Boolean Table Example (1st problem)
# 

# %%
s_booltab = [
    ["0", "0", "1", "0", "0"],
    ["0", "1", "0", "0", "0"],
    ["0", "0", "1", "1", "1"],
    ["1", "0", "0", "1", "1"],
    ["0", "1", "1", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "1", "0", "1", "0"]
]
full_attributes_booltab = {"x1":["0", "1"], "x2":["0", "1"], "x3":["0", "1"], "x4":["0", "1"]}
remaining_attributes_booltab = ["x1", "x2", "x3", "x4"]

# %% [markdown]
# ### Car

# %%
s_cartraining = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/car-4/train.csv", header=None).to_numpy()
s_cartesting = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/car-4/test.csv", header=None).to_numpy()

full_attributes_car = {
    "buying":["vhigh", "high", "med", "low"],
    "maint":["vhigh", "high", "med", "low"],
    "doors":["2", "3", "4", "5more"],
    "persons":["2", "4", "more"],
    "lug_boot":["small", "med", "big"],
    "safety":["low", "med", "high"]}
remaining_attributes_car = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# %% [markdown]
# ### Bank

# %%
s_banktraining = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/bank-4/train.csv", header=None).to_numpy()
s_banktesting = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/bank-4/test.csv", header=None).to_numpy()

full_attributes_bank = {
    "age":"numeric",
    "job":["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
    "marital":["married","divorced","single"],
    "education":["unknown","secondary","primary","tertiary"],
    "default":["yes", "no"],
    "balance":"numeric",
    "housing":["yes", "no"],
    "loan":["yes", "no"],
    "contact":["unknown","telephone","cellular"],
    "day":"numeric",
    "month":["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration":"numeric",
    "campaign":"numeric",
    "pdays":"numeric",
    "previous":"numeric",
    "poutcome":["unknown","other","failure","success"]
}
remaining_attributes_bank = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]

# %% [markdown]
# Filling in unknowns in dataset

# %%
#Find all modes for each data point
attributes_mode_bank = {}
for i in range(len(remaining_attributes_bank)):
  mode = max(set([k for k in s_banktraining[:, i] if k != "unknown"]), key=[k for k in s_banktraining[:, i] if k != "unknown"].count)
  attributes_mode_bank[remaining_attributes_bank[i]] = mode

# Set the data back
s_banktraining_filled = copy.deepcopy(s_banktraining)
for datapoint in range(len(s_banktraining_filled)):
  for eleInd in range(len(s_banktraining_filled[datapoint])):
    if(s_banktraining_filled[datapoint, eleInd]=="unknown"):
      s_banktraining_filled[datapoint, eleInd]=attributes_mode_bank[remaining_attributes_bank[eleInd]]

# Set the data for testing back
s_banktesting_filled = copy.deepcopy(s_banktesting)
for datapoint in range(len(s_banktesting_filled)):
  for eleInd in range(len(s_banktesting_filled[datapoint])):
    if(s_banktesting_filled[datapoint, eleInd]=="unknown"):
      s_banktesting_filled[datapoint, eleInd]=attributes_mode_bank[remaining_attributes_bank[eleInd]]

full_attributes_bank_filled = {
    "age":"numeric",
    "job":["admin.","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
    "marital":["married","divorced","single"],
    "education":["secondary","primary","tertiary"],
    "default":["yes", "no"],
    "balance":"numeric",
    "housing":["yes", "no"],
    "loan":["yes", "no"],
    "contact":["telephone","cellular"],
    "day":"numeric",
    "month":["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration":"numeric",
    "campaign":"numeric",
    "pdays":"numeric",
    "previous":"numeric",
    "poutcome":["other","failure","success"]
}

# %% [markdown]
# # ID3 Algorithm

# %% [markdown]
# ## Main Algorithm

# %%
class Node:
  def __init__(self, s, atrs, parent=None, parentVal=None, label=None):
    self.s = s
    self.atrs = atrs
    self.parent = parent
    self.label = label
    self.children = []
    self.parentVal = parentVal

  def addChild(self, node):
    self.children.append(node)

# %%
def ID3(s, remaining_attributes, full_attributes, purity_func, parentNode, weights=None, level=0, stopping=math.inf, printing=False, random_forest_flag=False, random_forest_select=4):
  if weights is None:
    weights=np.ones(len(s))
  #If doing a random forest implementation
  if(random_forest_flag):
    random_forest_attr = np.random.choice(remaining_attributes, random_forest_select)
  #First, check if all training examples s have same label. First stopping condition
  if(labelSame(s)[0]):
    if(printing):
      print("Same Label", labelSame(s)[1])
    parentNode.label = labelSame(s)[1]
    return
  #Second, check if exhausted attributes and if so return leaf node with most common label. Second stopping condition
  if(len(remaining_attributes) == 0):
    if(printing):
      print("Exhausted Attributes", mostCommonLabel(s, 1, weights)[0][0])
    parentNode.label = mostCommonLabel(s, 1, weights)[0][0]
    return
  #Third, check if have hit maximum stopping depth
  #print("hit ", level, " out of ", stopping)
  if(level >= stopping):
    parentNode.label = mostCommonLabel(s, 1, weights)[0][0]
    return
  level += 1
  #If didn't stop, then we can move onto the main part of the algorithm
  #Create root if we don't have one
  if(parentNode==None):
    parentNode = Node(s=s, atrs=remaining_attributes, parent=None)
  #Find attribute best splits S
  if(random_forest_flag):
    bestAtr = findBestSplit(s, weights, random_forest_attr, full_attributes, purity_func, printing)
  else:
    bestAtr = findBestSplit(s, weights, remaining_attributes, full_attributes, purity_func, printing)
  parentNode.label = bestAtr
  if(printing):
    print(bestAtr)
  less_remaining_attributes = remaining_attributes[:]
  #print(less_remaining_attributes, bestAtr)
  less_remaining_attributes.remove(bestAtr)
  #Go thru all values for best attribute
  loop_attrs = full_attributes[bestAtr]
  numericFilter=False
  if(loop_attrs=="numeric"):
    numericFilter=True
    loop_attrs = ["lower than median", "higher than median"]
  for val in loop_attrs:
    s_v, weights_v = filterSOnAtrVal(s, weights, bestAtr, val, full_attributes, numericFilter)
    #Check if s_v empty
    if(len(s_v)==0):
      if(printing):
        print("S_v empty", val, mostCommonLabel(s, 1, weights)[0][0])
      newNode = Node(s=s_v, atrs=less_remaining_attributes, parent=parentNode, parentVal=val, label=mostCommonLabel(s, 1, weights)[0][0])
      parentNode.addChild(newNode)
    else:
      newNode = Node(s=s_v, atrs=less_remaining_attributes, parent=parentNode, parentVal=val)
      if(printing):
        print("New node", val)
      parentNode.addChild(newNode)
      ID3(s_v, less_remaining_attributes, full_attributes, purity_func, newNode, weights=weights_v, level=level, stopping=stopping, printing=printing)
  return parentNode

# %% [markdown]
# ## Helper Functions

# %% [markdown]
# Does two functions, first returns true if all labels are the same for s. Second returns the label if all labels are same or else returns none

# %%
def labelSame(s):
  label = s[0][len(s[0])-1]
  for i in s:
    if(i[len(i)-1] != label):
      return False, None
  return True, label

# %% [markdown]
# Another helper function, returns the n most common label in s

# %%
def mostCommonLabel(s, n, weights):
  labels = {}
  for i in range(len(s)):
    if(s[i][len(s[i])-1] in labels):
      labels[s[i][len(s[i])-1]] += weights[i]
    else:
      labels[s[i][len(s[i])-1]] = weights[i]
  #Find most common label out of all the labels
  #print("Labels:", labels)
  mostCommon = sorted(labels.items(), key=lambda x:x[1], reverse=True)
  #print("Most Common:", mostCommon)
  return mostCommon

# %% [markdown]
# This helper finds the attribute that results in the best split

# %%
def findBestSplit(s, weights, atrs, full_attributes, purity_func, printing):
  max_info_gain = 0
  max_info_gain_atr = atrs[0]
  for atr in atrs:
    info_gained = find_info_gained(s, weights, atr, full_attributes, purity_func, printing)
    if(printing):
      print(atr, info_gained)
    if(info_gained > max_info_gain):
      max_info_gain = info_gained
      max_info_gain_atr = atr
  return max_info_gain_atr

# %% [markdown]
# This helper gives the information gained from a single attribute

# %%
def find_info_gained(s, weights, atr, full_attributes, purity_func, printing):
  if(full_attributes[atr]=="numeric"):
    return find_numeric_info_gained(s, weights, atr, full_attributes, purity_func, printing)
  else:
    return find_categorical_info_gained(s, weights, atr, full_attributes, purity_func, printing)

# %%
def find_numeric_info_gained(s, weights, atr, full_attributes, purity_func, printing):
  atr_ind = list(full_attributes.keys()).index(atr)
  #Create dictionary that contains attribute categories and what labels each attribute category has
  atr_type_count_dict = {"low":[], "high":[]}
  count = 0
  median = statistics.median(x[atr_ind] for x in s)
  for i in s:
    if i[atr_ind] < median:
      atr_type_count_dict["low"].append([i[len(i)-1], weights[count]])
    else:
      atr_type_count_dict["high"].append([i[len(i)-1], weights[count]])
    count += 1
  #This puts s into format of (label, weight)
  pur_s = []
  count = 0
  for i in s:
    pur_s.append([i[len(i)-1], weights[count]])
    count += 1
  starting_info = purity_func(pur_s)
  if(printing):
    print("Starting Info: ", starting_info)
  weighted_sum = 0
  for j in atr_type_count_dict.values():
    weighted_sum += sum(i[1] for i in j)/sum(weights)*purity_func(j)
  return starting_info-weighted_sum

# %%
def find_categorical_info_gained(s, weights, atr, full_attributes, purity_func, printing):
  atr_ind = list(full_attributes.keys()).index(atr)
  #Create dictionary that contains attribute categories and what labels each attribute category has
  atr_type_count_dict = {}
  count = 0
  for i in s:
    if i[atr_ind] in atr_type_count_dict:
      atr_type_count_dict[i[atr_ind]].append([i[len(i)-1], weights[count]])
    else:
      atr_type_count_dict[i[atr_ind]] = [[i[len(i)-1], weights[count]]]
    count += 1
  #Use proportions to run info gain function and find total information gain
  #This puts s into format of (label, weight)
  pur_s = []
  count = 0
  for i in s:
    pur_s.append([i[len(i)-1], weights[count]])
    count += 1
  starting_info = purity_func(pur_s)
  if(printing):
    print("Starting Info: ", starting_info)
  weighted_sum = 0
  for j in atr_type_count_dict.values():
    if(printing):
      print("J", j)
    weighted_sum += sum(i[1] for i in j)/sum(weights)*purity_func(j)
  return starting_info-weighted_sum

# %% [markdown]
# Filters S based on an attribute value to get S_v

# %%
def filterSOnAtrVal(s, weights, atr, val, full_attributes, numericFilter):
  filtered = []
  filteredWeights = []
  count = 0
  atr_ind = list(full_attributes.keys()).index(atr)
  for i in s:
    if(numericFilter):
      if(val == "lower than median"):
        if(i[atr_ind] < statistics.median(x[atr_ind] for x in s)):
          filtered.append(i)
          filteredWeights.append(weights[count])
      else:
        if(i[atr_ind] >= statistics.median(x[atr_ind] for x in s)):
          filtered.append(i)
          filteredWeights.append(weights[count])
    else:
      if(i[atr_ind] == val):
        filtered.append(i)
        filteredWeights.append(weights[count])
    count += 1
  return filtered, filteredWeights

# %% [markdown]
# ## Purity Functions

# %%
def entropy(labels):
  labeldf = pd.DataFrame(labels, columns=["Label", "Weight"])
  grouped = labeldf.groupby("Label").sum()
  #print(grouped)
  vec_counts = np.array(grouped).flatten()
  vec_props = vec_counts/np.sum(vec_counts)
  return -np.dot(vec_props, np.log2(vec_props))

# %%
def GiniInd(labels):
  labeldf = pd.DataFrame(labels, columns=["Label", "Weight"])
  grouped = labeldf.groupby("Label").sum()
  vec_counts = np.array(grouped).flatten()
  vec_props = vec_counts/np.sum(vec_counts)
  return 1-np.dot(vec_props, vec_props)

# %%
def majorityError(labels):
  labeldf = pd.DataFrame(labels, columns=["Label", "Weight"])
  grouped = labeldf.groupby("Label").sum()
  vec_counts = np.array(grouped).flatten()
  vec_props = vec_counts/np.sum(vec_counts)
  vec_props.sort()
  return np.sum(vec_props[:len(vec_props)-1])

# %% [markdown]
# ## Display Tree

# %%
def unravelTree(q1, q2, childQ, level):
  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
  while(not q1.empty()):
    node = q1.get()
    childThere = childQ.get()
    firstChild = True
    print("ParentVal: ", node.parentVal)
    print("Label: ", node.label)
    if(not childThere):
      print("----")
    for child in node.children:
      q2.put(child)
      if(firstChild):
        firstChild = False
      else:
        childQ.put(True)
    if(len(node.children)!= 0):
      childQ.put(False)
  level += 1
  q1 = q2
  q2 = queue.Queue()
  if(q1.empty()):
    return
  unravelTree(q1, q2, childQ, level)

# %% [markdown]
# This one creates a tree and returns the root node

# %%
def createTree(s, rem, full_attr, pur, weights, stop, printing=False):
    root = ID3(s, rem, full_attr, pur, None, weights=weights, stopping=stop, printing=printing)
    return root

# %%
def createRFTree(s, rem, full_attr, pur, weights, stop, printing=False, random_forest_selector=4):
    root = ID3(s, rem, full_attr, pur, None, weights=weights, stopping=stop, printing=printing, random_forest_flag=True, random_forest_select=random_forest_selector)
    return root

# %% [markdown]
# This prints a tree

# %%
def displayTree(root):
    q = queue.Queue()
    q2 = queue.Queue()
    q.put(root)
    childQ = queue.Queue()
    childQ.put(True)

    unravelTree(q, q2, childQ, 0)

# %% [markdown]
# # Run ID3

# %% [markdown]
# ## Set Data

# %% [markdown]
# index = 3
# s = [s_booltab, s_tennis, s_cartraining, s_banktraining, s_banktraining_filled][index]
# s_test = ["","", s_cartesting, s_banktesting, s_banktraining_filled][index]
# rem = [remaining_attributes_booltab, remaining_attributes_tennis, remaining_attributes_car, remaining_attributes_bank, remaining_attributes_bank][index]
# full_attr = [full_attributes_booltab, full_attributes_tennis, full_attributes_car, full_attributes_bank, full_attributes_bank_filled][index]
# pur = [entropy, GiniInd, majorityError][0]
# stop = 1

# %% [markdown]
# ## Display Tree

# %% [markdown]
# weights = np.ones(len(s))
# print(weights)
# root = ID3(s, rem, full_attr, pur, None, weights=weights, stopping=stop, printing=True)
# 
# q = queue.Queue()
# q2 = queue.Queue()
# q.put(root)
# childQ = queue.Queue()
# childQ.put(True)
# 
# 
# unravelTree(q, q2, childQ, 0)

# %% [markdown]
# ## Evaluate Model

# %% [markdown]
# Finds accuracy of model on test data

# %%
def findAcc(root, test, full_attr):
  matched = 0
  for i in test:
    matched += travTree(root, i, full_attr)
  return matched/len(test)

# %% [markdown]
# Traverses the tree recursively, returns 1 if test label matches model and 0 otherwise

# %%
def travTree(root, datapoint, full_attr):
  #Stopping Condition, root no children
  if(len(root.children) == 0):
    if(root.label == datapoint[len(datapoint)-1]):
      #print("matched")
      return 1
    else:
      #print("no match")
      return 0
  #Otherwise, traverse down
  curr_attr = root.label
  do_numeric = False
  if(full_attr[curr_attr] == "numeric"):
    do_numeric = True
  #print("current attribute: ", curr_attr)
  atr_ind = list(full_attr.keys()).index(curr_attr)
  for child in root.children:
    if(do_numeric):
      median = statistics.median(x[atr_ind] for x in root.s)
      #print("numeric: ", median, " ", datapoint[atr_ind])
      if((datapoint[atr_ind] < median and child.parentVal=="lower than median") or (datapoint[atr_ind] >= median and child.parentVal=="higher than median")):
        return travTree(child, datapoint, full_attr)
    else:
      if(child.parentVal == datapoint[atr_ind]):
        #print("Going down on val: ", child.parentVal)
        return travTree(child, datapoint, full_attr)

# %% [markdown]
# root = ID3(s, rem, full_attr, pur, None, stopping=stop)
# 
# findAcc(root, s_test, full_attr)

# %% [markdown]
# # Run Boolean Table

# %% [markdown]
# ## Set Vars

# %% [markdown]
# index = 0
# s = [s_booltab, s_tennis, s_cartraining, s_banktraining, s_banktraining_filled][index]
# s_test = ["","", s_cartesting, s_banktesting, s_banktraining_filled][index]
# rem = [remaining_attributes_booltab, remaining_attributes_tennis, remaining_attributes_car, remaining_attributes_bank, remaining_attributes_bank][index]
# full_attr = [full_attributes_booltab, full_attributes_tennis, full_attributes_car, full_attributes_bank, full_attributes_bank_filled][index]
# pur = [entropy, GiniInd, majorityError][0]
# stop = math.inf

# %% [markdown]
# ## Run ID3 & Print

# %% [markdown]
# root = ID3(s, rem, full_attr, pur, None, stopping=stop, printing=True)
# 
# q = queue.Queue()
# q2 = queue.Queue()
# q.put(root)
# childQ = queue.Queue()
# childQ.put(True)
# 
# 
# unravelTree(q, q2, childQ, 0)

# %% [markdown]
# # Run Tennis Example

# %% [markdown]
# ## Set Vars

# %% [markdown]
# index = 1
# s = [s_booltab, s_tennis, s_cartraining, s_banktraining, s_banktraining_filled][index]
# s_test = ["","", s_cartesting, s_banktesting, s_banktraining_filled][index]
# rem = [remaining_attributes_booltab, remaining_attributes_tennis, remaining_attributes_car, remaining_attributes_bank, remaining_attributes_bank][index]
# full_attr = [full_attributes_booltab, full_attributes_tennis, full_attributes_car, full_attributes_bank, full_attributes_bank_filled][index]
# pur = [entropy, GiniInd, majorityError][0]
# stop = math.inf

# %% [markdown]
# ## Run ID3

# %% [markdown]
# ### Entropy

# %% [markdown]
# root = ID3(s, rem, full_attr, entropy, None, stopping=stop, printing=True)
# 
# q = queue.Queue()
# q2 = queue.Queue()
# q.put(root)
# childQ = queue.Queue()
# childQ.put(True)
# 
# 
# unravelTree(q, q2, childQ, 0)

# %% [markdown]
# ### Gini Index

# %% [markdown]
# root = ID3(s, rem, full_attr, GiniInd, None, stopping=stop, printing=True)
# 
# q = queue.Queue()
# q2 = queue.Queue()
# q.put(root)
# childQ = queue.Queue()
# childQ.put(True)
# 
# 
# unravelTree(q, q2, childQ, 0)

# %% [markdown]
# ### Majority Error

# %% [markdown]
# root = ID3(s, rem, full_attr, majorityError, None, stopping=stop, printing=True)
# 
# q = queue.Queue()
# q2 = queue.Queue()
# q.put(root)
# childQ = queue.Queue()
# childQ.put(True)
# 
# 
# unravelTree(q, q2, childQ, 0)

# %% [markdown]
# #Using modified Tennis Example

# %%
s_tennis_modified = [["sunny", "hot", "high", "weak", "-"],
     ["sunny", "hot", "high", "strong", "-"],
     ["overcast", "hot", "high", "weak", "+"],
     ["rainy", "mild", "high", "weak", "+"],
     ["rainy", "cool", "normal", "weak", "+"],
     ["rainy", "cool", "normal", "strong", "-"],
     ["overcast", "cool", "normal", "strong", "+"],
     ["sunny", "mild", "high", "weak", "-"],
     ["sunny", "cool", "normal", "weak", "+"],
     ["rainy", "mild", "normal", "weak", "+"],
     ["sunny", "mild", "normal", "strong", "+"],
     ["overcast", "mild", "high", "strong", "+"],
     ["overcast", "hot", "normal", "weak", "+"],
     ["rainy", "mild", "high", "strong", "-"],
     #Modifications buddy
     ["overcast", "mild", "normal", "weak", "+"]]

# %% [markdown]
# root = ID3(s_tennis_modified, rem, full_attr, entropy, None, stopping=stop, printing=True)
# 
# q = queue.Queue()
# q2 = queue.Queue()
# q.put(root)
# childQ = queue.Queue()
# childQ.put(True)
# 
# 
# unravelTree(q, q2, childQ, 0)

# %% [markdown]
# # Used to help with fractional counts

# %% [markdown]
# def entropyWeighted(positive, negative):
#   whole = positive + negative
#   return (-(positive/whole)*math.log2(positive/whole) if positive/whole > 0 else 0)-((negative/whole)*math.log2(negative/whole) if negative/whole > 0 else 0)

# %% [markdown]
# entropyWeighted(3+5/14, 2)

# %% [markdown]
# poss = [3+5/14, 2+5/14]
# negs = [1, 1]
# S = 15
# summ = 0
# for i in range(len(poss)):
#   S_v = poss[i] + negs[i]
#   print("Pos: ", poss[i], " Neg: ", negs[i], " entropy weighted: ", (S_v)/S*entropyWeighted(poss[i], negs[i]))
#   summ += (S_v)/S*entropyWeighted(poss[i], negs[i])
# summ

# %% [markdown]
# # Car Example problem 2

# %% [markdown]
# ## Set Data

# %% [markdown]
# can set stop to halt tree growth after a certain number of steps

# %% [markdown]
# index = 2
# s = [s_booltab, s_tennis, s_cartraining, s_banktraining, s_banktraining_filled][index]
# s_test = ["","", s_cartesting, s_banktesting, s_banktraining_filled][index]
# rem = [remaining_attributes_booltab, remaining_attributes_tennis, remaining_attributes_car, remaining_attributes_bank, remaining_attributes_bank][index]
# full_attr = [full_attributes_booltab, full_attributes_tennis, full_attributes_car, full_attributes_bank, full_attributes_bank_filled][index]
# pur = [entropy, GiniInd, majorityError][0]
# stop = 4

# %% [markdown]
# ## Running with entropy

# %% [markdown]
# root = ID3(s, rem, full_attr, entropy, None, stopping=stop)
# 
# findAcc(root, s_test, full_attr)

# %% [markdown]
# ## Running with gini index

# %% [markdown]
# root = ID3(s, rem, full_attr, GiniInd, None, stopping=stop)
# 
# findAcc(root, s_test, full_attr)

# %% [markdown]
# ## Running with majority error

# %% [markdown]
# root = ID3(s, rem, full_attr, majorityError, None, stopping=stop)
# 
# findAcc(root, s_test, full_attr)

# %% [markdown]
# ## Running through all levels & purity functions

# %% [markdown]
# for purity in range(0, 3):
#   print("~~~~~~~~~~~~~~~~~~~~~~~~~")
#   print("Using:", ["entropy", "GiniInd", "majorityError"][purity])
#   for i in range(1, 7):
#     root = ID3(s, rem, full_attr, [entropy, GiniInd, majorityError][purity], None, stopping=i)
#     accTest = 1-findAcc(root, s_test, full_attr)
#     accTrain = 1-findAcc(root, s, full_attr)
#     print("Stopping at", i, "obtains test error", accTest, "and training error", accTrain)

# %% [markdown]
# # Bank Example Problem 3

# %% [markdown]
# ## Unknowns as a value

# %% [markdown]
# index = 3
# s = [s_booltab, s_tennis, s_cartraining, s_banktraining, s_banktraining_filled][index]
# s_test = ["","", s_cartesting, s_banktesting, s_banktraining_filled][index]
# rem = [remaining_attributes_booltab, remaining_attributes_tennis, remaining_attributes_car, remaining_attributes_bank, remaining_attributes_bank][index]
# full_attr = [full_attributes_booltab, full_attributes_tennis, full_attributes_car, full_attributes_bank, full_attributes_bank_filled][index]
# pur = [entropy, GiniInd, majorityError][0]
# stop = 4

# %% [markdown]
# for purity in range(0, 3):
#   print("~~~~~~~~~~~~~~~~~~~~~~~~~")
#   print("Using:", ["entropy", "GiniInd", "majorityError"][purity])
#   for i in range(1, 17):
#     root = ID3(s, rem, full_attr, [entropy, GiniInd, majorityError][purity], None, stopping=i)
#     accTest = 1-findAcc(root, s_test, full_attr)
#     accTrain = 1-findAcc(root, s, full_attr)
#     print("Stopping at", i, "obtains test error", accTest, "and training error", accTrain)
#     

# %% [markdown]
# ## Car with unknowns filled

# %% [markdown]
# index = 4
# s = [s_booltab, s_tennis, s_cartraining, s_banktraining, s_banktraining_filled][index]
# s_test = ["","", s_cartesting, s_banktesting, s_banktesting_filled][index]
# rem = [remaining_attributes_booltab, remaining_attributes_tennis, remaining_attributes_car, remaining_attributes_bank, remaining_attributes_bank][index]
# full_attr = [full_attributes_booltab, full_attributes_tennis, full_attributes_car, full_attributes_bank, full_attributes_bank_filled][index]
# pur = [entropy, GiniInd, majorityError][0]
# stop = 4

# %% [markdown]
# for purity in range(0, 3):
#   print("~~~~~~~~~~~~~~~~~~~~~~~~~")
#   print("Using:", ["entropy", "GiniInd", "majorityError"][purity])
#   for i in range(1, 17):
#     root = ID3(s, rem, full_attr, [entropy, GiniInd, majorityError][purity], None, stopping=i)
#     accTest = 1-findAcc(root, s_test, full_attr)
#     accTrain = 1-findAcc(root, s, full_attr)
#     print("Stopping at", i, "obtains test error", accTest, "and training error", accTrain)



# %% [markdown]
# Run this to get decision tree which we use as our classifier

# %%
#%run "../DecisionTree/CS5350HW1.ipynb"

# %% [markdown]
# # Adaboost Algorithm

# %% [markdown]
# ## Main Algorithm

# %%
#Define D_t
#for t=1,2,..,T
    #find h_t whose weighted classification error is better than chance
    #Compute a_t = 1/2ln((1-e_t)/e_t)
    #Update D_t+1(i) = D_t(i)/Z_t*e^{-a_ty_ih_t(x_i)}
#H_final = sgn(sum(a_th_t(x)))

def Adaboost(D_t, iterations, s_train, full_attributes, attributes, purity_function):
    classifiers = []
    alphas = []
    for i in range(iterations):
        #print("Iteration:", i)
        root = createTree(s_train, attributes, full_attributes, purity_function, D_t, 1)
        #print(root.label)
        classifiers.append(copy.deepcopy(root))
        error = findError(root, s_train, D_t, full_attributes)
        a_t = 0.5*np.log((1-error)/error)
        D_t_next = np.ones(len(s_train))
        for trainInd in range(len(s_train)):
            #travTree returns 1 if matched and 0 otherwise
            pred = travTree(root, s_train[trainInd], full_attributes)
            if(pred == 0):
                D_t_next[trainInd] = D_t[trainInd]*np.e**(a_t)
            else:
                D_t_next[trainInd] = D_t[trainInd]*np.e**(-a_t)
        #print(D_t_next)
        Z_t = np.sum(D_t_next)
        D_t_next = D_t_next/Z_t
        #print(D_t)
        #print(D_t_next)
        #print(error)
        #displayTree(root)
        D_t = D_t_next
        alphas.append(a_t)
    return classifiers, alphas, D_t
            
            

# %%
def findError(root, s, D_t, full_attributes):
  matched = 0
  for i in range(len(s)):
    #Note, (1-x) maps 0 -> 1 and 1 -> 0
    matched += D_t[i]*(1-travTree(root, s[i], full_attributes))
  return matched

# %% [markdown]
# ## Finding accuracy

# %%
def findModelAccuracy(s_test, classifiers, alphas, full_attributes):
    matched = 0
    for test in s_test:
        h_f = computeHFinal(classifiers, alphas, test, full_attributes)
        #print(h_f, test[len(test)-1])
        if(h_f > 0 and convertStringToBinary(test[len(test)-1]) > 0):
            matched += 1
        elif(h_f < 0 and convertStringToBinary(test[len(test)-1]) < 0):
             matched += 1
    return matched/len(s_test)


# %%
def computeHFinal(classifiers, alphas, x_i, full_attributes):
    final = 0
    for i in range(len(classifiers)):
        #print(alphas[i])
        #print(travTreeWithoutLabel(classifiers[i], x_i, full_attributes))
        final += alphas[i]*travTreeWithoutLabel(classifiers[i], x_i, full_attributes)
    return final

# %%
def convertStringToBinary(y_i):
    if(y_i == "yes"):
        return 1
    else:
        return -1

# %% [markdown]
# Pretty much travTree, but now will return 1 for a yes label and -1 for a no label

# %%
def travTreeWithoutLabel(root, datapoint, full_attr):
  #Stopping Condition, root no children
  if(len(root.children) == 0):
    if(root.label == "yes"):
      #print("matched")
      return 1
    else:
      #print("no match")
      return -1
  #Otherwise, traverse down
  curr_attr = root.label
  do_numeric = False
  if(full_attr[curr_attr] == "numeric"):
    do_numeric = True
  #print("current attribute: ", curr_attr)
  atr_ind = list(full_attr.keys()).index(curr_attr)
  for child in root.children:
    if(do_numeric):
      median = statistics.median(x[atr_ind] for x in root.s)
      #print("numeric: ", median, " ", datapoint[atr_ind])
      if((datapoint[atr_ind] < median and child.parentVal=="lower than median") or (datapoint[atr_ind] >= median and child.parentVal=="higher than median")):
        return travTreeWithoutLabel(child, datapoint, full_attr)
    else:
      if(child.parentVal == datapoint[atr_ind]):
        #print("Going down on val: ", child.parentVal)
        return travTreeWithoutLabel(child, datapoint, full_attr)

# %% [markdown]
# # Testing

# %% [markdown]
# iterations = 20
# s_train = s_banktraining
# s_test = s_banktesting
# full_attributes = full_attributes_bank
# attributes = remaining_attributes_bank
# purity_function = GiniInd
# 
# D_1 = np.ones(len(s_train))/len(s_train)
# 
# classifiers, alphas = Adaboost(D_t=D_1, iterations=iterations, s_train=s_train, full_attributes=full_attributes, attributes=attributes, purity_function=purity_function)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(alphas)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# %% [markdown]
# T=500
# classifierArr = []
# alphaArr = []
# D_t = np.ones(len(s_train))/len(s_train)
# s_train = s_banktraining
# s_test = s_banktesting
# full_attributes = full_attributes_bank
# attributes = remaining_attributes_bank
# purity_function = GiniInd
# 
# for i in range(1, T):
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(i)
#     classifiers, alphas, weights = Adaboost(D_t=D_t, iterations=1, s_train=s_train, full_attributes=full_attributes, attributes=attributes, purity_function=purity_function)
#     classifierArr.append(classifiers[0])
#     alphaArr.append(alphas[0])
#     D_t = weights
#     print("train:", findModelAccuracy(s_test=s_train, classifiers=classifierArr, alphas=alphaArr, full_attributes=full_attributes))
#     print("test:",findModelAccuracy(s_test=s_test, classifiers=classifierArr, alphas=alphaArr, full_attributes=full_attributes))



# %%
#%run "../EnsembleLearning/Adaboost.ipynb"

# %%
import random
import math
import numpy as np

# %%
#Repeat 1,..,T
#Draw m samples from training with replacement
#Learn decision tree C_t using ID3/Cart
#Vote/Average T predictions

def bagging(s_train, T, attributes, full_attributes, purity_function, voting=True):
    trees = []
    for i in range(T):
        print(i)
        sampled = random.choices(s_train, k=len(s_train))
        weights = np.ones(len(s_train))
        root = createTree(sampled, attributes, full_attributes, purity_function, weights, stop=math.inf)
        trees.append(root)
        print("Accuracy:", findModelAccuracyBag(trees, s_train, full_attributes))
        print("Accuracy:", findModelAccuracyBag(trees, s_test, full_attributes))
    return trees

# %%
def findModelAccuracyBag(trees, s, full_attributes):
    matched = 0
    for datapoint in s:
        preds = []
        for tree in trees:
            pred = travTreeWithoutLabel(tree, datapoint, full_attributes)
            preds.append(pred)
        #print(preds)
        votedPred = convertBinaryToString(max(set(preds), key=preds.count))
        #print(votedPred)
        if(votedPred == datapoint[len(datapoint)-1]):
            matched += 1
    return matched/len(s)    

# %%
def convertBinaryToString(bin):
    if(bin == 1):
        return "yes"
    else:
        return "no"

# %% [markdown]
# s_train = s_banktraining
# s_test = s_banktesting
# full_attributes = full_attributes_bank
# attributes = remaining_attributes_bank
# purity_function = GiniInd
# T=20
# 
# trees = bagging(s_train=s_train, T=T, attributes=attributes, full_attributes=full_attributes, purity_function=purity_function)

# %%
#%run "../EnsembleLearning/Bagging.ipynb"

# %%
import random

# %%
def RFbagging(s_train, T, attributes, full_attributes, purity_function, s_test, voting=True, random_forest_selector=4):
    trees = []
    for i in range(T):
        print(i)
        sampled = random.choices(s_train, k=len(s_train))
        weights = np.ones(len(s_train))
        root = createRFTree(sampled, attributes, full_attributes, purity_function, weights, stop=math.inf, random_forest_selector=random_forest_selector)
        trees.append(root)
        print("Accuracy Train:", findModelAccuracyBag(trees, s_train, full_attributes))
        print("Accuracy Test:", findModelAccuracyBag(trees, s_test, full_attributes))
    return trees

# %% [markdown]
# s_train = s_banktraining
# s_test = s_banktesting
# full_attributes = full_attributes_bank
# attributes = remaining_attributes_bank
# purity_function = GiniInd
# T=19
# 
# trees = RFbagging(s_train=s_train, T=T, attributes=attributes, full_attributes=full_attributes, purity_function=purity_function, s_test=s_test, random_forest_selector=2)

# %% [markdown]
# # Setup

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

# %% [markdown]
# ## Fetch Data

# %%
concrete_header = ["Cement", "Slag", "Fly Ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "SLUMP Flow"]
train_concrete = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/concrete/train.csv", names=concrete_header)
test_concrete = pd.read_csv("https://raw.githubusercontent.com/DoubekSeth/ToyDatasets/main/concrete/test.csv", names=concrete_header)

# %% [markdown]
# # Algorithm

# %% [markdown]
# Method that adds ones to a dataframe, makes algorithm simplier to implement

# %%
def addOnesForBiasOnDataset(df):
    rows = df.shape[0]
    columns = df.shape[1]
    ones = np.ones(rows)
    #Need to check if actually need to insert
    if("Bias Term" not in df.columns):
        df.insert(columns-1, "Bias Term", ones, False)

# %%
def BatchGD(train_df, gradient_of_cost_func, r, weights, convergence):
    converged = False
    t=0
    costs=[]
    while(not converged):
        grad = gradient_of_cost_func(train_df, weights)
        #print(grad)
        new_weights = weights - r*grad
        if(abs(np.sum(weights-new_weights)) < convergence):
            converged=True
        weights = new_weights
        #Print cost
        cost = cost_MSE_df(weights, train_df)
        t+=1
        print("Cost at step", t, ":", cost)
        costs.append(cost)
    return weights, costs, t

# %% [markdown]
# Returns the gradient of the training data with the weights for a MSE cost function

# %%
def gradient_Batch_MSE(train_df, weights):
    grad = np.zeros(train_df.shape[1]-1) #Subtract one for the label
    X = train_df.drop("SLUMP Flow", axis=1)
    Y = train_df["SLUMP Flow"]
    for i in range(len(grad)):
        X_i = train_df.iloc[:, i]
        #print("WTX", np.dot(X, weights))
        #print("Y-WTX", Y-np.dot(X, weights))
        #print("X_i", X_i)
        #print("-(Y-WTX)X_i", -np.dot((Y-np.dot(X, weights)), X_i))
        grad[i] = -np.dot((Y-np.dot(X, weights)), X_i)

    return grad

# %%
def StochasticGD(train_df, r, weights, convergence):
    converged = False
    t=0
    costs=[]
    while(not converged):
        for index, example in train_df.iterrows():
            new_weights = copy.copy(weights)
            for j in range(len(weights)):
                #print(train_df.iloc[index, train_df.shape[1]-1])
                #print(np.dot(example.drop("SLUMP Flow"), weights))
                #print(train_df.iloc[index, j])
                new_weights[j] = weights[j] + r*(train_df.iloc[index, train_df.shape[1]-1]-np.dot(example.drop("SLUMP Flow"), weights)*train_df.iloc[index, j])            
            if(abs(np.sum(weights-new_weights)) < convergence):
                converged=True
            weights=new_weights
            t+=1
            cost = cost_MSE_df(weights, train_df)
            print("Cost at step", t, ":", cost)
            costs.append(cost)
    return weights, costs, t

# %%
def cost_MSE_df(weights, data):
    X = data.drop("SLUMP Flow", axis=1)
    Y = data["SLUMP Flow"]
    return 0.5*(np.sum(np.square(Y-np.dot(X, weights))))

# %%
def cost_MSE(weights, datum):
    return 0.5*(datum["SLUMP Flow"]-np.dot(weights, datum.drop("SLUMP Flow")))

# %% [markdown]
# # Accuracy Evaluations

# %% [markdown]
# weights = np.zeros(8)
# addOnesForBiasOnDataset(train_concrete)
# #print(gradient_Batch_MSE(train_concrete, weights))
# weights, costs, steps = BatchGD(train_concrete, gradient_Batch_MSE, 0.005, weights, .000001)
# plt.plot(np.linspace(1, steps, steps), costs, marker='o', linestyle='-')
# plt.xlabel("Steps")
# plt.ylabel("Cost")
# plt.grid(True)
# plt.show
# print(weights)

# %% [markdown]
# addOnesForBiasOnDataset(test_concrete)
# cost_MSE_df(weights, test_concrete)

# %% [markdown]
# weights = np.zeros(8)
# addOnesForBiasOnDataset(train_concrete)
# #print(gradient_Batch_MSE(train_concrete, weights))
# weights, costs, steps = StochasticGD(train_concrete, 0.02, weights, .00002)
# plt.plot(np.linspace(1, steps, steps), costs, marker='o', linestyle='-')
# plt.xlabel("Steps")
# plt.ylabel("Cost")
# plt.grid(True)
# plt.show
# print(weights)
# addOnesForBiasOnDataset(test_concrete)
# print(cost_MSE_df(weights, test_concrete))

# %% [markdown]
# weights, residuals, rank, s = np.linalg.lstsq(train_concrete.drop("SLUMP Flow", axis=1), train_concrete["SLUMP Flow"])
# print(weights)
# cost_MSE_df(weights, test_concrete)

# %% [markdown]
# paper_df = pd.DataFrame([[1, -1, 2, 1], [1, 1, 3, 4], [-1, 1, 0, -1], [1, 2, -4, -2], [3, -1, -1, 0]])
# paper_df.columns = ["x_1", "x_2", "x_3", "SLUMP Flow"]
# addOnesForBiasOnDataset(paper_df)
# print(paper_df)
# 
# StochasticGD(paper_df, r=0.1, weights=np.zeros(4), convergence=0.001)





# %%
#%run "../EnsembleLearning/RandomForest.ipynb"
#%run "../LinearRegression/LMS.ipynb"

# %%
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Running Batch Gradient Descent for LMS")
weights = np.zeros(8)
addOnesForBiasOnDataset(train_concrete)
#print(gradient_Batch_MSE(train_concrete, weights))
print(BatchGD(train_concrete, gradient_Batch_MSE, 0.001, weights, .01))

# %%
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Running Stochastic Gradient Descent for LMS")
weights = np.zeros(8)
addOnesForBiasOnDataset(train_concrete)
#print(gradient_Batch_MSE(train_concrete, weights))
print(StochasticGD(train_concrete, 0.02, weights, .02))

# %%
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Running Adaboost")

T=3
classifierArr = []
alphaArr = []
s_train = s_banktraining
D_t = np.ones(len(s_train))/len(s_train)
s_test = s_banktesting
full_attributes = full_attributes_bank
attributes = remaining_attributes_bank
purity_function = GiniInd

for i in range(1, T):
    print(i)
    classifiers, alphas, weights = Adaboost(D_t=D_t, iterations=1, s_train=s_train, full_attributes=full_attributes, attributes=attributes, purity_function=purity_function)
    classifierArr.append(classifiers[0])
    alphaArr.append(alphas[0])
    D_t = weights
    print("train:", findModelAccuracy(s_test=s_train, classifiers=classifierArr, alphas=alphaArr, full_attributes=full_attributes))
    print("test:",findModelAccuracy(s_test=s_test, classifiers=classifierArr, alphas=alphaArr, full_attributes=full_attributes))

# %%
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Starting bagging")
trees = bagging(s_train=s_train, T=2, attributes=attributes, full_attributes=full_attributes, purity_function=purity_function)

# %%
print("~~~~~~~~~~~~~~~~~~~")
print("Starting Random Forest")
trees = RFbagging(s_train=s_train, T=2, attributes=attributes, full_attributes=full_attributes, purity_function=purity_function, s_test=s_test, random_forest_selector=2)


