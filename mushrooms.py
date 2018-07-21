#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 02:04:13 2017

@author: mathiasgronstad
"""



import dtree
tree = dtree.dtree()


poisonous,classes,features = tree.read_data('agaricus-lepiota.data')
#poisonous,classes,features = tree.read_data('newdata.data')
#Finding gain for each feature
for feature in features:
    print(tree.calc_info_gain(poisonous,classes,feature))




#For predicting classes:
# =============================================================================
# =============================================================================
# 
# #For predicting '?'
# data, classes, missing, classes_missing, features = tree.read_data('agaricus-lepiota.data')
# 
# split = int(3*len(data)/4)    
# train = data[:split][:]
# traint = classes[:split]
# test = data[split:][:]
# testt = classes[split:]
# 
# 
# 
# train_tree=tree.make_tree(train,traint,features)
# tree.printTree(train_tree,' ')
# 
# test_classify = tree.classifyAll(train_tree,test)
# 
# #Accuracy when classifying all data in test set
# count=0
# for i in range(len(test_classify)):
#     if(test_classify[i]==testt[i]):
#         count+=1
#         
# accuracy=count/len(test_classify)
# print(count,len(test_classify),accuracy)
# 
# #Classifying the missing data 
# classify_missing = tree.classifyAll(train_tree,missing)
# 
# #Some of the values become "None". I don't know why, so I replace them with the most common, 'b'
# for i in range(len(classify_missing)):
#     if (classify_missing[i] == None): classify_missing[i] = 'b'
# classes_missing = classify_missing
# 
# #Recombining data
# classes = classes + classify_missing
# data = data + missing
# for d in range(len(data)):
#    data[d] = data[d][0:10] + [classes[d]] + data[d][10:] 
# 
# #Writing to new text file with predicted '?'
# f = open('newdata.data', 'r+')
# file = open("newdata.data", "w")
# for x in range(len(data)):
# 	for y in range(len(data[x])):
# 		file.write(data[x][y])
# 		if x < len(data) and y < len(data[x])-1:
# 			file.write(",")
# 	file.write("\n")
# file.close()
# =============================================================================


        
split = int(2*len(poisonous)/3)    
train = poisonous[:split][:]
traint = classes[:split]
test = poisonous[split:][:]
testt = classes[split:]


#Training data on training set "train"
train_tree=tree.make_tree(train,traint,features)
#tree.printTree(train_tree,' ')


#Classifying all points in test set
test_classify = tree.classifyAll(train_tree,test)

#Accuracy when classifying all data in test set
count=0
for i in range(len(test_classify)):
    #print(test_classify[i], testt[i])
    if(test_classify[i]==testt[i]):
        count+=1
        
accuracy=count/len(test_classify)
print(count,len(test_classify),accuracy)





#Accuracy when classifying all data in test set
# =============================================================================
# #Classifying all points in validation set
#
# valid_classify = tree.classifyAll(train_tree,valid)# 
# count=0
# for i in range(len(valid_classify)):
#     if(valid_classify[i]==validt[i]):
#         count+=1
#         
# accuracy=count/len(valid_classify)
# print(accuracy)
# =============================================================================




#t=tree.make_tree(poisonous,classes,features)
#tree.printTree(t,' ')


#Avoid overfitting- don't grow a tree with only singeltons in its leaves. The number of leaves = the size of dataset.
#1. Stop growing the tree before it reaches perfection.
#or 2. Allow the tree to fully grow, and then post-prune some of the branches from it.


