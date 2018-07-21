
# Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import random
import itertools
class dtree:
    """ A basic Decision Tree"""
	
    def __init__(self):
        """ Constructor """

    def read_data(self,filename):
        fid = open(filename,"r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(","))
        fid.close()
    
        
        #Finding the most frequently occurring value in for feature nr. 10 (where '?' occurs)
        occurs = {}
        for d in range(1,len(data)): 
            if data[d][10] not in occurs:
                occurs[data[d][10]] = 1
            else:
                occurs[data[d][10]]+=1               

#       Printing most freqeuntly occuring data
#         for i in occurs:
#             print (i, occurs[i])
            
        #List of features 
        self.featureNames = [i for i in range(len(data[0])-1)]
        self.classes = []
        
        # =============================================================================
        #Dealing with missing data by replacing with predicted value:
# =============================================================================
#         
#         #Missing data rows
#         missing = []
#         
#         #Ok data rows
#         data_filtered = []
#         
#         for d in range(len(data)):
#             if data[d][10] == '?':
#                 missing.append(data[d])
#             else:
#                 data_filtered.append(data[d])
#         
#         #Removing duplicates
#         data_filtered.sort()
#         missing.sort()
#         data_filtered = [data_filtered[i] for i in range(len(data_filtered)) if i == 0 or data_filtered[i] != data_filtered[i-1]]
#         missing = [missing[i] for i in range(len(missing)) if i == 0 or missing[i] != missing[i-1]]
#         
          #Important to shuffle data because it is in some order
#         missing = random.sample(missing, len(missing)) 
#         data_filtered = random.sample(data_filtered, len(data_filtered)) 
#         
#         
#         #Separating classes for traning
#         for d in range(len(data_filtered)):
#            self.classes.append(data_filtered[d][10])
#            data_filtered[d] = data_filtered[d][0:10]+data_filtered[d][11:]   
#         
#         classes_missing = []  
#         #Separating missing classes for prediction
#         for d in range(len(missing)):
#            classes_missing.append(missing[d][10])
#            missing[d] = missing[d][0:10]+missing[d][11:]   
# =============================================================================
       
        
# =============================================================================
#         # =============================================================================
#         #Dealing with missing data by replacing with most frequent:
#         for d in range(len(data)):
#             if data[d][10] == '?':
#                 data[d][10] = 'b'
#                 
#         #Removing duplicates      
#         data.sort()
#         #Using sort and itertools.groupby to remove duplicates
#         #data = list(data for data,_ in itertools.groupby(data))
#         data = [data[i] for i in range(len(data)) if i == 0 or data[i] != data[i-1]]
#         
#         #Shufling data
#         data = random.sample(data, len(data)) 
# 
#         #Splitting data into classes and data
#         for d in range(len(data)):
#            self.classes.append(data[d][0])
#            data[d] = data[d][1:]      
# 
# =============================================================================

# =============================================================================
# 
#           
          # =============================================================================
#         #Dealing with missing categorical data by deleting rows
#         data_filtered = []
#         for d in range(len(data)):
#             if data[d][10] != '?':
#                 #print(data[d][10])
#                 data_filtered.append(data[d])
#                 #self.classes.append(data[d][0])
#         
#         #Removing duplicates      
#         data_filtered.sort()
#         data_filtered = [data_filtered[i] for i in range(len(data_filtered)) if i == 0 or data_filtered[i] != data_filtered[i-1]]
#         
#         #Randomizing
#         data_filtered = random.sample(data_filtered, len(data_filtered))                
#                
#         #Splitting data into classes and data (after removing '?' rows)
#         for d in range(len(data_filtered)):
#             self.classes.append(data_filtered[d][0])
#             data_filtered[d] = data_filtered[d][1:] 
# 
# =============================================================================
            
       #Not dealing with '?' since we load newdata.data which is already handled by replacing with predicted values   
        data = [data[i] for i in range(len(data)) if i == 0 or data[i] != data[i-1]]
         
         #Shufling data
        data = random.sample(data, len(data)) 
 
         #Splitting data into classes and data
        for d in range(len(data)):
            self.classes.append(data[d][0])
            data[d] = data[d][1:]      
           
          


        #return data_filtered,self.classes,self.featureNames #For deleting '?'
        return data,self.classes,self.featureNames #For replacing '?' or doing nothing
        #return data_filtered, self.classes, missing, classes_missing, self.featureNames  #For predicting '?'


    def classify(self,tree,datapoint):

        if type(tree) == type("string"):
			# Have reached a leaf
            return tree
        else:
            a = list(tree.keys())[0]
            for i in range(len(self.featureNames)):
                if self.featureNames[i]==a:
                    break
			
            try:
                t = tree[a][datapoint[i]]
                return self.classify(t,datapoint)
            except:
                return None
        

    def classifyAll(self,tree,data):
        results = []
        for i in range(len(data)):
            results.append(self.classify(tree,data[i]))
        return results


    def make_tree(self,data,classes,featureNames,maxlevel=-1,level=0):
        """ The main function, which recursively constructs the tree"""


        #EARLY STOPPING
# =============================================================================
        edible=0
        threshold  = 1
        

        for c in classes:
            if c=="e":
                edible+=1
        ep_ratio = edible/len(classes)
        if(ep_ratio)>threshold :
            return "e"
        elif(ep_ratio)<(1-threshold ):
            return "p"
            


#EARLY STOPPING FOR PREDICING '?'            
# =============================================================================
#         cs = 0
#         es = 0
#         rs = 0
#         bs= 0
#         n = len(classes)
#         for c in classes:
#             if c=="c":
#                 cs+=1
#             elif c=="e":
#                 es+=1 
#             elif c=="r":
#                 rs+=1 
#             elif c=="b":
#                 bs+=1                     
# 
#         if(cs/n)>threshold:
#             return "c"
#         elif(es/n)>threshold:
#             return "s" 
#         elif(rs/n)>threshold:
#             return "r"
#         elif(bs/n)>threshold:
#             return "b" 
# =============================================================================
# =============================================================================        
        else:
            nData = len(data)
            nFeatures = len(data[0])
            
    
            try: 
                self.featureNames
            except:
                self.featureNames = featureNames
            
            # List the possible classes
            newClasses = []
            for aclass in classes:
                if newClasses.count(aclass)==0:
                    newClasses.append(aclass)
    
            # Compute the default class (and total entropy)
            frequency = np.zeros(len(newClasses))
    
            totalEntropy = 0
            #  totalGini = 0
            index = 0
            for aclass in newClasses:
                frequency[index] = classes.count(aclass)
                totalEntropy += self.calc_entropy(float(frequency[index])/nData)
                #  totalGini += (float(frequency[index])/nData)**2
        
                index += 1
    
            #  totalGini = 1 - totalGini
            default = classes[np.argmax(frequency)]
    
            if nData==0 or nFeatures == 0 or (maxlevel>=0 and level>maxlevel):
                # Have reached an empty branch
                return default
            elif classes.count(classes[0]) == nData:
                # Only 1 class remains
                return classes[0]
            else:
    
                # Choose which feature is best      
                gain = np.zeros(nFeatures)
                #ggain = np.zeros(nFeatures)
    
                for feature in range(nFeatures):
                    g = self.calc_info_gain(data,classes,feature)
                    gain[feature] = totalEntropy - g
                    #  ggain[feature] = totalGini - gg
                    
                bestFeature = np.argmax(gain)

                tree = {featureNames[bestFeature]:{}}
    
                # List the values that bestFeature can take
                values = []
                for datapoint in data:
                    # From github: https://github.com/tback/MLBook_source/blob/master/6%20Trees/dtree.py
                    if datapoint[bestFeature] not in values:
                    # From book website
                    #  if datapoint[feature] not in values:
                        values.append(datapoint[bestFeature]) 
    
                for value in values:
                    # Find the datapoints with each feature value
                    newData = []
                    newClasses = []
                    index = 0
                    for datapoint in data:
                        if datapoint[bestFeature]==value:
                            if bestFeature==0:
                                newdatapoint = datapoint[1:]
                                newNames = featureNames[1:]
                            elif bestFeature==nFeatures:
                                newdatapoint = datapoint[:-1]
                                newNames = featureNames[:-1]
                            else:
                                newdatapoint = datapoint[:bestFeature]
                                newdatapoint.extend(datapoint[bestFeature+1:])
                                newNames = featureNames[:bestFeature]
                                newNames.extend(featureNames[bestFeature+1:])
                            newData.append(newdatapoint)
                            newClasses.append(classes[index])
                        index += 1
    
                    # Now recurse to the next level 
                    subtree = self.make_tree(newData,newClasses,newNames,maxlevel,level+1)
                    
                    # And on returning, add the subtree on to the tree
                    tree[featureNames[bestFeature]][value] = subtree
                return tree

    def printTree(self,tree,name):
        if type(tree) == dict:
            print (name, list(tree.keys())[0])
            for item in list(tree.values())[0].keys():
                print (name, item)
                self.printTree(list(tree.values())[0][item], name + "\t")
        else:
            print (name, "\t->\t", tree)

    def calc_entropy(self,p):
        if p!=0:
            return -p * np.log2(p)
        else:
            return 0

    def calc_info_gain(self,data,classes,feature):
        gain = 0
        nData = len(data)
        # List the values that feature can take
        values = []
        #print(feature)
        for datapoint in data:            
            if datapoint[feature] not in values:
                    values.append(datapoint[feature])
    
        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0
        # Find where those values appear in data[feature] and the corresponding class
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature]==value:
                    featureCounts[valueIndex]+=1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1
    
            # Get the values in newClasses
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass)==0:
                    classValues.append(aclass)
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex]+=1
                classIndex += 1
    
            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex]) / sum(classCounts))
            gain += float(featureCounts[valueIndex])/nData * entropy[valueIndex]
            valueIndex += 1
        return gain