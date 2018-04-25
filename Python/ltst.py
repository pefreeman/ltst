### LTST -- Python Implementation ###
import numpy as np
import random
#simport csv

from numpy import newaxis
from scipy.sparse import csc_matrix
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

class ltst(object):
    def __init__(self,predTrain,respTrain,numTree=1000,sampSize=0,randomState=101):
        self.predTrain = predTrain
        self.respTrain = respTrain
        self.numTree   = numTree
        if sampSize <= 0:
            self.sampSize = np.round(np.power(np.shape(predTrain)[0],0.7)).astype(int)
        self.inbag = np.zeros((np.shape(self.predTrain)[0],self.numTree)).astype(int)
        for ii in range(self.numTree):
            s = random.sample(range(len(self.predTrain)),self.sampSize)
            self.inbag[s,ii] = 1
        self.randomState = randomState
    
    def predict(self,predTest):
        assert(np.unique(self.inbag.sum(axis=0)).shape[0] == 1 and 
               np.unique(self.inbag.sum(axis=0))[0] != 0)
        (rows,cols) = self.inbag.shape
        N = csc_matrix(self.inbag)
        avg = np.mean(N,axis=1)
        
        pred = self.CBcforest(predTest)
        predAggr = np.mean(pred,axis=1)[...,newaxis]
        predCent = pred - predAggr
        predSums = predCent.sum(axis=1)
        predSums = np.reshape(predSums,(1,len(predSums)))

        C = N.dot(predCent.T) - avg.dot(predSums)
        var = (np.multiply(C,C).sum(axis=0) / cols**2).T
        
        return np.column_stack((predAggr,var))
    
    def test(self,predictedProportion,globalProportion=0.5):
        testStat = np.array((predictedProportion[:,0]-globalProportion)/np.sqrt(predictedProportion[:,1])).flatten()
        pval = norm.cdf(testStat).flatten()
        idx = pval > 0.5
        pval[idx] = 1-pval[idx]
        testPVal = fdrcorrection(2*pval)[1]
        high = (testPVal < 0.05) & (testStat > 0)
        low  = (testPVal < 0.05) & (testStat < 0)
        result = np.full(len(pval),-1).astype(int)
        result[high] = 1
        result[low]  = 0
        return result
    
    #############################
    
    def CBcforest(self,predTest):
        htree = []
        for ii in range(self.numTree):
            idx = self.inbag[:,ii]>0
            treePredTrain = self.predTrain[idx,:]
            treeRespTrain = self.respTrain[idx]
            htree.append(self.honestTree(treePredTrain,treeRespTrain,predTest))
        return np.array(htree).T

    def honestTree(self,treePredTrain,treeRespTrain,predTest):
        N      = treePredTrain.shape[0]
        idx    = random.sample(range(N), N//2)
        pred1  = treePredTrain[idx,...]
        resp1  = treeRespTrain[idx]
        pred2  = np.delete(treePredTrain,idx,0)
        resp2  = np.delete(treeRespTrain,idx,0)
        tree   = DecisionTreeRegressor(min_samples_split=2,min_samples_leaf=1,
                                       min_impurity_decrease=0.0001,random_state=self.randomState)
        tree.fit(pred1,resp1)
        predTestNode = tree.apply(predTest)
        predTest = np.column_stack((predTest,predTestNode))
        
        predTrainSplitNode = tree.apply(pred2)
        predTrainSplitComp = np.column_stack((predTrainSplitNode,resp2))

        aggPredNode = np.unique(predTrainSplitNode)
        aggPredAvg = np.array([np.mean(
                predTrainSplitComp[predTrainSplitComp[..., 0] == node, 1])
                for node in aggPredNode])
        aggPred = np.column_stack((aggPredNode,aggPredAvg))

        if (np.unique(tree.apply(pred1)).size != aggPredNode.size) :
            classTree0 = np.setdiff1d(np.unique(tree.apply(pred1)),aggPredNode)
            append = np.column_stack((classTree0,np.array([0.5]*classTree0.size)))
            aggPred = np.vstack((aggPred,append))

        idx = np.array([np.where(aggPred[...,0] == node)[0][0] for node in predTestNode])
        predTest = np.column_stack((predTest,((aggPred[..., 1])[idx])))
        return predTest[...,-1]
    
#### Need to fix floating point error ###
#
## The readFile(path) function is adapted from the CMU 15-112 website
## Week 3, Strings, Basic File IO Notes
## https://www.cs.cmu.edu/~112/notes/notes-strings.html#basicFileIO
## Modifications include transforming syntax to read in .csv files and to
## convert this file to a 2D list of numeric values
#
#def readFile(path):
#	dataList = []
#	skipHeader = True
#
#	with open(path, 'rt') as csvfile:
#		dataFile = csv.reader(csvfile)
#
#		for row in dataFile:
#			if(skipHeader):
#				skipHeader = False
#				continue
#
#			temp = []
#			for data in row:
#				if("e" in data):
#					ind = data.find("e")
#					base = float(data[:ind])
#					exp = int(data[ind + 1:])
#					temp.append(base * 10**exp)
#				else:
#					temp.append(float(data))
#			dataList.append(temp)
#	return dataList