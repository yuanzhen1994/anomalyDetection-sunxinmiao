# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:15:01 2018

@author: xmsun
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score

from PattRela_Yuanzhen import petriNet_tree_system
from PattRela_Yuanzhen import trainingDataSampleSet_NestedLoops_YZ_tree
from PattRela_Yuanzhen import testDataSampleSet_NestedLoops_YZ_tree
from PattRela_Yuanzhen import relationTableAlgorithm_loops_YZ_tree


# Plot AUC vs #. simulations
aucVectors_loopyz = []  #loopyz: pattern relation with loops and modified by yuanzhen
f1Vectors_loopyz = []

numSample = range(5,500,50); programe_mode = 1;


cycle_mode = programe_mode
x0, A, lamda = petriNet_tree_system()

for i in numSample:
    print(i)

    if programe_mode == 1:
        trainData = trainingDataSampleSet_NestedLoops_YZ_tree( i, x0, A, lamda) #2：带嵌套环和环中环的数据  LoopinLoops and NestedLoops
        testData, y_test = testDataSampleSet_NestedLoops_YZ_tree(i, x0, A, lamda)
    elif programe_mode == 2 or programe_mode == 3:
        TempDir = r'../stidetraces_work';
        trainData=list(np.load(TempDir+'/NormalData_r2.npy'))
        if programe_mode == 2:
            testData=list(np.load(TempDir+'/AbnormalData_r2.npy'))
            y_test = [1 for _ in range(len(testData))];
        elif programe_mode == 3:
            testData = trainData[-2000:]
            del(trainData[-2000:])
            y_test = [0 for _ in range(len(testData))];
            
            
    
    if cycle_mode != 0:
        if programe_mode == 2 or programe_mode == 3:
            cycle_mode = 0
        y_predict_loopyz = relationTableAlgorithm_loops_YZ_tree(trainData, testData, y_test)

    
    fpr_loopyz, tpr_loopyz, thresholds_loopyz = roc_curve(y_test, y_predict_loopyz)
    aucValue_loopyz = auc(fpr_loopyz, tpr_loopyz)
    aucVectors_loopyz.append(aucValue_loopyz)
    f1_loopyz = f1_score(y_test, y_predict_loopyz) 
    f1Vectors_loopyz.append(f1_loopyz)
    
plt.figure(1)
lw = 2
plt.plot(numSample, aucVectors_loopyz, lw=lw, label='pattern table algorithm (YZ)')
plt.xlabel('# of tranining samples')
plt.ylabel('AUC values')
plt.legend()
plt.grid(True)
plt.show()


# Figure for F1-score
plt.figure(2)
lw = 2
plt.plot(numSample, f1Vectors_loopyz, lw=lw, label='pattern table algorithm (YZ)')
plt.xlabel('# of tranining samples')
plt.ylabel('F1-score')
plt.legend()
plt.grid(True)
plt.show()

if programe_mode == 2:
    print(r'数据集中共有'+str(len(testData))+r'个测试序列，其中'+str(list(y_predict_loopyz).count(1))+r'个序列被判别为异常！')