# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:45 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import copy
from dataSetGeneration import make_tree
from dataSetGeneration import petriNet_tree
from dataSetGeneration import petriNet_Verify_tree

def petriNet_tree_system():
    A = np.matrix([                            
            [-1,1,0,0,0,1,0,0,0,1,0,0,0,0],#0-4
            [0,-1,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,-1,1,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,-1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,-1,1,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,-1,0,0,0,0,0,0,0,0,0],#5-9
            [0,0,0,0,0,-1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,-1,1,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,-1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,-1,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,-1,1,0,0,0],#10-14
            [0,0,0,0,0,0,0,0,0,0,-1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,-1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,-1,0],
 			  [0,0,0,0,0,0,0,0,0,0,1,0,-1,0],
            [0,0,0,0,-1,0,0,0,-1,0,0,0,-1,1]]);  #LoopinLoops and NestedLoops
    lamda = [1 for i in range(len(A))];
    x0 = set([0]);
    return x0, A, lamda;

def trainingDataSampleSet_NestedLoops_YZ_tree(sampleNumber, x0, A, lamda):
    # Generate trainSet we need for anomaly detection
    trainSet = []
    for i in range(sampleNumber):
        #firedList, seq = petriNet(x0, A, I, W_pt, lamda)
        a,b,c,d=make_tree(A);
        seq=petriNet_tree(x0,a,b,c,d,lamda);
        trainSet.append(seq)
    return trainSet

def testDataSampleSet_NestedLoops_YZ_tree(sampleNumber, x0, A, lamda):

    # Generate testSet we need for nomaly detection
    testSet_normal = []
    for i in range(sampleNumber):
        #firedList, seq = petriNet(x0, A, I, W_pt, lamda)
        a,b,c,d=make_tree(A);
        seq=petriNet_tree(x0,a,b,c,d,lamda);
        testSet_normal.append(seq)
    # Generate testSet we need for abnomaly detection    
    lamda_abnormal = lamda.copy()
    lamda_abnormal[1]=3
    testSet_abnormal = []
    for i in range(int(sampleNumber/4)):
        #firedList, seq_ab = petriNet(x0, A, I, W_pt, lamda_abnormal)
        seq_ab=petriNet_tree(x0,a,b,c,d,lamda);
        testSet_abnormal.append(seq_ab)
    # Case 2: order abnormal
    for i in range(2*int(sampleNumber/4)):
        #firedList, seq = petriNet(x0, A, I, W_pt, lamda)
        seq=petriNet_tree(x0,a,b,c,d,lamda);
        shuffle_seq = list(seq)
        np.random.shuffle(shuffle_seq)
        #shuffle 对数组重新随机排列 ； 
        #而np.random.permutation(10) 是直接生成一个随机排列的数组
        testSet_abnormal.append(shuffle_seq)
    # Case 3: task missing
    length_of_a_seq = len(testSet_normal[0]) 
    removed_index = np.random.randint(0,length_of_a_seq,int(sampleNumber/4))
    #numpy.random.randint(low, high=None, size=None, dtype='l')
    #返回一个随机整型数，范围从[low, high)含左不含右，如果没有写参数high的值，则返回[0,low)的值。
    for i in range(int(sampleNumber/4)):
        #firedList, seq = petriNet(x0, A, I, W_pt, lamda)
        seq=petriNet_tree(x0,a,b,c,d,lamda);
        pos = removed_index[i]
        testSet_abnormal.append(seq[:pos] + seq[(pos+1):])    
    # Case 4: task redundance
    added_index = np.random.randint(0,length_of_a_seq,sampleNumber - 3*int(sampleNumber/4))
    for i in range(sampleNumber - 3*int(sampleNumber/4)):
        #firedList, seq = petriNet(x0, A, I, W_pt, lamda)
        seq=petriNet_tree(x0,a,b,c,d,lamda);
        pos = added_index[i]
        seq.insert(added_index[i],added_index[i])
        testSet_abnormal.append(seq)

    testSet = testSet_normal + testSet_abnormal
    y_test = np.zeros(len(testSet))
    
    for i in range(len(testSet)):
        if testSet[i] == None:
            y_test[i] = 0;
        elif petriNet_Verify_tree(x0,a,b,c,d,lamda, testSet[i]) == -1:
            y_test[i] = 1;
        else:
            y_test[i] = 0;
    
    return testSet, y_test

def make_event_table_tree(sub_seqence_set, trainingData):
    sub_seqence_set.sort();
    seqlen = len(sub_seqence_set)
    max_item = max(sub_seqence_set)
    min_item = min(sub_seqence_set)
    itemcount = max(abs(max_item), abs(min_item), abs(max_item - min_item), abs(max_item + min_item))
    temp_list = [0]*itemcount
    event_table = [sub_seqence_set, temp_list, []]
    for i in range(seqlen):
        event_table[1][sub_seqence_set[i]] = i;
        event_table[2].append(set())

    for seq in trainingData:
        for i in range(len(seq) -1):
            a = temp_list[seq[i]]
            event_table[2][a].update([seq[i+1]])

#        for j in range(len(sub_seqence_set)):
#            pair = str(sub_seqence_set[i])+str(sub_seqence_set[j])
#            for seq in trainingData:
#                if pair in seq:
#                    if event_table.iloc[i,j] == '<--':
#                        event_table.iloc[i,j] = '||'
#                        event_table.iloc[j,i] = '||'
#                    elif event_table.iloc[i,j] == '.':
#                        event_table.iloc[i,j] = '-->'
#                        event_table.iloc[j,i] = '<--'       
    return event_table

def Add_StartAndEndFlag_tree(Dataset):
    if type(Dataset[0])==list:
        for seq in Dataset:
            seq.insert(0,-1)
            seq.append(-2)
    elif type(Dataset[0])==int:
        Dataset.insert(0,-1)
        Dataset.append(-2)   
    else:
        Dataset = [-1,-2];
    return Dataset

def find_loops_S_tree(trainData):
    
    #事件集合
    U = set();
    #重复事件集合
    P = set();
    for each in trainData:
        each_set = set(each);
        U.update(each_set);
        for unique in each_set:
            if not unique in P:
                if each.count(unique) > 1:
                    #P.update(unique);
                    P.update([unique]);
    
    #单一事件集合
    Q = U-P;
    newData = copy.deepcopy(trainData);
    for i in range(len(newData)):
        for unique in Q:
            while unique in newData[i]:
                newData[i].remove(unique);

            #newData[i] = newData[i].replace(unique, '');
            
    #必然事件集合
    R =  U.copy();
    for each in trainData:
        each_set = set(each);
        temp = R - each_set;
        if not temp == set():
            R = R - temp;
    
    #重复事件子数据集（k个）
    #Sx = [set() for i in range(len(P))];
    Sx = [[] for i in range(len(P))];
    P_list = list(P);
    for k in range(len(P)):
        for i in range(len(newData)):
            j1=-1; j2=-1;
            for j in range(len(newData[i])):
                if newData[i][j] == P_list[k]:
                    j1 = j2; j2 = j;
                    if j1>=0:
                        #Sx[k].update([newData[i][j1:j2+1]]);
                        Sx[k].append(newData[i][j1:j2+1]);
    
    #重复事件子序列（k个）
    S = [[] for i in range(len(Sx))];
    for k in range(len(Sx)):
        for xtx in Sx[k]:
            if len(xtx) < len(S[k]) or len(S[k]) == 0:
                S[k] = xtx;
    
    #重复事件子序列（m个, 0<=m<=k）,去掉重复的
    S_bank = copy.deepcopy(S);
    for j in range(len(S_bank)):
        for i in range(j):
            if set(S_bank[i]) == set(S_bank[j]):
                if S_bank[i] in S:
                    S.remove(S_bank[i]);
                    
    #重复事件子序列相关表
    event_table_S = [pd.DataFrame() for i in range(len(S))];
    #Add_StartAndEndFlag(newData);
    #Add_StartAndEndFlag(S);
    Add_StartAndEndFlag_tree(newData);
    Add_StartAndEndFlag_tree(S);

    for i in range(len(S)):
        loopData = [];
        S_i_set = set(S[i]);
        for j in range(len(newData)):
            loopData.append([]);
            for event in newData[j]:
                if event in S_i_set:
                    loopData[j].append(event);
        event_table_S[i] = make_event_table_tree(list(S_i_set), loopData);
        
    return list(U), list(R), S, event_table_S
     
def isAbnormal_unique_tree(relation_table, unique_events, seq):
    if not set(seq)-set(relation_table[0]) == set():
        return True
    if not set(unique_events) -set(seq) == set():
        return True
    for i in range(len(seq) - 1):
        if not seq[i+1] in relation_table[2][ relation_table[1][seq[i]] ]:
            return True
    return False 

def relationTableAlgorithm_loops_YZ_tree(trainData, testData, y_test):
    trainData_copy = copy.deepcopy(trainData);
    testData_copy = copy.deepcopy(testData);
    Add_StartAndEndFlag_tree(trainData_copy);
    Add_StartAndEndFlag_tree(testData_copy);
    
    #U所有事件集合
    #R必然事件集合
    #S重复事件子序列（k个）
    #event_table_S重复事件子序列相关表（k个）
    U, R, S, event_table_S = find_loops_S_tree(trainData_copy);
    U.sort()
    R.sort()
    event_table2 = make_event_table_tree(U, trainData_copy);
#    print(event_table2, 'U=',U, 'Q=',Q, 'S=',S, 'event_table_S=',event_table_S)
    l = len(testData_copy)
    y_predict = np.zeros(l)
    for i in range(l):
        if isAbnormal_unique_tree(event_table2, R, testData_copy[i]):
            y_predict[i] = 1;
#            if y_test[i]==0: 
#                print(testData_copy[i],'y_test = 0,y_predict = 1,A')
            continue;
        for j in range(len(S)):
            loops = testData_copy[i].copy();
            for unique in testData_copy[i]:
                if not unique in S[j]:
                    loops.remove(unique)
            if isAbnormal_unique_tree(event_table_S[j], [], loops):
                y_predict[i] = 1;
#                if y_test[i]==0:
#                    print(testData_copy[i],'y_test = 0,y_predict = 1,B')
                break;
#        if y_test[i]==1:
#            print(testData_copy[i],'y_test = 1,y_predict = 0')
    return y_predict        
















