# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:21:30 2018

@author: xmsun
"""

import numpy as np

def make_tree(A):
    (rows_t,cols_P) = np.shape(A);
    Petri_P = [[set(),set(),0] for x in range(cols_P)];
    Petri_t = [[set(),set(),0] for x in range(rows_t)];
    for j in range(rows_t):
        for i in range(cols_P):
            if A[j,i] == 1:
                Petri_t[j][1].add(i);
                Petri_P[i][0].add(j);
            elif A[j,i] == -1:
                Petri_t[j][0].add(i);
                Petri_P[i][1].add(j);
            elif A[j,i] == 2:
                Petri_t[j][1].add(i);
                Petri_t[j][0].add(i);               
                Petri_P[i][1].add(j);
                Petri_P[i][0].add(j);
    return Petri_P, Petri_t, rows_t, cols_P

def petriNet_tree(x0, Petri_P, Petri_t, rows_t, cols_P, lamda):
    out1 = [];
    #out2 = [];
    x0_copy = x0.copy();
    (j0,T0)=(0,0);
    for i in x0_copy:
        for j in Petri_P[i][1]:
            if Petri_t[j][0] - x0_copy == set():
                if Petri_t[j][2] == 0:
                    T1 = 0;
                    for pp in Petri_t[j][0]:
                        if T1<Petri_P[pp][2]:
                            T1 = Petri_P[pp][2];
                    if Petri_t[j][2] == 0:
                        Petri_t[j][2] = T1 + np.random.exponential(lamda[j]);
                if Petri_t[j][2]< T0 or T0 ==0:
                    (j0,T0) = (j,Petri_t[j][2]);
                    
    while (j0,T0) != (0,0):
        out1.append(j0);
        #out2.append((j0,T0));
        for pp in Petri_t[j0][0]: #t的父亲p
            x0_copy.remove(pp);
            Petri_P[pp][2]=0;
        for pb in Petri_t[j0][1]: #t的孩子p
            Petri_P[pb][2]= Petri_t[j0][2];
            x0_copy.add(pb);
        for pp in Petri_t[j0][0]: #t的父亲p
            for tb in Petri_P[pp][1]:
                Petri_t[tb][2]=0;
                
        (j0,T0)=(0,0);
        for i in x0_copy:
            for j in Petri_P[i][1]:
                if Petri_t[j][0] - x0_copy == set():
                    if Petri_t[j][2] == 0:
                        T1 = 0;
                        for pp in Petri_t[j][0]:
                            if T1<Petri_P[pp][2]:
                                T1 = Petri_P[pp][2];
                        Petri_t[j][2] = T1 + np.random.exponential(lamda[j]);
                    if Petri_t[j][2]< T0 or T0 ==0:
                        (j0,T0) = (j,Petri_t[j][2]);
                        
    Petri_P[-1][2]=0;
                        
#    out3 = '';
#    for c in out1:
#        out3 = out3+chr(c+ord('a'));
    
    return out1;  
    

def petriNet_Verify_tree(x0, Petri_P, Petri_t, rows_t, cols_P, lamda, seq):
    seq_copy = list(seq);
    x0_copy = x0.copy();
    return_flag = 0;
    for i in x0_copy:
        Petri_P[i][2] = 1;
        for j in Petri_P[i][1]:
            if Petri_t[j][0] - x0_copy == set():
                Petri_t[j][2] = 1;       
    while True:
        try:
            j0 = seq_copy[0];
            if Petri_t[j0][2] != 1:   #孩子p
                return_flag = -1;
                break;
            for pp in Petri_t[j0][0]: #t的父亲p
                x0_copy.remove(pp);
                Petri_P[pp][2]=0;
            for pb in Petri_t[j0][1]: #t的孩子p
                Petri_P[pb][2]= 1;
                x0_copy.add(pb);
            for pp in Petri_t[j0][0]: #t的父亲p
                for tb in Petri_P[pp][1]:
                    Petri_t[tb][2]=0;

    
            for i in x0_copy:
                for j in Petri_P[i][1]:
                    if Petri_t[j][0] - x0_copy == set():
                        Petri_t[j][2] = 1; 
            seq_copy.pop(0);
            if seq_copy == []:
                break;
        except:
            return_flag = -1;
            break;
        
    for i in range(cols_P):
        Petri_P[i][2]=0;
    for j in range(rows_t):
        Petri_t[j][2]=0;
    return return_flag;
