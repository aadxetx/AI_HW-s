# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:43:38 2022

@author: 12036
"""

import numpy as np
from hmmlearn import hmm

L = ["H", "C"]
L_label = len(L)  

C = ["s", "m","l"]
C_set = len(C)

initial_probability = np.array([0.6, 0.4])

trans = np.array([[0.7, 0.3],[0.4, 0.6]])

emission = np.array([[0.1, 0.4, 0.5],[ 0.7, 0.2, 0.1]])
print('\n Markov Model...\n')
model = hmm.MultinomialHMM(n_components=L_label) 


model.startprob_=initial_probability
model.transmat_=trans
model.emissionprob_=emission

S = np.array([[0,0,2,1,2]]).T      
logprob, box = model.decode(S, algorithm="viterbi")
S = [0,0,2,1,2]


seen = np.array([[0,0,2,1,2]]).T 
box2 = model.predict(seen)      
seen = [0,0,2,1,2]
print("\Size sequence :", ", ".join(map(lambda x: C[x], seen)))
print("\nTemperature  Sequence :", ", ".join(map(lambda x: L[x], box2)))


seen = np.array([[0,0,2,1,2]]).T                 
print("\nProbability P(HC)= ",model.score(seen))