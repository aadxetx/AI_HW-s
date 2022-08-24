# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:38:40 2022

@author: 12036
"""

import numpy as np
from hmmlearn import hmm

L = ["B1", "B2", "B3"]
L_label = len(L)  

C = ["red", "green","yellow"]
C_color = len(C)

initial_probability = np.array([0.4, 0.35, 0.25])

trans = np.array([
  [0.3, 0.2, 0.5],
  [0.1, 0.3, 0.6],
  [0.7, 0.25, 0.05]
])

emission = np.array([
  [0.8, 0.1, 0.1],
  [0.2, 0.4, 0.4],
  [0.15, 0.25, 0.6]
])
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
print("\nColor sequence :", ", ".join(map(lambda x: C[x], seen)))
print("\nBottle Sequence :", ", ".join(map(lambda x: L[x], box2)))


seen = np.array([[0,0,2,1,2]]).T                 
print("\nProbability P(rrygy)= ",model.score(seen))