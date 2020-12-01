#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:34:16 2020
The training log output produced by the MADDPG code in text file is parsed to
extract running average 
@author: subashkhanal
"""

import numpy as np
import matplotlib.pyplot as plt
import os

location = '/Users/subashkhanal/Desktop/Fall_2020/Sequential_Decision_Making/Final_Project/maddpg/results'

#filename = os.path.join(location,'simple_20000_1000.txt')
filename = os.path.join(location,'simpleWorldComm_2_20000_1000.txt')

run_avg = []
with open(filename,'r') as infile:
    txt = infile.read()
    txt = txt.split('\n')[2:-1]
    for line in txt:
        avg = float(line.split(' ')[7].strip(','))
        
        run_avg.append(avg)

#numpy_file = os.path.join(location,'simple_20000_1000.npy')
numpy_file = os.path.join(location,'simpleWorldComm_2_20000_1000.npy')
np.save(numpy_file,np.array(run_avg))
x = range(len(run_avg))       
plt.plot(x,run_avg)
#plt.title('Simple Environment')
plt.title('Simple  World Communication Environment')
plt.xlabel('1000th  training episode')
plt.ylabel('Running average score')
