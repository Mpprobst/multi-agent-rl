#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:34:16 2020
The training log output produced by the MADDPG code in text file is parsed to
extract running average
@author: subashkhanal
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

#location = '/Users/subashkhanal/Desktop/Fall_2020/Sequential_Decision_Making/Final_Project/maddpg/results'

#filename = os.path.join(location,'simple_20000_1000.txt')
#filename = os.path.join(location,'simpleWorldComm_2_20000_1000.txt')
parser = argparse.ArgumentParser(description='Define the problem to solve.')
parser.add_argument('-f', default='simple.py', help='Path of the file to read Python')
args = parser.parse_args()

run_avg = []
"""
with open(filename,'r') as infile:
    txt = infile.read()
    txt = txt.split('\n')[2:-1]
    for line in txt:
        avg = float(line.split(' ')[7].strip(','))

        run_avg.append(avg)
"""
with open(args.f, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        print(row)
        run_avg.append(row[1])

#numpy_file = 'simple_500_10.npy'
numpy_file = 'simpleWorldComm_4_50_10.npy'
np.save(numpy_file,np.array(run_avg))

x = range(len(run_avg))
plt.plot(x,run_avg)
#plt.title('Simple Environment')
plt.title('Simple  World Communication Environment')
plt.xlabel('10th training episode')
plt.ylabel('average score')
print('plotted')
