# requires scipy, dtw, fastdtw, numpy, matplotlib
# all available by pip install
from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import math
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.stats.stats import pearsonr

# load the time serie we want to compare
x = np.array([])
# load the time series of known points
y = [np.array([])]

# test time series
x = np.array([1, 1, 1, 1, 2, 4, 2, 1, 2, 1])
y = [np.array([10, 11, 11, 11, 2, 4, 3, 7, 8, 9]),  np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 1])]

x1 = np.array([1, 1, 1, 3, 6, 13, 25, 22, 7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
y1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 12, 24, 23, 8, 3, 1, 1, 1, 1, 1, 1])


"""

Root Mean Squared Error calculation

"""
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print ("RMSE: " + str(rmse(x, y[0])))
print ("RMSE: " + str(rmse(x1,y1)))

"""

Euclidian distance between 2 time series

"""
def euclid_dist(t1,t2):
    return np.sqrt(sum((t1-t2)**2))
print("Euclidian distance: " + str(euclid_dist(x,y[0])))
print ("Euclidian distance: " + str(euclid_dist(x1,y1)))


"""

Log returns, returns, moving average
Pearson's correlation coefficient

"""

x_logreturn = np.log(1 + (x / np.roll(x, -1)))
y_logreturn = np.log(1 + (y[0] / np.roll(y[0], -1)))
x1_logreturn = np.log(1 + (x1 / np.roll(x1, -1)))
y1_logreturn = np.log(1 + (y1 / np.roll(y1, -1)))

print ("Pearson's correlation on log returns: " + str(pearsonr(x_logreturn,y_logreturn)))
print ("Pearson's correlation on log returns: " + str(pearsonr(x1_logreturn,y1_logreturn)))




# DTW implementation 1
def dtw1 (x, y):
    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)
    dist, cost, acc, path = dtw(x_reshaped, y_reshaped, dist=lambda x_reshaped, y_reshaped: norm(x_reshaped - y_reshaped, ord=1))
    return dist

# DTW Implementation 2 Ref : https://jeremykun.com/2012/07/25/dynamic-time-warping/
def dtw2(seqA, seqB, d = lambda x,y: abs(x-y)):
    # create the cost matrix
    numRows, numCols = len(seqA), len(seqB)
    cost = [[0 for _ in range(numCols)] for _ in range(numRows)]
    # initialize the first row and column
    cost[0][0] = d(seqA[0], seqB[0])
    for i in range(1, numRows):
        cost[i][0] = cost[i-1][0] + d(seqA[i], seqB[0])
    for j in range(1, numCols):
        cost[0][j] = cost[0][j-1] + d(seqA[0], seqB[j])
    # fill in the rest of the matrix
    for i in range(1, numRows):
        for j in range(1, numCols):
            choices = cost[i-1][j], cost[i][j-1], cost[i-1][j-1]
            cost[i][j] = min(choices) + d(seqA[i], seqB[j])
    return cost[-1][-1]

# DTW Implementation 3 Ref: http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
def dtw3(s1, s2):
    DTW={}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

"""
#¤ DTW Implementation 4 Ref: https://effectiveml.com/dynamic-time-warping-for-sequence-classification.html
Implementation to fix before using
def dtw4(p,q):
    p = p.reshape(-1, 1)
    q = q.reshape(-1, 1)
    ep = np.sqrt(np.sum(np.square(p),axis=1));
    eq = np.sqrt(np.sum(np.square(q),axis=1));
    D = 1 - np.dot(p,q.T)/np.outer(ep,eq) # work out D all at once
    S = np.zeros_like(D)
    Lp = np.shape(p)[0]
    Lq = np.shape(q)[0]
    N = np.shape(p)[1]
    for i in range(Lp):
        for j in range(Lq):          
            if i==0 and j==0:  S[i,j] = D[i,j]
            elif i==0: S[i,j] = S[i,j-1] + D[i,j]
            elif j==0: S[i,j] = S[i-1,j] + D[i,j]
            else: S[i,j] = np.min([S[i-1,j],S[i,j-1],S[i-1,j-1]]) + D[i,j]
    return np.sqrt(S[-1,-1]) # return the bottom right hand corner distance
"""

# DTW Implementation 5 Ref : https://pypi.org/project/fastdtw/0.3.0/
def dtw5 (x, y):
    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)
    distance, path = fastdtw(x_reshaped, y_reshaped, dist=euclidean)
    return distance


# iterating over the known points and find the most similar
rmse_result = []
euclid_result = []
dtw1_result = []
dtw2_result = []
dtw3_result = []
dtw5_result = []
for i in range(len(y)):
    rmse_result.append(rmse(x, y[i]))
    euclid_result.append(euclid_dist(x,y[i]))
    dtw1_result.append(dtw1(x, y[i]))
    dtw2_result.append(dtw2(x, y[i]))
    dtw3_result.append(dtw3(x, y[i]))
    dtw5_result.append(dtw5(x, y[i]))

print("")
print(rmse_result)
print(euclid_result)
print(dtw1_result)
print(dtw2_result)
print(dtw3_result)
print(dtw5_result)
print("")

print ("Most similar rmse is point               : " + str(np.argmin(rmse_result)) + " " +str(rmse_result[np.argmin(rmse_result)]))
print ("Most similar euclidian distance is point : " + str(np.argmin(euclid_result)) + " " +str(euclid_result[np.argmin(euclid_result)]))
print ("Most similar dtw1 is point               : " + str(np.argmin(dtw1_result)) + " " +str(dtw1_result[np.argmin(dtw1_result)]))
print ("Most similar dtw2 is point               : " + str(np.argmin(dtw2_result)) + " " +str(dtw2_result[np.argmin(dtw2_result)]))
print ("Most similar dtw3 is point               : " + str(np.argmin(dtw3_result)) + " " +str(dtw3_result[np.argmin(dtw3_result)]))
print ("Most similar dtw5 is point               : " + str(np.argmin(dtw5_result)) + " " +str(dtw5_result[np.argmin(dtw5_result)]))