#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:27:12 2020

@author: atticus
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse

# =================== CE Auxiliary Functions =================================

# CE aux
def get_clusters_points(triclusters):
    
    points = dict()
    for k in triclusters.keys():
        rows = triclusters[k][0]
        cols = triclusters[k][1]
        ctxs = triclusters[k][2]
        points[k] = [(row, col, ctx) for row in rows for col in cols for ctx in ctxs]
        #print('Tric ', k, ' Length: ', len(points[k]))
    
    return points

# CE aux
def calculate_confusion_matrix(solution1, solution2):
    
    confusion_matrix = np.zeros((len(solution1), len(solution2)))
    
    for s1 in range(0, len(solution1)):
        for s2 in range(0, len(solution2)):
            s1_key = list(solution1.keys())[s1]
            s2_key = list(solution2.keys())[s2]
            confusion_matrix[s1][s2] = len(set(solution1[s1_key])
                                           .intersection(solution2[s2_key]))
    
    return confusion_matrix

# CE aux
def support_matrix(solution, nrows, ncols, nctxs):
    
    matrix = np.zeros((nrows, ncols, nctxs))
    
    for k in solution.keys():
        for row, col, ctx in solution[k]:
            matrix[row][col][ctx] += 1
    
    return matrix
    
# CE aux
def calculate_union_size(solution1, solution2, nrows, ncols, nctxs):
    
    s1 = support_matrix(solution1, nrows, ncols, nctxs)
    s2 = support_matrix(solution2, nrows, ncols, nctxs)
    
    return np.maximum(s1, s2).sum()

# ============================================================================

# =================== MSR Auxiliary Functions ================================
def calculate_mu(data):
    
    d_IJK = data.mean()
    d_iJK = np.full(data.shape[1], np.inf)
    d_IjK = np.full(data.shape[2], np.inf)
    d_IJk = np.full(data.shape[0], np.inf)
    mu_ijk = 0
    
    for k in range(0, data.shape[0]):
        for i in range(0, data.shape[1]):
            for j in range(0, data.shape[2]):
                
                if d_iJK[i] == np.inf:
                    d_iJK[i] = data[:, [i], :].mean()
                
                if d_IjK[j] == np.inf:
                    d_IjK[j] = data[:, :, [j]].mean()

                if d_IJk[k] == np.inf:
                    d_IJk[k] = data[k].mean()
                                
                mu_ijk += pow((data[k][i][j] - d_iJK[i] - d_IjK[j] - d_IJk[k] + 2 * d_IJK),2)
    
    
    #print('d_IJK=', d_IJK)
    #print('d_iJK=', d_iJK)
    #print('d_IjK=', d_IjK)
    #print('d_IJk=', d_IJk, '\n\n')
    #print('mu_ijk=', mu_ijk)
    
    return mu_ijk

# ============================================================================

# =================== Recoveravility Auxiliary Functions ====================
def intersection_rate(t, s):
   
    #print('S:', s)
    
    gt_rows = set(t[0])
    gt_cols = set(t[1])
    gt_ctxs = set(t[2])
    
    rows_intersect = gt_rows.intersection(set(s[0]))
    cols_intersect = gt_cols.intersection(set(s[1]))
    ctxs_intersect = gt_ctxs.intersection(set(s[2]))
    
    shared_elems = len(rows_intersect) * len(cols_intersect) * len(ctxs_intersect)
    dim = len(gt_rows) * len(gt_cols) * len(gt_ctxs)
    
    return shared_elems / dim

# ============================================================================

# ========================= ITHa Auxiliary Functions =========================
def calc_observations_pc(data, t):
    
    mr_t = data[t].mean(axis=0)
    pc = np.nan_to_num([np.corrcoef(data[t, r, :], mr_t)[0,1] for r in range(0, data.shape[1])], nan=1)
    return pc.mean()
# ============================================================================

# Clustering Error 3D
def ClusteringError3D(ground_truth, clust_solution, nrows, ncols, nctxs):
    ground_truth = get_clusters_points(ground_truth)
    clust_solution = get_clusters_points(clust_solution)
    
    #print('Ground Truth: {}\n'.format(ground_truth))
    #print('Clustering Solution: {}\n'.format(clust_solution))
    
    confusion_matrix = calculate_confusion_matrix(ground_truth, clust_solution)
    confusion_matrix = np.negative(confusion_matrix)
    #print('Confusion Matrix calculated: {}\n'.format(confusion_matrix))
    
    row_ind, col_ind = linear_sum_assignment(confusion_matrix)
    d_max = -confusion_matrix[row_ind, col_ind].sum()
    #print('DMax = {}'.format(d_max))
    
    union_size = calculate_union_size(ground_truth, clust_solution, nrows, ncols, nctxs)
    #print('Union size = {}\n\n'.format(union_size))
    
    error = (union_size - d_max) / union_size
    #error_matlab = 1 - ((union_size - d_max) / union_size)
    
    #print('Error = {}\n'.format(error))
    #print('Error MATLAB = {}\n'.format(error_matlab))
    return error

# Mean Squared Residue 3D
def MSR3D(data):
    
    data_dim = data.shape[0] * data.shape[1] * data.shape[2]    
    mu_ijk = calculate_mu(data)
    return (1 / data_dim) *  mu_ijk

    
def IntraTemporalHomogeneity(data):
    
    avg_pc = np.array([calc_observations_pc(data, t) for t in range(0, data.shape[0])])
    return avg_pc.mean()

def InterTemporalHomogeneity(data):
        
    mmr = data.mean(axis = 0).mean(axis = 0)
    #print(mmr)
    pc = np.nan_to_num([np.corrcoef(data[t].mean(axis=0), mmr)[0,1] for t in range(0, data.shape[0])],
                       nan=1)
    #print(pc)
    
    return np.array(pc).mean()

def Recoverability(ground_truth, solution):
    
    res = dict()
    
    for i, t in ground_truth.items():
        #print(t)
        res[i] = 0 if len(solution) == 0 else round(max([intersection_rate(t, s) for s in solution.values()]), 2)
    
    return res

#data = np.array([[[1,3,2,5], [3,5,2,5], [2,3,3,6]]])
    
#data = np.array([[[1,1,1],[3,3,3],[9,9,9]],
#                 [[2,2,2], [6,6,6], [18,18,18]],
#                 [[4,4,4], [12,12,12], [36,36,36]]])
    
#data = np.array([[[1,1,1],[1,1,1],[1,1,1]],
#                 [[1,5,1], [1,5,1], [1,5,1]],
#                 [[1,1,1], [1,1,1], [1,1,1]]])

#data = np.array([[[1,5,4],[2,4,3],[1,4,2]],
 #                [[1,3,2],[1,2,2],[3,5,4]],
 #                [[5,7,3],[4,6,1],[3,8,7]],
 #                [[4,5,1],[3,7,4],[5,10,1]]])

#print('Intra-Temporal Homogeneity: ', IntraTemporalHomogeneity(data))
#print('Inter-Temporal Homogeneity: ', InterTemporalHomogeneity(data))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', metavar='metric', help='CE: Clustering Error\n R: Recoverability')
    parser.add_argument('--gt', help='Ground Truth using a dict in the form {<tric_id>:[[rows],[cols],[ctxs]]}')
    parser.add_argument('--sol', help='Solution using a dict in the form {<tric_id>:[[rows],[cols],[ctxs]]}')
    parser.add_argument('--nrows', nargs='?', type=int, help='N. of rows in the dataset')
    parser.add_argument('--ncols', nargs='?', type=int, help='N. of cols in the dataset')
    parser.add_argument('--nctxs', nargs='?', type=int, help='N. of ctxs in the dataset')
    args = parser.parse_args()
    
    if args.metric == 'CE':
        print(ClusteringError3D(eval(args.gt), eval(args.sol), args.nrows, args.ncols, args.nctxs))
    elif args.metric == 'R':
        print(Recoverability(eval(args.gt), eval(args.sol)))
    else:
        print('Function not supported')