# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 01:36:56 2019

@author: pranitj
"""

import pandas as pd
import numpy as np
import time

start = time.time()

# 1.1 Read input file and assign column names
def read_file(file):
    
    print("\n--- Step 1: Data Input ---")
    
    df = pd.read_csv(file, header = None)
    df.columns = ['journals', 'cited_journals', 'links']
    
    return(df)
    
# 1.2 Create adjacency matrix
def calculate_Z(df):
    
    print("\n--- Step 2: Creating Adjacency Matrix ---")
    
    unique_columns_rows = np.unique(df[['journals', 'cited_journals']])
    adjacency_matrix_Z = df.set_index(['cited_journals', 'journals'])['links'].unstack(fill_value = 0).reindex(columns = unique_columns_rows, index = unique_columns_rows, fill_value = 0).values
    adjacency_matrix_Z = adjacency_matrix_Z.astype(float)
    
    print("\nAdjacency Matrix\n")
    print(adjacency_matrix_Z)
    
    return(adjacency_matrix_Z)

# 1.3 Modify adjacency matrix
def modifying_Z(adjacency_matrix_Z):
    
    print("\n--- Step 3: Modifying Adjacency Matrix ---")
    
    # set diagonal to zero
    np.fill_diagonal(adjacency_matrix_Z, 0)

    # normalize columns
    sum_of_each_col = adjacency_matrix_Z.sum(axis = 0)

    H = adjacency_matrix_Z

    for i in range(0, adjacency_matrix_Z.shape[1]):
        if(sum_of_each_col[i] == 0):
            continue
        H[:, i] = np.true_divide(adjacency_matrix_Z[:, i], sum_of_each_col[i])
    
    print("\nModified Adjacency Matrix\n")
    print(H)
    
    return(H)
    
# 1.4 Identify the dangling nodes
def dangling_nodes(H):

    print("\n--- Step 4: Identifying Dangling Nodes ---")
    
    d = []
    
    # create dangling nodes row vector
    for i in range(0, H.shape[1]):
        if (np.count_nonzero(H[:, i]) > 0):
            d.append(0)
        else:
            d.append(1)
    
    d = np.asarray([d])
    
    print("\nDangling Nodes Row Vector\n")
    print(d)
    
    return(d)

# 1.5 Calculating the Influence Vector
def calculate_influence_vector(H, d):
    
    print("\n--- Step 5: Calculating the Stationary Vector ---")
    
    # create Article vector with size (H.shape[1], 1) and values as element divided by (1 / H.shape[1]) 
    a = np.full((H.shape[1], 1), (1 / H.shape[1]))
    A = a / a.sum(axis = 0)
    
    # create (k = 0) pi with size as (H.shape[1], 1) and element values as (1 / H.shape[1])
    pi_current = np.full((H.shape[1], 1), (1 / H.shape[1]))
    
    # initialize alpha and epsilon 
    alpha = 0.85
    epsilon = 0.00001
    
    # calculate (k + 1) pi
    equation1 = np.dot(alpha, np.matmul(H, pi_current))
    equation2 = np.dot(alpha, np.matmul(d, pi_current)) + (1 - alpha)
    equation2_scalar_value = equation2.item([0][0])
    equation3 = np.dot(equation2_scalar_value, A)
    pi_calculated = equation1 + equation3
    
    count = 1
    
    # calculate Influence vector
    while (np.linalg.norm(pi_calculated - pi_current) > epsilon):
        
        pi_current = pi_calculated
        equation1 = np.dot(alpha, np.matmul(H, pi_current))
        equation2 = np.dot(alpha, np.matmul(d, pi_current)) + (1 - alpha)
        equation2_scalar_value = equation2.item([0][0])
        equation3 = np.dot(equation2_scalar_value, A)
        pi_calculated = equation1 + equation3
        
        count += 1
        
    print("\nNumber of iterations to convergence: {}".format(count))
    print("\nInfluence Vector\n")
    print(pi_calculated)
    
    return(pi_calculated)

# 1.6 Calculating Eigenfactor (EF)
def eigen_factor(H, pi_calculated):
    
    print("\n--- Step 6: Calculationg the EigenFactor (EF) Score ---")
    
    # calculate numerator of EF equation
    H_pi = np.dot(H, pi_calculated)
    
    # calculate denominator of EF equation
    H_pi_sum = H_pi.sum(axis = 0)
    
    # calculate EF
    EF = 100 * (H_pi / H_pi_sum)
    
    # store EF numpy array to pandas dataframe to store Journal as index 
    EF_df = pd.DataFrame(EF).sort_values(by = 0, ascending = False)
    EF_df.columns = ['eigenfactor_scores']
    
    # print top 20 journals with its eigen factors
    print("\nTop 20 Journals with EigenFactor Scores\n")
    print(EF_df.head(20))
 
# Execution of functions
df = read_file('links.txt')
adjacency_matrix_Z = calculate_Z(df)
H = modifying_Z(adjacency_matrix_Z)
d = dangling_nodes(H)
pi_calculated = calculate_influence_vector(H, d)
eigen_factor(H, pi_calculated)

# calculate total time of execution
end = time.time()
print("\nTotal time for execution: {}".format(end - start))


"""
(a) Reported Scores
Journal 4408 and EigenFactor Score 1.447538
Journal 4801 and EigenFactor Score 1.412038
Journal 6610 and EigenFactor Score 1.234606
Journal 2056 and EigenFactor Score 0.679335
Journal 6919 and EigenFactor Score 0.664692
Journal 6667 and EigenFactor Score 0.634253
Journal 4024 and EigenFactor Score 0.576867
Journal 6523 and EigenFactor Score 0.480609
Journal 8930 and EigenFactor Score 0.477589
Journal 6857 and EigenFactor Score 0.439622
Journal 5966 and EigenFactor Score 0.429627
Journal 1995 and EigenFactor Score 0.385984
Journal 1935 and EigenFactor Score 0.385048
Journal 3480 and EigenFactor Score 0.379524
Journal 4598 and EigenFactor Score 0.372625
Journal 2880 and EigenFactor Score 0.330194
Journal 3314 and EigenFactor Score 0.327306
Journal 6569 and EigenFactor Score 0.319195
Journal 5035 and EigenFactor Score 0.316591
Journal 1212 and EigenFactor Score 0.311212

(b) Execution Time (Approx)
8.x seconds

(c) Number of iterations to get to the answer
21
"""