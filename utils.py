#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 21:09:14 2022

@author: Chenyin Gao
"""

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly import decomposition
from scipy.special import expit
from tensorly import tucker_to_tensor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis
import itertools
import matplotlib.pyplot as plt
## handy functions
# estimation error in the probably scale
def theta2prob(theta_hat):
    return np.apply_along_axis(np.cumprod, 1, expit(theta_hat))  

# Data Generation
def gen_data_potential_Y_binary_a(N, T, B, 
                                  X0, b, d = 3):
    # generate factors
    eta_N = np.ones(d); eta_T = np.ones(T); eta_B = 1
    ## subject factor
    e_U1 = X0 @ eta_N
    ## temporal factor
    e_U2 = eta_T * np.linspace(1, T, num = T)/T
    ## treatment factor
    e_U3 = 1 + eta_B * np.sum([int(a) for a in (bin(b)[2:])]) 
    theta_true_b = np.outer(e_U1, e_U2) * e_U3
    return theta_true_b
    
def gen_data_potential_Y_binary(seed, N, T, k, d = 3):
    
    B = int(2**k) 
    np.random.seed(seed)
    theta_true = np.zeros((N, T, B))
    X0 = np.random.normal(0, 1, size = (N, d))
    for b in range(B):
        theta_true[:, :, b] = gen_data_potential_Y_binary_a(N, T, B, 
                                                            X0, b, d)
    error = np.random.logistic(scale = 1, loc = 0, size = (N, T, B))
    # generate the potential outcomes under all treatment history
    Y = np.zeros((N, T, B), dtype = 'float')
    A = np.zeros((N,k), dtype = 'int')
    delta = np.zeros((N))
    for i in range(N):
        # randomly assigned treatment for all the k regimes
        for k_a in range(k):
            b = np.random.binomial(1, expit(np.sum(.5 * X0[i, :])))
            A[i, k_a] = b
        # set up the binary summation
        binary_count = np.array([pow(2, power_value) for power_value in range(k)])[::-1]
        b_decimal = (binary_count * A[i,:]).sum()
        for t in range(T):
            if t==0: 
                Y[i, t, b_decimal] = int((theta_true[i, t, b_decimal] + \
                    error[i, t, b_decimal])>0) #np.random.binomial(1, prob_Y)
            elif Y[i, t-1, b_decimal] == 0:
                Y[i, t, b_decimal] = 0
            else:
                Y[i, t, b_decimal] =  int((theta_true[i, t, b_decimal] + \
                    error[i, t, b_decimal])>0)
        # generate censoring
        ## randomly select 20% to be censored
        delta[i] = np.random.binomial(size=1, n = 1, p = 1 - 0.2)
        if delta[i] == 0:
            C = np.random.uniform(0, Y[i, :, :].sum(), 1)
            Y[i, int(np.floor(C)) : int(Y[i, :, :].sum()), b_decimal] = 0
    return Y, A, delta, X0, theta_true, error


# Class for the low-rank tensor block hazard model
class TensorCompletionCovariateBinary():
    """
    Tensor Completion projected on the covariate space (similar to STEFA)
    """
    def __init__(self,
                 Y, A, X, delta = None, rho = None,
                 # optimization parameter
                 stepsize = 1e-10,
                 momentum = 0,
                 niters = 5000, tol = 1e-8,
                 r1_list = [4], r2_list = [2], r3_list = [8],
                 lam_list = [0],
                 # split = False, 
                 verbose = True, 
                 method = 'block'):
        """
        stepsize: stepsize for projected gradient descent
        tau: the number of iterations
        """
        
        self.Y = Y
        self.A = A
        self.X = X
        self.rho = rho
        
        if delta is None:
            self.delta = np.ones(len(Y)) # assume all customer exits the company
        else:
            self.delta = delta
        self.niters = niters
        self.stepsize = stepsize
        self.momentum = momentum
        self.tol = tol
        
        self.r1_list = r1_list
        self.r2_list = r2_list
        self.r3_list = r3_list
        
        if lam_list is None:
            self.lam_list = [0.01 * np.prod(Y.shape[:2])]
        else:
            self.lam_list = lam_list
        
        self.verbose = verbose
        self.method = method
    def label2matrix(self, labels, K):
        # identify the numbder of cluster
        # K = np.unique(labels).shape[0]
        # create the place-holder
        W = np.zeros((labels.shape[0], K))
        
        for j in range(labels.shape[0]):
            W[j, int(labels[j])] = 1
        W_sum = np.sum(W, axis = 0)
        W_sum[W_sum == 0] = 1
        # return W/W_sum
        return W
    def _TC_Tucker_project(self, Y, A, X, rho,
                           niters, stepsize, momentum,
                           r1, r2, r3, lam):
        
        
        mu_max = 10
        L_max = 20
        N, T, B = Y.shape
        k = int(np.log2(B))
        if rho is None:
            rho = np.ones(N)
        # GLM modeling for initialization
        theta_tensor = get_theta_binary(Y = Y, A = A, X = X)
        
        
        # trimm for weights larger than 1/0.1
        P_X0 = X @ np.linalg.pinv(X.T@X) @ X.T
        
        
        # should code the high-order spectral clustering
        F_core, [U_1, U_2, U_3] = decomposition.tucker(theta_tensor, n_iter_max = niters, 
                                                          rank = [r1, r2, r3],
                                                           svd = 'truncated_svd',
                                                          )
        
        if self.method == 'block':
            # k-means on the third mode for discretization
            theta_3 = U_3 @ U_3.T @ tl.unfold(tl.tenalg.multi_mode_dot(
                tl.tenalg.multi_mode_dot(F_core, [U_1, U_2, U_3], [0, 1, 2]),
                [U_1.T, U_2.T], [0, 1]),
                mode = 2)
            
            # initialization by k-means algorithm  (need relaxation?)
            kmeans = KMeans(n_clusters = r3, n_init = r3)
            kmeans.fit(theta_3)
            labels = kmeans.labels_
            # update the initial U_3
            U_3 = self.label2matrix(labels, r3)
            # update the core tensor
            U3_sum = np.sum(U_3, axis = 0)
            U3_sum[U3_sum == 0] = 1
            F_core = tl.tenalg.multi_mode_dot(theta_tensor, [U_1.T, U_2.T, (U_3/U3_sum).T], [0, 1, 2])
        # initialization
        U_1 = P_X0 @ U_1
        loss = []
        tol_temp = 1e3
        
        U_1_change = np.zeros((N, r1));
        U_2_change = np.zeros((T, r2)) 
        U_3_change = np.zeros((B, r3)) 
        # gradient descent
        for it in range(niters):
            theta_pre = tucker_to_tensor((F_core, [(U_1), 
                                                    U_2,
                                                    U_3]))
            
            L_nabula = -expit(theta_pre) * (1 - expit(theta_pre))* \
                (Y  + (Y - 1) * self.delta[:, None, None])/expit((2 * Y - 1) * theta_pre)
            # add a mask related to Y_{i,t-1,l}
            mask0 = np.zeros((N, B))
            binary_count = np.array([pow(2, power_value) for power_value in range(k)])[::-1]
            for i in range(N):
                b_decimal = (binary_count * A[i,:]).sum()
                # print(b_decimal)
                mask0[i, b_decimal] = 1/rho[i]
                
                
            rho_tensor = np.repeat(np.repeat(rho[:, None, None], T , axis = 1),
                      B, axis = 2)
            mask = np.append(mask0[:, None, :], Y/rho_tensor,
                             axis = 1)[:, :T, :]
            self.mask = mask
            L_nabula = L_nabula * mask
            
            # record the loss function pre-GD
            loss_pre = -(np.log(expit((2 * Y - 1) * theta_pre)) * mask).sum()/N
            
            # else: # no split for gradient descent
            # grad for core F
            grad_f_X_unflod_mode1 = tl.unfold(L_nabula, mode = 0)
            grad_f_X_unflod_mode2 = tl.unfold(L_nabula, mode = 1)
            grad_f_X_unflod_mode3 = tl.unfold(L_nabula, mode = 2)
            
            
            grad_f_F_fold = tl.tenalg.multi_mode_dot(L_nabula,
                                   [(U_1).T, U_2.T, U_3.T],
                                  # [U_1.T, U_2.T, U_3.T],
                                  [0, 1, 2])
            # grad for U1
            # X.T @ 
            grad_U_1 = grad_f_X_unflod_mode1 @ \
                tl.tenalg.kronecker([U_2, U_3]) @\
                    tl.unfold(F_core, 0).T
            # grad for U2
            grad_U_2 = grad_f_X_unflod_mode2 @ \
            tl.tenalg.kronecker([(U_1), U_3]) @\
                    tl.unfold(F_core, 1).T
                        # tl.tenalg.kronecker([U_1, U_3]) @\
                    # tl.tenalg.kronecker([(X @ U_1), U_3]) @\
            def otimes(A):
                return A @ A.T

                
            #------------Gradient descent
            new_change_U1 = stepsize * grad_U_1 + momentum * U_1_change
            U_1 = U_1 - new_change_U1
            U_1_change = new_change_U1
            # add information from baseline X
            U_1 =  X @ np.linalg.pinv(X.T@X) @ X.T @ U_1 # projected GD
            
            
            new_change_U2 = stepsize * grad_U_2 + momentum * U_2_change
            U_2 = U_2 - new_change_U2
            U_2_change = new_change_U2
            
            # project the factor matrices onto the restricted set
            U_1_2max = np.linalg.norm(U_1, axis = 1)**2*N/r1
            U_1[(U_1_2max > mu_max).nonzero()[0], :] = U_1[(U_1_2max > mu_max).nonzero()[0], :]/\
                U_1_2max[(U_1_2max > mu_max).nonzero()[0], None] * mu_max
            
            U_2_2max = np.linalg.norm(U_2, axis = 1)**2*T/r2
            U_2[(U_2_2max > mu_max).nonzero()[0], :] = U_2[(U_2_2max > mu_max).nonzero()[0], :]/\
                U_2_2max[(U_2_2max > mu_max).nonzero()[0], None] * mu_max
                
            # update
            F_core = F_core - stepsize * (grad_f_F_fold +\
                                          lam * np.ones((r1, r2, r3)) * np.sign(F_core))
            
            G_norm2_max = np.array([np.linalg.norm(tl.unfold(F_core, 0), ord = 2),
                      np.linalg.norm(tl.unfold(F_core, 1), ord = 2),
                      np.linalg.norm(tl.unfold(F_core, 2), ord = 2)]).max()
            G_norm2_limit = L_max * np.sqrt(N*T*B/(mu_max**3/2*(r1*r2*r3)**(1/2)))
            F_core = F_core/G_norm2_max*G_norm2_limit
            
            # update U_3
            if self.method == 'block':
                # project G onto the restricted set
                F_core_3 = tl.unfold(tl.tenalg.multi_mode_dot(
                    tl.tenalg.multi_mode_dot(F_core, [U_1, U_2, U_3], [0, 1, 2]),
                                          [U_1.T, U_2.T, (U_3/U3_sum).T], [0, 1, 2]), 
                    mode = 2)
                # compute the projected block mean based on labels
                theta_3 = tl.unfold(tl.tenalg.multi_mode_dot(
                    tl.tenalg.multi_mode_dot(F_core, [U_1, U_2, U_3], [0, 1, 2]),
                                          [U_1.T, U_2.T], [0, 1]),
                          mode = 2)
                
                labels = np.zeros(B)
                for j in range(B):
                    cluster_idx = np.sum((F_core_3 - theta_3[j, :])**2,
                                         axis = 1).argmin()
                    labels[j] = cluster_idx
                    
                U_3 = self.label2matrix(labels, r3)
            elif self.method == 'continuous':
                
                # grad for U3
                grad_U_3 = grad_f_X_unflod_mode3 @ \
                    tl.tenalg.kronecker([(U_1), U_2]) @\
                        tl.unfold(F_core, 2).T
                new_change_U3 = stepsize * grad_U_3 + momentum * U_3_change
                U_3 = U_3 - new_change_U3
                U_3_change = new_change_U3
                
            # compute the post-iteration loss
            theta_after = tucker_to_tensor((F_core, [(U_1), 
                                                    U_2,
                                                    U_3]))
            
            # record the loss function post-GD
            loss_after = -(np.log(expit((2 * Y - 1) * theta_after)) * mask).sum()/N
            loss.append(np.linalg.norm(loss_after))
            
            # record the relative loss change
            tol_temp = np.abs(np.linalg.norm(loss_pre) - np.linalg.norm(loss_after))/\
                np.linalg.norm(loss_pre)
            
            if not (it % 1000):
            # if verbose:
                print(f'(CO-Tucker): {it}th iteration with loss: {np.round(loss[-1],3)}')
                # print()
            if tol_temp < 1e-10:
                print('Stop: not enough improvement')
                break
        return (F_core, U_1, U_2, U_3), loss
    
    # BIC criterion
    def BIC(self, F_core, U_1, U_2, U_3, lam):
        theta_hat = tl.tenalg.multi_mode_dot(F_core,
                                      [U_1,
                                       U_2, U_3])
        P1 = (-(np.log(expit((2 * self.Y - 1) * theta_hat)) * self.mask).sum() + \
              lam * np.abs(F_core).sum())/np.product(self.Y.shape)  # estimation error
        P2 = np.log(np.product(self.Y.shape))/np.product(self.Y.shape) * \
            (np.product(F_core.shape) + np.product(U_1.shape) + np.product(U_2.shape) + np.product(U_3.shape)-\
             -U_1.shape[1]**2 - U_2.shape[1] **2 - U_3.shape[1]**2)
        return P1+P2
    
    # Sequentially tuning
    def _tuning_bic(self, r1, r2, r3, lam,
                    verbose = True):
        (F_core, U_1, U_2, U_3), loss  = \
            self._TC_Tucker_project(Y = self.Y, A = self.A, X = self.X, rho = self.rho,
            r1 = r1, r2 = r2, r3 = r3, lam = lam,
            stepsize = self.stepsize,
            momentum = self.momentum,
            niters = self.niters)
        bic = self.BIC(F_core, U_1, U_2, U_3, lam)
        return bic
    
    def SequentialTuning(self, out = True):
        # initialization
        r1_list = self.r1_list
        r2_list = self.r2_list
        r3_list = self.r3_list
        lam_list = self.lam_list
        
        bic = 1e10
        r1_opt = np.random.choice(r1_list);
        r2_opt = np.random.choice(r2_list);
        r3_opt = np.random.choice(r3_list);
        lam_opt = np.random.choice(lam_list)
        
        # if length = 1
        if len(r1_list) == 1:
            pass
        else:
            # for rank1  
            bic_list = [self._tuning_bic(r1, r2_opt, r3_opt, lam_opt) for r1 in r1_list]
            if np.min(bic_list) < bic:
                bic = np.min(bic_list)
                r1_opt = r1_list[np.argmin(bic_list)]
            
            
        # if length = 1
        if len(r2_list) == 1:
            pass
        else:
            # for rank1  
            bic_list = [self._tuning_bic(r1_opt, r2, r3_opt, lam_opt) for r2 in r2_list]
            if np.min(bic_list) < bic:
                bic = np.min(bic_list)
                r2_opt = r2_list[np.argmin(bic_list)]
            
        
        # if length = 1
        if len(r3_list) == 1:
            pass
        else:
            # for rank1  
            bic_list = [self._tuning_bic(r1_opt, r2_opt, r3, lam_opt) for r3 in r3_list]
            if np.min(bic_list) < bic:
                bic = np.min(bic_list)
                r3_opt = r3_list[np.argmin(bic_list)]
        
        # tuning the lam
        if len(lam_list) == 1:
            pass
        else:
            # for rank1  
            bic_list = [self._tuning_bic(r1_opt, r2_opt, r3_opt, lam) for lam in lam_list]
            if np.min(bic_list) < bic:
                bic = np.min(bic_list)
                lam_opt = lam_list[np.argmin(bic_list)]
                
        (F_core, U_1, U_2, U_3), loss = self._TC_Tucker_project(Y = self.Y, A = self.A, X = self.X,
                                                                rho = self.rho, 
                                                                r1 = r1_opt, r2 = r2_opt, r3 = r3_opt, lam = lam_opt, 
                                                                stepsize = self.stepsize, 
                                                                momentum = self.momentum,
                                                                niters = self.niters)
        self.bic = bic
        self.r1_opt = r1_opt
        self.r2_opt = r2_opt
        self.r3_opt = r3_opt
        self.lam_opt = lam_opt
        self.Y_hat = tl.tenalg.multi_mode_dot(F_core,
                                              [(U_1), # already add the projection self.X0 @ 
                                               U_2, U_3])
        self.loss = loss
        
        # store the loading matrices
        self.F_core = F_core
        self.U_1 = U_1; self.U_2 = U_2; self.U_3 = U_3
        
        if out:
            return self.Y_hat
        
# benchmark analyses of other models
## binary classification
def get_theta_binary(Y, A, X, 
                     delta = None, X_test = None, strata = False,
                     method = 'logit'):
    N, T, B = Y.shape
    k = int(np.log2(B))
    if X_test is None:
        X_test = X
    if delta is None:
        delta = np.ones(N)
    theta_hat_pred = np.zeros((len(X_test), T, B))
    _, d = X.shape
    
    # names for treatment A and covariates X
    X_names = ["{}{}".format(a_, b_) for a_, b_ in zip(np.repeat('X', d), range(1, d+1))]
    A_names = ["{}{}".format(a_, b_) for a_, b_ in zip(np.repeat('A', k), range(1, k+1))]
    col_names = ['Y'] + A_names + X_names
    for t in range(T):
        # if only contains one class
        if strata:
            for b in range(B):
                An = np.array([int(a) for a in (bin(b)[2:].zfill(k))])
                if t == 0:
                    # fit GLM at each time point
                    df = pd.DataFrame(np.column_stack((Y[(A[:]==An).flatten(), t, b],
                                                       A[(A[:]==An).flatten(), :],
                                                       X[(A[:]==An).flatten(), :])), columns = col_names)
                    
                    Xn = df[X_names].copy()
                else:
                    df = pd.DataFrame(np.column_stack((Y[(A[:]==An).flatten() & (Y[:, t-1, b]==1), t, b],
                                                        A[(A[:]==An).flatten() & (Y[:, t-1, b]==1), :], 
                                                       X[(A[:]==An).flatten() & (Y[:, t-1, b]==1), :])), 
                                      columns = col_names)
                # no such treatment regime is observed
                if sum((df[A_names]== An).all(1)) == 0:
                   theta_hat_pred[:, t, b] = theta_hat_pred.max()
                else:
                    df_b = df.loc[(df[A_names]== An).all(1),]
                    # the observed outcome is all 1 or 0 for this treatment 
                    if int(df_b['Y'].sum()) in [df_b.shape[0], 0]:
                        theta_hat_pred[:, t, b] = theta_hat_pred.max()
                    else:
                        logreg = LogisticRegression()
                        logreg.fit(X = df_b[X_names],
                                   y = df_b['Y'])
                        # update the treatment for prediction
                        # Xn[A_names] = An
                        theta_hat_pred[:, t, b] = np.array(logreg.coef_ @ X.T).flatten()
        else:
            if t == 0:
                # fit GLM at each time point
                df = pd.DataFrame(np.column_stack((Y[:, t, :].sum(axis = 1),
                                                   A, X)), columns = col_names)
                
                Xn = df[X_names].copy()
            else:
                df = pd.DataFrame(np.column_stack((Y[Y[:, t-1, :].sum(axis = 1)==1, t, :].sum(axis = 1),
                                                   A[Y[:, t-1, :].sum(axis = 1)==1, :], 
                                                   X[Y[:, t-1, :].sum(axis = 1)==1, :])), 
                                  columns = col_names)
        
            if int(df['Y'].sum()) in [df.shape[0], 0]:
                theta_hat_pred[:, t, :] = theta_hat_pred.max()
            else:
                if method == 'logit':
                    fit_binary = LogisticRegression().\
                        fit(X = df[X_names + A_names],
                               y = df['Y'])
                if method == 'SVM':
                    fit_binary = svm.SVC(probability = True).\
                        fit(X = df[X_names + A_names],
                               y = df['Y'])
                
                if method == 'GDboost':
                    fit_binary = GradientBoostingClassifier().\
                        fit(X = df[X_names + A_names],
                               y = df['Y'])
                
                if method == 'AdaBoost':
                    fit_binary = AdaBoostClassifier().\
                        fit(X = df[X_names + A_names],
                               y = df['Y'])
                
                if method == 'RandomForest':
                    fit_binary = RandomForestClassifier(random_state = 1).\
                        fit(X = df[X_names + A_names],
                            y = df['Y'])
                
                if method == 'NeuralNetwork':
                    fit_binary = MLPClassifier(hidden_layer_sizes = 2,
                                          random_state = 1, 
                                           learning_rate = 'adaptive',
                                          # activation = 'logistic',
                                          solver = 'adam',
                                          max_iter = 10000).\
                        fit(X = df[X_names + A_names], y = df['Y'])
                
                if method == 'Vote':
                    
                    clf1 = LogisticRegression()
                    clf2 = svm.SVC(probability = True)
                    clf3 = GradientBoostingClassifier()
                    clf4 = AdaBoostClassifier()
                    clf5 = RandomForestClassifier(random_state = 1)
                    clf6 = MLPClassifier(hidden_layer_sizes = 2,
                                          random_state = 1, 
                                           learning_rate = 'adaptive',
                                          # activation = 'logistic',
                                          solver = 'adam',
                                          max_iter = 10000)
                    
                    fit_binary = VotingClassifier(
                        estimators=[('lr', clf1), 
                                    ('svm', clf2), 
                                    ('gb', clf3),
                                    ('ada', clf4),
                                    ('rf', clf5),
                                    ('NN', clf6)],
                                    voting = 'soft').\
                        fit(X = df[X_names + A_names],
                               y = df['Y'])
                
                
                # prediction for all possible treatment regime
                for b in range(B):
                    # print(bin(b)[2:].zfill(k))
                    An = np.array([int(a) for a in (bin(b)[2:].zfill(k))])
                    # update the treatment for prediction
                    Xn = pd.DataFrame(X_test, columns = X_names)
                    Xn[A_names] = An
                    theta_hat_pred[:, t, b] = fit_binary.predict_proba(Xn)[:, 1]
                    
    return theta_hat_pred

## survival modeling
def get_theta_survival(Y, A, X, delta = None, X_test = None, method = 'coxPH'):
    N, T, B = Y.shape
    k = int(np.log2(B))
    if X_test is None:
        X_test = X
    if delta is None:
        delta = np.ones(N)
    surv_pred = np.zeros((len(X_test), T, B))
    _, d = X.shape
    
    # names for treatment A and covariates X
    X_names = ["{}{}".format(a_, b_) for a_, b_ in zip(np.repeat('X', d), range(1, d+1))]
    A_names = ["{}{}".format(a_, b_) for a_, b_ in zip(np.repeat('A', k), range(1, k+1))]
    # col_names = ['Y'] + A_names + X_names
    # for t in range(T):
    # document the event time
    Y_event = Y.sum(axis = (1, 2))
    # delta = Y_event < T
    Y_surv = [(e1, e2) for e1, e2 in zip(delta, Y_event)]
    Y_surv = np.array(Y_surv,
              dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    # construct the data set for survival analysis
    Xn = pd.DataFrame(np.column_stack((X, A)), columns = X_names + A_names)
    
    if method == 'coxPH':
        if d < 10:
            fit_surv = CoxPHSurvivalAnalysis().fit(X = Xn,
                                        y = Y_surv)
        else:
            fit_surv = CoxnetSurvivalAnalysis(fit_baseline_model = True).fit(X = Xn,
                                    y = Y_surv)
    if method == 'RandomForest':
        fit_surv = RandomSurvivalForest(random_state = 1).fit(X = Xn,
                                    y = Y_surv)
    
    if method == 'GDboost':
        fit_surv = GradientBoostingSurvivalAnalysis(random_state = 1).fit(X = Xn,
                                    y = Y_surv)
        
    if method == 'mboost':
        fit_surv = ComponentwiseGradientBoostingSurvivalAnalysis(random_state = 1).fit(X = Xn,
                                    y = Y_surv)
    
    for b in range(B):
        # print(bin(b)[2:].zfill(k))
        An = np.array([int(a) for a in (bin(b)[2:].zfill(k))])
        # update the treatment for prediction
        Xn = pd.DataFrame(X_test, columns = X_names)
        Xn[A_names] = An
        pred = fit_surv.predict_survival_function(X = Xn)
        predM = np.array([p.y for p in pred])
        # if full time is observed
        if predM.shape[1] > T:
            predM = predM[:, range(T)]
        # if not, use the last obs carry forward
        if predM.shape[1] < T:
            predExtra = np.repeat(predM[:, -2:-1], 
                                  T - predM.shape[1], axis = 1)
            predM = np.concatenate((predM, predExtra), axis = 1)
        surv_pred[:, :, b] = predM
    return surv_pred   

# Metrics for evaluation
## cumulative regret
def regime_eval(Y, A, rho, X, d, theta_true,
                theta_hat = None, prob_NTL = None):
    N, T, B = Y.shape
    prob_NL_true = np.apply_along_axis(np.cumprod, 1, expit(theta_true)).\
        mean(axis = (1))
    if prob_NTL is None:
        prob_NTL = np.apply_along_axis(np.cumprod, 1, expit(theta_hat))
    prob_sum_NL = prob_NTL.mean(axis = (1))
    # obtain the optimal regime for each individual
    regime_opt_true = prob_NL_true.argmax(axis = 1)
    regime_opt = prob_sum_NL.argmax(axis = 1)
    
    # true
    theta_opt = np.array([gen_data_potential_Y_binary_a(N, T, B, X[i, :], 
                                  regime_opt_true[i], d) for i in range(N)])
    prob_opt = np.apply_along_axis(np.cumprod, 1, expit(theta_opt)).\
        sum(axis = (1,2)).mean()
    
    # estimate
    theta_regime = np.array([gen_data_potential_Y_binary_a(N, T, B, X[i, :], 
                                  regime_opt[i], d) for i in range(N)])
    
    prob_regime = np.apply_along_axis(np.cumprod, 1, expit(theta_regime))
    return prob_opt - prob_regime.sum(axis = (1,2)).mean()
## decision accuracy
def regime_prec(Y, A, rho, X, d, theta_true,
                theta_hat = None, prob_NTL = None):
    N, T, B = Y.shape
    prob_NL_true = np.apply_along_axis(np.cumprod, 1, expit(theta_true)).\
        mean(axis = (1))
    if prob_NTL is None:
        prob_NTL = np.apply_along_axis(np.cumprod, 1, expit(theta_hat))
    prob_sum_NL = prob_NTL.mean(axis = (1))
    # obtain the optimal regime for each individual
    regime_opt_true = prob_NL_true.argmax(axis = 1)
    regime_opt = prob_sum_NL.argmax(axis = 1)
    
    return (regime_opt_true == regime_opt).mean()

# visualization
def plot3D_tensor(Y,
                  N, T, k, ax,
                  cmp = plt.get_cmap('bwr'),
                  title_main = None,
                  vmin = -400, vmax = 800):        
    # plot of potential outcomes
    x = range(1, (N+1)); y = range(1, (T+1)); z = range(1, (2**k+1))           
    points = []
    for element in itertools.product(x, y, z):
        points.append(element)
    xi, yi, zi = zip(*points)

    # select out exact zero entries (i.e. missing)
    xi_obs = [xi[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    yi_obs = [yi[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    zi_obs = [zi[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    Y_obsi = [Y.flatten()[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    
    # full
    p1 = ax.scatter3D(xi_obs, yi_obs, zi_obs, c = Y_obsi, 
               cmap = cmp,
               alpha = 0.5, s = 50,
               vmin = vmin, vmax = vmax)
    # fig.colorbar(p1, ax = ax, shrink = 0.5, aspect = 5)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # labels
    # ax.set_xlabel('subject')
    # ax.set_ylabel('time')
    # ax.set_zlabel('treatment regime')
    # ax.set_title(title_main)
    return p1