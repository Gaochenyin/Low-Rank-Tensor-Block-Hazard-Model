# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:36:34 2023

@author: hp
"""
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
from scipy.special import expit
from utils import theta2prob, gen_data_potential_Y_binary, \
    TensorCompletionCovariateBinary, \
    get_theta_binary, get_theta_survival, \
        regime_eval, regime_prec
import scipy
import os
import time
def main(seed, N, T, k):
    Y, A, delta, X, theta_true, error = gen_data_potential_Y_binary(seed = seed, N = N, T = T, k = k, d = d)
    
    # weight calibration for generalized propensity score estimation for each treatment
    def weight_contd_grad(coef,
                          XX_t, A_t):
        
        N = XX_t.shape[0]
        X_cov = np.hstack((np.ones((N, 1)), XX_t))
        # compute the weight
        linear_pred = np.dot(X_cov, coef)
        # obj_grad_CAL = (A_t/expit(linear_pred) -1).dot(XX_t)/N
        obj_grad_CAL = (A_t / expit(linear_pred) -1).dot(X_cov)/N
        # obj_grad_CAL = (expit(linear_pred) - A_t).dot(XX_t)/N
        return obj_grad_CAL
    def rho_trt(kn):
        weights_coef1 = scipy.optimize.root(fun = partial(weight_contd_grad,
                                          XX_t = X, A_t = A[:, kn]),
                            x0 = np.zeros(d+1),
                            # jac = partial(weight_contd_hess,
                            #               XX_t = CV_X,
                            #               A_t = CV_A),
                            method = 'linearmixing').x
        weights_coef0 = scipy.optimize.root(fun = partial(weight_contd_grad,
                                          XX_t = X, A_t = 1 - A[:, kn]),
                            x0 = np.zeros(d+1),
                            # jac = partial(weight_contd_hess,
                            #               XX_t = CV_X,
                            #               A_t = CV_A),
                            method = 'linearmixing').x
        # compute the estimated propensity weights
        rho = expit(X @ weights_coef1[1:] + weights_coef1[0]) * A[:, kn] +\
            expit(X @ weights_coef0[1:] + weights_coef0[0]) * (1-A[:, kn]) 
        return rho
    
    # focus on past k treatments
    PS_trt = np.array([rho_trt(kn) for kn in range(k)]) 
    rho = np.apply_along_axis(np.prod, 0, PS_trt)
    # replace Nan with zero for computational purpose
    Y[np.isnan(Y)]= 0
    
    ## OUR MODEL    
    # weighted tensor block hazard model
    start_time = time.time()
    TC_bry_w = TensorCompletionCovariateBinary(Y = Y, A = A, X = X, delta = delta,
                                               rho = rho,
                                            stepsize = 1e-6, momentum = 0.0,
                                            niters = 10000, tol = 1e-8,
                                            r1_list = [4], r2_list = [2], r3_list = [k + 1],
                                            lam_list = [0], method = 'discrete')
    theta_hat_w = TC_bry_w.SequentialTuning()
    time_w  = time.time() - start_time
      
    ## binary classification
    # logistic regression
    start_time = time.time()
    theta_hat_GLM = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'logit')
    time_GLM = time.time() - start_time
    # new added method    
    start_time = time.time()
    theta_hat_SVM = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'SVM')
    time_SVM = time.time() - start_time
    
    
    start_time = time.time()
    theta_hat_gdClassifer = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'GDboost')
    time_GDboost = time.time() - start_time
    
    start_time = time.time()
    theta_hat_adaB = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'AdaBoost')
    time_AdaBoost = time.time() - start_time
    
    # random forest
    start_time = time.time()
    theta_hat_rf = get_theta_binary(Y = Y, A = A, X = X, delta = delta, 
                                    method = 'RandomForest')
    time_rf = time.time() - start_time
    # nerual network 
    start_time = time.time()
    theta_hat_nn = get_theta_binary(Y = Y, A = A, X = X, delta = delta, 
                                    method = 'NeuralNetwork')
    time_nn = time.time() - start_time
    
    
    # a voting classifier is a combination of above four estimators
    start_time = time.time()
    theta_hat_vote = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'Vote') 
    time_vote = time.time() - start_time
    
    np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_vote))/\
        np.linalg.norm(theta2prob(theta_true))
    
    ## survival analysis
    # Cox PH model
    start_time = time.time()
    pred_hat_coxph = get_theta_survival(Y, A, X, delta, method = 'coxPH')
    time_cox = time.time() - start_time
    # survival random forest
    start_time = time.time()
    pred_hat_rf = get_theta_survival(Y, A, X, delta, method = 'RandomForest')
    time_coxrf = time.time() - start_time
    # Gradient descent boosting
    start_time = time.time()
    pred_hat_gdb = get_theta_survival(Y, A, X, delta, method = 'GDboost')
    time_coxgdb = time.time() - start_time
    # model-based boosting
    start_time = time.time()
    pred_hat_mb = get_theta_survival(Y, A, X, delta, method = 'mboost')
    time_coxmb = time.time() - start_time
    
    ## estimation time
    time_series = pd.Series({'GLM': time_GLM, 
                           'SVM': time_SVM,
                           'GDboost': time_GDboost,
                           'AdaBoost': time_AdaBoost,
                           'rf': time_rf,
                           'nn': time_nn,
                           'vote': time_vote,
                           'coxph': time_cox,
                           'coxrf': time_coxrf,
                           'coxgdb': time_coxgdb,
                           'coxmb': time_coxmb,
                           'Our': time_w})    
    
    ## estimation error of survival probability
    loss_prob = pd.Series({'GLM': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_GLM))/np.linalg.norm(theta2prob(theta_true)),
                           'SVM': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_SVM))/np.linalg.norm(theta2prob(theta_true)),
                           'GDboost': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_gdClassifer))/np.linalg.norm(theta2prob(theta_true)),
                           'AdaBoost': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_adaB))/np.linalg.norm(theta2prob(theta_true)),
                           'rf': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_rf))/np.linalg.norm(theta2prob(theta_true)),
                           'nn': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_nn))/np.linalg.norm(theta2prob(theta_true)),
                           'vote': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_vote))/np.linalg.norm(theta2prob(theta_true)),
                           'coxph': np.linalg.norm(theta2prob(theta_true) - pred_hat_coxph)/np.linalg.norm(theta2prob(theta_true)),
                           'coxrf': np.linalg.norm(theta2prob(theta_true) - pred_hat_rf)/np.linalg.norm(theta2prob(theta_true)),
                           'coxgdb': np.linalg.norm(theta2prob(theta_true) - pred_hat_gdb)/np.linalg.norm(theta2prob(theta_true)),
                           'coxmb': np.linalg.norm(theta2prob(theta_true) - pred_hat_mb)/np.linalg.norm(theta2prob(theta_true)),
                           'TC_w': np.linalg.norm(theta2prob(theta_true) - theta2prob(theta_hat_w))/np.linalg.norm(theta2prob(theta_true))})

    ## cumulative regret in survival probability over time
    optimal_out = pd.Series({'GLM': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                  theta_true = theta_true,
                                  theta_hat = theta_hat_GLM),
                             'SVM': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                           theta_true = theta_true,
                                                           theta_hat = theta_hat_SVM),
                             'GDboost': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                           theta_true = theta_true,
                                                           theta_hat = theta_hat_gdClassifer),
                             'AdaBoost': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                           theta_true = theta_true,
                                                           theta_hat = theta_hat_adaB),
                             'rf': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                               theta_true = theta_true,
                                               theta_hat = theta_hat_rf),
                             'nn': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                               theta_true = theta_true,
                                               theta_hat = theta_hat_nn),
                             'vote': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                 theta_true = theta_true,
                                                 theta_hat = theta_hat_vote),
                             'coxph': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                  theta_true = theta_true,
                                                  prob_NTL = pred_hat_coxph),
                             'coxrf': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                  theta_true = theta_true,
                                                  prob_NTL = pred_hat_rf),
                             'coxgdb': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                   theta_true = theta_true,
                                                   prob_NTL = pred_hat_gdb),
                             'coxmb': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                  theta_true = theta_true,
                                                  prob_NTL = pred_hat_mb),
                             'TC_w': regime_eval(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                 theta_true = theta_true,
                                                 theta_hat = theta_hat_w)})
    ## decision accuracy 
    prec_out = pd.Series({'GLM': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                                  theta_true = theta_true,
                                  theta_hat = theta_hat_GLM),
                          'SVM': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                        theta_true = theta_true,
                                                        theta_hat = theta_hat_SVM),
                          'GDboost': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                        theta_true = theta_true,
                                                        theta_hat = theta_hat_gdClassifer),
                          'AdaBoost': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                        theta_true = theta_true,
                                                        theta_hat = theta_hat_adaB),
               'rf': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                           theta_true = theta_true,
                           theta_hat = theta_hat_rf),
               'nn': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                           theta_true = theta_true,
                           theta_hat = theta_hat_nn),
               'vote': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                                                        theta_true = theta_true,
                                                        theta_hat = theta_hat_vote),
               'coxph': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                           theta_true = theta_true,
                           prob_NTL = pred_hat_coxph),
               'coxrf': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                           theta_true = theta_true,
                           prob_NTL = pred_hat_rf),
               'coxgdb': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                           theta_true = theta_true,
                           prob_NTL = pred_hat_gdb),
               'coxmb': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                           theta_true = theta_true,
                           prob_NTL = pred_hat_mb),
               'TC_w': regime_prec(Y = Y, A = A, rho = rho, X = X, d = d, 
                           theta_true = theta_true,
                           theta_hat = theta_hat_w)})
    
   

    # save the estimated tensor
    np.save(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'theta_hat_w_N{N}_T{T}_k{k}_seed{seed}.csv'),
            theta_hat_w)
    np.save(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'theta_true_N{N}_T{T}_k{k}_seed{seed}.csv'),
            theta_true)
    # save the result
    np.savetxt(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'loss_N{N}_T{T}_k{k}_niters{niters}_seed{seed}.csv'),
               loss,
               delimiter = ',')

    np.savetxt(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'loss_prob_N{N}_T{T}_k{k}_niters{niters}_seed{seed}.csv'),
               loss_prob,
               delimiter = ',')

    np.savetxt(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'optimal_out_N{N}_T{T}_k{k}_niters{niters}_seed{seed}.csv'),
                   optimal_out,
                   delimiter = ',')
    
    np.savetxt(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'prec_out_N{N}_T{T}_k{k}_niters{niters}_seed{seed}.csv'),
                   prec_out,
                   delimiter = ',')
    np.savetxt(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'running_time_N{N}_T{T}_k{k}_niters{niters}_seed{seed}.csv'),
                   time_series,
                   delimiter = ',')
    
    np.savetxt(os.path.join('/gpfs_common/share01/statistics/cgao6/tensor_binary', f'pfactor_N{N}_T{T}_k{k}_niters{niters}_seed{seed}.csv'),
               TC_bry_w.U_3,
               delimiter = ',')
    # return loss



for b in range(B):
    np.savetxt(f'prob_binary\prob_truth_k{k}_b{b}.csv', np.apply_along_axis(np.cumprod, 1, expit(theta_true[:,:,b])),
               delimiter = ',')
    
    # np.savetxt(f'prob_binary\prob_TC_k{k}_b{b}.csv', np.apply_along_axis(np.cumprod, 1, expit(theta_hat[:,:,b])),
    #            delimiter = ',')
    
    np.savetxt(f'prob_binary\prob_TC_w_k{k}_b{b}.csv', np.apply_along_axis(np.cumprod, 1, expit(theta_hat_w[:,:,b])),
               delimiter = ',')
    
    np.savetxt(f'prob_binary\prob_GLM_k{k}_b{b}.csv', np.apply_along_axis(np.cumprod, 1, expit(theta_hat_GLM[:,:,b])),
               delimiter = ',')
    np.savetxt(f'prob_binary\pfactor_TC_w_k{k}.csv', TC_bry_w.U_3,
               delimiter = ',')
    # np.savetxt(f'prob_binary\pfactor_TC_k{k}.csv', TC_bry.U_3,
    #            delimiter = ',')

# res_df = pd.read_csv('results_pre.csv')
with multiprocessing.Pool() as p:
    main_partial = partial(main, N = 100, T = 5, B = B)
    p.map(main_partial, range(100))

with multiprocessing.Pool() as p:
    main_partial = partial(main, N = 100, T = 5, B = B)
    p.map(main_partial, range(100))
    
with multiprocessing.Pool() as p:
    main_partial = partial(main, N = 100, T = 5, B = B)
    p.map(main_partial, range(100))

with multiprocessing.Pool() as p:
    main_partial = partial(main, N = 100, T = 5, B = B)
    p.map(main_partial, range(100))


if __name__ == '__main__':

    
    N = 500; T = 10; k = 3; d = 3; seed = 1
    res_N100_T10 = pd.concat((main(seed, N = 100, T = 10, B = B) for seed in range(100)), axis=1)
    res_N300_T10 = pd.concat((main(seed, N = 300, T = 10, B = B) for seed in range(100)), axis=1)
    res_N500_T10 = pd.concat((main(seed, N = 500, T = 10, B = B) for seed in range(100)), axis=1)
    res_N1000_T10 = pd.concat((main(seed, N = 1000, T = 10, B = B) for seed in range(100)), axis=1)
    
    res_N100_T10 = res_N100_T10.T; res_N100_T10['N'] = 100; res_N100_T10['T'] = 10
    res_N300_T10 = res_N300_T10.T; res_N300_T10['N'] = 300; res_N300_T10['T'] = 10
    res_N500_T10 = res_N500_T10.T; res_N500_T10['N'] = 500; res_N500_T10['T'] = 10
    res_N1000_T10 = res_N1000_T10.T; res_N1000_T10['N'] = 1000; res_N1000_T10['T'] = 10
    
    res_df = pd.concat([res_N100_T10, res_N300_T10, res_N500_T10, res_N1000_T10])
    res_df.to_csv('results_pre_k5.csv', index=False)
    
    truth_mat = np.zeros((T, B))
    bias_TC_mat = np.zeros((T, B))
    bias_GLM_mat = np.zeros((T, B))
    for b in range(B):
        # truth
        truth_mat[:, b] = np.apply_along_axis(np.cumprod, 1, expit(theta_true[:,:,b])).\
             mean(axis = 0)
        # bias
        bias_TC_mat[:, b] = np.abs(np.apply_along_axis(np.cumprod, 1, expit(theta_true[:,:,b])).\
             mean(axis = 0)- \
        np.apply_along_axis(np.cumprod, 1, expit(theta_hat[:,:,b])).\
             mean(axis = 0))
            
        bias_GLM_mat[:, b] = np.abs(np.apply_along_axis(np.cumprod, 1, expit(theta_true[:,:,b])).\
             mean(axis = 0)- \
        np.apply_along_axis(np.cumprod, 1, expit(theta_hat_GLM[:,:,b])).\
             mean(axis = 0))
    
    np.savetxt(f"truth_k{k}.csv", truth_mat, delimiter=",")
    np.savetxt(f"bias_TC_k{k}.csv", bias_TC_mat, delimiter=",")
    np.savetxt(f"bias_GLM_k{k}.csv", bias_GLM_mat, delimiter=",")

    
