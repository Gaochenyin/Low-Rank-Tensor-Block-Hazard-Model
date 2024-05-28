# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:36:34 2023

@author: hp
"""
from functools import partial
import numpy as np
import pandas as pd
from scipy.special import expit
from utils import theta2prob, gen_data_potential_Y_binary, \
    TensorCompletionCovariateBinary, \
    get_theta_binary, get_theta_survival, \
        regime_eval, regime_prec
import scipy
def main(seed, N, T, k, d = 3):
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
    TC_bry_w = TensorCompletionCovariateBinary(Y = Y, A = A, X = X, delta = delta,
                                               rho = rho,
                                               stepsize = 1e-5,     
                                               niters = 10000,
                                               r1_list = [4], r2_list = [2], r3_list = [k + 1])
    theta_hat_w = TC_bry_w.SequentialTuning()
      
    ## binary classification
    # logistic regression
    theta_hat_GLM = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'logit')
    # new added method    
    theta_hat_SVM = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'SVM')    
    theta_hat_gdClassifer = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'GDboost')
    theta_hat_adaB = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'AdaBoost')
    
    # random forest
    theta_hat_rf = get_theta_binary(Y = Y, A = A, X = X, delta = delta, 
                                    method = 'RandomForest')
    # nerual network 
    theta_hat_nn = get_theta_binary(Y = Y, A = A, X = X, delta = delta, 
                                    method = 'NeuralNetwork')
    
    
    # a voting classifier is a combination of above four estimators
    theta_hat_vote = get_theta_binary(Y = Y, A = A, X = X, delta = delta,
                                     method = 'Vote') 
    
    ## survival analysis
    # Cox PH model
    pred_hat_coxph = get_theta_survival(Y, A, X, delta, method = 'coxPH')
    # survival random forest
    pred_hat_rf = get_theta_survival(Y, A, X, delta, method = 'RandomForest')
    # Gradient descent boosting
    pred_hat_gdb = get_theta_survival(Y, A, X, delta, method = 'GDboost')
    # model-based boosting
    pred_hat_mb = get_theta_survival(Y, A, X, delta, method = 'mboost')
    
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
    
   

    return pd.DataFrame({'loss':loss_prob, 'regret': optimal_out, 'precision': prec_out})




if __name__ == '__main__':

    # an example: sample size N = 500, time T = 10 and short-term treatment history k = 3
    N = 500; T = 10; k = 3; d = 3; seed = 2
    df = main(seed, N = N, T = T, k = k, d = d)

    
