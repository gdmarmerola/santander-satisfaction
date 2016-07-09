# search space for hyperparameter optimization
xgb_space = {'model': xgb.XGBClassifier,
             'params': {'n_estimators' : hp.normal('xgb_n', 500, 100),
                        'learning_rate' : hp.uniform('xgb_eta', 0.01, 0.03),
                        'max_depth' : hp.quniform('xgb_max_depth', 2, 8, 1),
                        'min_child_weight' : hp.quniform('xgb_min_child_weight', 1, 6, 1),
                        'subsample' : hp.uniform('xgb_subsample', 0.8, 1),
                        'gamma' : hp.uniform('xgb_gamma', 0.0, 0.4),
                        'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.2, 0.8),
                        'objective': hp.choice('xgb_obj', ['binary:logistic']),
                        'scale_pos_weight': hp.uniform('xgb_w', 1.0, 4.0)
                        },
             'preproc': {'na_input': {'strategy': 'mean'},
                         'var_thres': {'threshold': 0.0},
                         'sel_perc': {'score_func': hp.choice('sel_sf', [f_classif, chi2]),
                                      'percentile': hp.quniform('sel_p', 50, 100, 5)}
                        },
             'resmpl': hp.choice('resmpl', [{'method': False, 'params': False},
                                            {'method': SMOTE,
                                             'params': {'ratio': hp.uniform('rat_s', 1, 24),
                                                        'verbose': False,
                                                        'kind': 'regular'}},
                                            {'method': NearMiss,
                                             'params': {'ratio': hp.uniform('rat_nm', 1, 24),
                                                        'verbose': False,
                                                        'version': hp.choice('k_n',[1, 2, 3])}},
                                                        ]),
             'data': hp.choice('dc',[{'real': 'data/engineered-real/train.csv',
                                      'cat': 'data/engineered-cat/train', 
                                      'ground-truth': 'data/target.csv'},
                                    {'real': 'data/selected/st-train.csv',
                                      'cat': None, 
                                      'ground-truth': 'data/target.csv'}]),                
             'feat_exp': {'n': 0}, #hp.quniform('exp_n', 0, 100, 20)
             'fit_params': {'eval_metric': 'auc'},
             'y_transf': hp.choice('trf', [None]),
            }
# over and undersampling did not improve results
# just testing stability selection
# search space for hyperparameter optimization
# search space for hyperparameter optimization
# search space for hyperparameter optimization
xgb_space = {'model': xgb.XGBClassifier,
             'params': {'n_estimators' : hp.normal('xgb_n', 500, 100),
                        'learning_rate' : hp.uniform('xgb_eta', 0.01, 0.03),
                        'max_depth' : hp.quniform('xgb_max_depth', 2, 8, 1),
                        'min_child_weight' : hp.quniform('xgb_min_child_weight', 1, 6, 1),
                        'subsample' : hp.uniform('xgb_subsample', 0.8, 1),
                        'gamma' : hp.uniform('xgb_gamma', 0.0, 0.4),
                        'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.2, 0.8),
                        'objective': hp.choice('xgb_obj', ['binary:logistic']),
                        'scale_pos_weight': hp.uniform('xgb_w', 1.0, 4.0)
                        },
             'preproc': {'na_input': {'strategy': 'mean'},
                         'var_thres': {'threshold': 0.0},
                         'sel_perc': {False}
                        },
             'resmpl': hp.choice('resmpl', [{'method': False, 'params': False}]),
             'data': hp.choice('dc',[{'real': 'data/selected/st-train.csv',
                                      'cat': None, 
                                      'ground-truth': 'data/target.csv'}]),                
             'feat_exp': {'n': 0}, #hp.quniform('exp_n', 0, 100, 20)
             'fit_params': {'eval_metric': 'auc'},
             'y_transf': hp.choice('trf', [None]),
            }
