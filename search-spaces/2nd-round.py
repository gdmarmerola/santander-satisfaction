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
             'data': {'real': hp.choice('dr', ['data/engineered-real/train.csv',
                                               'data/no-duplicates/train.csv']),
                      'cat': hp.choice('dc', ['data/categorical/train']),
                      'ground-truth': hp.choice('gt', ['data/target.csv'])
                     },
             'feat_exp': {'n': 0}, #hp.quniform('exp_n', 0, 100, 20)
             'fit_params': {'eval_metric': 'auc'},
             'y_transf': hp.choice('trf', [None]),
            }

# model selection
eval_number = 0
trials = Trials()      
best_xgb = optimize(framework, xgb_space, 120, trials)

# saving trials
save_obj(trials, 'trials/2nd-round')
