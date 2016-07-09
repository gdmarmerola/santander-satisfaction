# search space for hyperparameter optimization
xgb_space = {'model': xgb.XGBClassifier,
             'params': {'n_estimators' : hp.quniform('xgb_n', 1000, 5000, 500),
                        'learning_rate' : hp.normal('xgb_eta', 0.05, 0.01),
                        'max_depth' : hp.quniform('xgb_max_depth', 2, 8, 1),
                        'min_child_weight' : hp.quniform('xgb_min_child_weight', 1, 6, 1),
                        'subsample' : hp.uniform('xgb_subsample', 0.2, 1),
                        'gamma' : hp.uniform('xgb_gamma', 0.01, 0.4),
                        'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.2, 1),
                        'objective': hp.choice('xgb_obj', ['binary:logistic']),
                        'scale_pos_weight': hp.uniform('xgb_w', 1.0, 5.0)
                        },
             'preproc': {'na_input': {'strategy': 'mean'},
                         'var_thres': {'threshold': hp.uniform('var_t', 0.00, 0.01)},
                         'std_scale': hp.choice('std_s', [{False}]),
                         'sel_perc': {'score_func': hp.choice('sel_sf', [f_classif, chi2]),
                                      'percentile': hp.quniform('sel_p', 5, 100, 5)}
                        },
             'data': {'real': hp.choice('dr', ['data/engineered-real/train.csv',
                                               'data/no-duplicates/train.csv']),
                      'cat': hp.choice('dc', ['data/engineered-cat/train',
                                              'data/categorical/train',
                                              None]),
                      'ground-truth': hp.choice('gt', ['data/target.csv'])
                     },
             'feat_exp': {'n': hp.quniform('exp_n', 0, 100, 20)},
             'fit_params': {'eval_metric': 'auc'},
             'y_transf': hp.choice('trf', [None]),
            }

# model selection
eval_number = 0
trials = Trials()      
best_xgb = optimize(framework, xgb_space, 120, trials)

# saving trials
save_obj(trials, 'trials/1st-round')
