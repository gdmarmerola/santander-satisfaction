''' Santander satisfaction base code: Guilherme Duarte Marmerola '''

from __future__ import division
import sys
import dill
import random
import pickle
import hyperopt
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from numbers import Number
from sklearn import metrics
from scipy.special import cbrt
import matplotlib.pyplot as plt 
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix, vstack, hstack
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand
from sklearn.decomposition import RandomizedPCA, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, chi2, VarianceThreshold, f_classif
from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, MultiTaskLasso, ARDRegression, RandomizedLogisticRegression
# from unbalanced_dataset import NearMiss, ClusterCentroids, OverSampler, SMOTE

####### miscellaneous functions #######

### getting plot ranges - useful for multiple plots ###
def get_plot_ranges(n_plots, n_features):
	
	n_batches, last_batch = divmod(n_features, n_plots)
	return [np.array(range(n_plots))+i*n_plots for i in range(n_batches)] + [np.array(range(last_batch))+n_batches*n_plots]

### saving and loading objects ###
def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        dill.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return dill.load(f)

### finding ocurrences in strings
def find_nth(haystack, needle, n):
	start = haystack.find(needle)
	while start >= 0 and n > 1:
		start = haystack.find(needle, start+len(needle))
		n -= 1
	if start == -1:
		return 999999
	else:
		return start

### hyperparameter summary
def hypersummary(trials_list, results, eval_metric):
	
	hyper_series = pd.DataFrame()
	for i in range(len(trials_list)):
		model_params = trials_list[i]['result']['parameters']['params']
		for key in model_params.keys():
			try:
				hyper_series[key][i] = model_params[key]
			except:
				hyper_series[key] = [None] * len(trials_list)
				hyper_series[key][i] = model_params[key]
									
		model_params = trials_list[i]['result']['parameters']['preproc']
		for key in model_params.keys():
			if key == 'sel_perc': 
				try:
					hyper_series[key][i] = model_params[key]
				except:
					hyper_series[key] = [None] * len(trials_list)
					hyper_series[key][i] = model_params[key]

		model_params = trials_list[i]['result']['parameters']['data']
		for key in model_params.keys():
			try:
				hyper_series[key][i] = model_params[key]
			except:
				hyper_series[key] = [None] * len(trials_list)
				hyper_series[key][i] = model_params[key]
	
	hyper_series[eval_metric] = results
	
	return hyper_series
	
### obtaining expected average from distribution of data ###
def expected_average(data, n_samples):
	
	avgs = []
	for i in range(200):
		avgs.append(np.mean(np.random.choice(data, size=n_samples, replace=False)))

	return sum(avgs)/200, np.std(avgs) 
	
	
### making stdout flush buffer ###
class Unbuffered(object):

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


### removing duplicated columns from data ###
class DuplicateRemove:
	
	def __init__(self):
		pass
	
	def fit_transform(self, data):
		
		# remove duplicated columns
		self.remove = []
		for i in range(data.shape[1]-1):
			vals = data[:,i]
			for j in range(i+1, data.shape[1]):
				if np.array_equal(vals, data[:,j]):
					self.remove.append(j)
					
		# also remove 0-variance features
		for i in range(data.shape[1]):
			if np.std(data[:,i]) == 0:
				self.remove.append(i)

		return np.delete(data, self.remove, axis=1)
		
	def transform(self, data):
		
		return np.delete(data, self.remove, axis=1)
		
class Resampling:
	
	def __init__(self, method, args):
		
		def dummy(X, y):
			return X, y
		
		if method == False:
			self.method = dummy
			self.transf = False
		else:
			self.method = method(**args).fit_transform
			self.transf = True
			
	def resample(self, X, y):
		if self.transf:
			try:
				X = X.todense()
			except:
				X = np.array(X)
		
		return self.method(X, np.array(y))
		

### transforming target variable - useful for regression problems
class TargetTransform:
	
	def __init__(self, transforms):
		
		if transforms == None:
			self.transf = lambda x: x
			self.inv = lambda x: x
		else:
			self.transf = transforms['transf']
			self.inv = transforms['inv']
	
	def transform(self, y):

		return self.transf(y)
		
	def inv_transform(self, y):

		return self.inv(y)

### tool for creating new features ###
class FeatureExpansion:
	
	def __init__(self):
		pass
		
	def fit_transform(self, data, n_features):
		
		#operations = ['square', 'product', 'sqrt', 'cbrt', 'log', 'exp', 'division']
		operations = ['division', 'product', 'sum', 'diff']
		features = {}
		for op in operations:
			features[op] = data.columns
		
		op_log = []
		for i in range(n_features):
		
			op = random.sample(operations, 1)[0]
			try:
				if op == 'log':
					# select one feature
					feat = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat+'-log'] = np.log(data[feat])
					# record operation
					op_log.append([op, feat])
					# pop dict entry 
					features[op].pop(feat)

				if op == 'exp':
					# select one feature
					feat = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat+'-exp'] = np.exp(data[feat])
					# record operation
					op_log.append([op, feat])
					# pop dict entry 
					features[op].pop(feat)
										
				if op == 'square':
					# select one feature
					feat = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat+'-square'] = np.square(data[feat])
					# record operation
					op_log.append([op, feat])
					# pop dict entry 
					features[op].pop(feat)
					
				if op == 'sqrt':
					feat = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat+'-sqrt'] = np.sqrt(data[feat])
					# record operation
					op_log.append([op, feat])
					# pop dict entry 
					features[op].pop(feat)
					
				if op == 'cbrt':
					feat = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat+'-cbrt'] = cbrt(data[feat])
					# record operation
					op_log.append([op, feat])
					# pop dict entry 
					features[op].pop(feat)

				if op == 'product':
					# select two features
					feat1 = random.sample(features[op], 1)[0]
					feat2 = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat1+'-'+feat2+'-prod'] = np.multiply(data[feat1], data[feat2])
					# record operation
					op_log.append([op, feat1, feat2])
					# pop dict entry 
					features[op].pop(feat1)
					features[op].pop(feat2)

				if op == 'sum':
					# select two features
					feat1 = random.sample(features[op], 1)[0]
					feat2 = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat1+'-'+feat2+'-sum'] = data[feat1] + data[feat2]
					# record operation
					op_log.append([op, feat1, feat2])
					# pop dict entry 
					features[op].pop(feat1)
					features[op].pop(feat2)

				if op == 'diff':
					# select two features
					feat1 = random.sample(features[op], 1)[0]
					feat2 = random.sample(features[op], 1)[0]
					# add column with transformed version of the feature
					data[feat1+'-'+feat2+'-diff'] = data[feat1] - data[feat2]
					# record operation
					op_log.append([op, feat1, feat2])
					# pop dict entry 
					features[op].pop(feat1)
					features[op].pop(feat2)
					
				if op == 'division':
					# select two features
					feat1 = random.sample(features[op], 1)[0]
					# second must be non-zero
					ok = False
					max_it = 10
					while (not ok) and (max_it > 0):
						feat2 = random.sample(features[op], 1)[0]
						if not any(data[feat] == 0):
							ok = True
						max_it = max_it - 1
					if ok:
						# add column with transformed version of the feature
						data[feat1+'-'+feat2+'-div'] = np.divide(data[feat1], data[feat2])
						# record operation
						op_log.append([op, feat1, feat2])
						# pop dict entry 
						features[op].pop(feat1)
						features[op].pop(feat2)
			except:
				pass
			
		return data, op_log
		
	def transform(self, data, op_log):
		
		for log in op_log:
		
			op = log[0]
			try:
				if op == 'log':
					# select one feature
					feat = log[1]
					# add column with transformed version of the feature
					data[feat+'-log'] = np.log(data[feat])
				
				if op == 'exp':
					# select one feature
					feat = log[1]
					# add column with transformed version of the feature
					data[feat+'-exp'] = np.log(data[feat])					
				
				if op == 'square':
					# select one feature
					feat = log[1]
					# add column with transformed version of the feature
					data[feat+'-square'] = np.square(data[feat])
					
				if op == 'sqrt':
					# select one feature
					feat = log[1]
					# add column with transformed version of the feature
					data[feat+'-sqrt'] = np.sqrt(data[feat])
				
				if op == 'cbrt':
					# select one feature
					feat = log[1]
					# add column with transformed version of the feature
					data[feat+'-cbrt'] = cbrt(data[feat])
					
				if op == 'product':
					# select two features
					feat1 = log[1]
					feat2 = log[2]
					# add column with transformed version of the feature
					data[feat1+'-'+feat2+'-prod'] = np.multiply(data[feat1], data[feat2])

				if op == 'sum':
					# select two features
					feat1 = log[1]
					feat2 = log[2]
					# add column with transformed version of the feature
					data[feat1+'-'+feat2+'-sum'] = data[feat1] + data[feat2]

				if op == 'diff':
					# select two features
					feat1 = log[1]
					feat2 = log[2]
					# add column with transformed version of the feature
					data[feat1+'-'+feat2+'-diff'] = data[feat1] - data[feat2]
					
				if op == 'division':
					feat1 = log[1]
					feat2 = log[2]
					# add column with transformed version of the feature
					data[feat1+'-'+feat2+'-div'] = np.divide(data[feat1], data[feat2])
			except:
				pass
		
		return data


class FeatureSelection:
	
	def __init__(self, estimator, params):
		
		self.estimator = estimator(**params)

	def fit_transform(self, X, y):
		
		return self.estimator.fit_transform(X, y)
		
	def transform(self, X):
		
		return self.estimator.transform(X)
		
		
### preprocessing wrapper for simple prototyping ###
class preprocessing:
	
	def __init__(self, params):
		
		self.params = params
	
	def fit_transform(self, data, y):
		
		functs = {'std_scale': StandardScaler, 
				  'minmax_scale': MinMaxScaler,
				  'na_input': Imputer,
				  'var_thres': VarianceThreshold,
				  'rm_dupl': DuplicateRemove,
				  'poly_feats': PolynomialFeatures,
				  'rand_pca': RandomizedPCA,
				  'svd': TruncatedSVD,
				  'sel_perc': SelectPercentile,
				  }
		
		self.fn_order = ['rm_dupl', 'na_input', 'var_thres', 'minmax_scale', 'std_scale', 'poly_feats', 'rand_pca', 'svd', 'sel_perc']

		self.fitted = {}
		if len(self.params.keys()) == 0:
			return data
		
		for method in self.fn_order:
			try:
				fn = functs[method](**self.params[method])
				
				try:
					data = fn.fit_transform(data)
				except:
					data = fn.fit_transform(data, y)
				
				self.fitted[method] = fn
			except:
				pass	
				
		return data
	
	def transform(self, data):
		
		for method in self.fn_order:
			try:
				fn = self.fitted[method]
				data = fn.transform(data)
			except KeyError:
				pass	
		
		return data
		
### wrapper for sklearn styled model presentation ###		
class sklearn_wrapper:
	
	def __init__(self, model, preproc_params, y_transform, resmpl={'method':False,'params':False}):

		self.model = model
		self.preproc = preprocessing(preproc_params)
		self.y_transform = y_transform
		self.resmpl = Resampling(resmpl['method'], resmpl['params'])
		
	def fit(self, X, y, fit_params={}):
		
		X = self.preproc.fit_transform(X, y)
		X, y = self.resmpl.resample(X, y)
		X = csr_matrix(X)
		y = self.y_transform.transform(y)
		self.model.fit(X, y, **fit_params)
		
	def predict(self, X):
		
		X = self.preproc.transform(X)
		preds = self.model.predict(X)
		return self.y_transform.inv_transform(preds)
		
	def predict_proba(self, X):
		
		X = self.preproc.transform(X)
		return self.model.predict_proba(X)

######## Validation environment ########


### simple and versatile CV environment ###
class cross_val:
	
	# initialize with cross_val type
	def __init__(self, val_type):

		self.val_type = val_type
	
	# get model, data, and run the CV
	def run_cv(self, model, X, y, n_reps=1, fit_params={}):
		
		sys.stdout.write('cv underway')
		
		self.count = 0
		self.preds = {}
		self.probs = {}
		self.gnd_truth = {}
		for rep in range(n_reps):
			for train_inds, test_inds in self.val_type:
				
				# splitting data between train and test
				X_train, y_train = X[train_inds,:], y[train_inds]
				X_test, y_test = X[test_inds,:], y[test_inds]
				
				# fitting model
				model.fit(X_train, y_train, fit_params=fit_params)
				
				# getting predictions (and probabilities when possible)
				self.preds[self.count] = model.predict(X_test)
				try:
					self.probs[self.count] = model.predict_proba(X_test)[:,1]
				except:
					pass
					
				# saving predictions to dict
				self.gnd_truth[self.count] = y_test
				self.count += 1
				sys.stdout.write('.')
			
	def evaluate_cv(self, metric, eval_type, drop_extremes=False):
		
		results = {}
		results['folds'] = {}
		results['folds']['preds']     = {}
		results['folds']['probs']     = {}
		results['folds']['gnd_truth'] = {}
		results['folds']['result']    = {}

		for fold in range(self.count):
			
			# calculating results through metric 
			# decides if evaluation should be done 
			# based on predictions or probabilities	
			if eval_type == 'preds':
				pred_eval = metric(self.gnd_truth[fold], self.preds[fold])				
				results['folds']['gnd_truth'][fold] = self.gnd_truth[fold]
				results['folds']['preds'][fold] = self.preds[fold]
				results['folds']['result'][fold] = pred_eval
										  	   	
			elif eval_type == 'probs':
				prob_eval = metric(self.gnd_truth[fold], self.probs[fold])
				results['folds']['gnd_truth'][fold] = self.gnd_truth[fold]
				results['folds']['probs'][fold] = self.probs[fold]
				results['folds']['result'][fold] = prob_eval
					   	   	
			else:
				print 'error in CV: please choose eval_type ->' \
				      '(probs or preds)'
				return 1
		
		# calcuting average result over folds
		if drop_extremes:
			vals = results['folds']['result'].values()
			best = vals.index(max(vals))
			worst = vals.index(min(vals))
			vals = [v for i, v in enumerate(vals) if i not in [best, worst]]
			results['avg'] = np.mean(vals)
			results['std'] = np.std(vals)
		else:
			results['avg'] = np.mean(results['folds']['result'].values())
			results['std'] = np.std(results['folds']['result'].values())
		
		return results


### optimization function for framework ###
def optimize(experimental_fmwk, space, max_evals, trials):

    global eval_number
    eval_number = 0

    fmin(experimental_fmwk, space, algo=tpe.suggest,
         trials=trials, max_evals=max_evals)

    return get_best(trials)


### hyperopt-friendly model selection framework ###		
def framework(space):

	# visual feedback
	global eval_number
	eval_number += 1
	print 'eval_number:', eval_number, space
	
	# reading csv from search space path
	X_real = pd.read_csv(space['data']['real'])
	y_train = pd.read_csv(space['data']['ground-truth'])['TARGET']

	# feature expansion
	feat_exp = FeatureExpansion()
	X_real, op_log = feat_exp.fit_transform(X_real, int(space['feat_exp']['n']))
	X_real = csr_matrix(X_real)
	space['feat_exp']['op_log'] = op_log

	# categorical data
	try:
		X_cat = load_obj(space['data']['cat'])
		X_train = hstack([X_real, X_cat]).tocsr()
	except:
		X_train = X_real.tocsr()

	# casting values to int
	space['params']['n_estimators'] = int(space['params']['n_estimators'])
	space['preproc']['sel_perc']['percentile'] = int(space['preproc']['sel_perc']['percentile'])
	
	# model from search space
	algo = space['model'](**space['params'])
	y_transform = TargetTransform(space['y_transf'])
	model = sklearn_wrapper(algo, space['preproc'], 
							y_transform, resmpl=space['resmpl'])

	# repeat validation some times
	auc_avgs = []
	auc_stds = []
	rounds_results = {}
	for rep in range(5):
		
		# cross validation parameters:
		# 7-fold dropping best and worst and repeating
		CV_params = {'y': y_train,
					 'n_folds': 7,
					 'shuffle': True,
					 'random_state': rep}

		# saving CV so that all trials have same test/train splits
		try:
			val_type = load_obj('data/CV/7-fold-cv-{}'.format(rep))
		except:
			val_type = StratifiedKFold(**CV_params)
			save_obj(val_type, 'data/CV/7-fold-cv-{}'.format(rep))

		# cross validation environment
		CV_env = cross_val(val_type)

		# running CV
		CV_env.run_cv(model, X_train, y_train, fit_params=space['fit_params'])

		# evaluating R-squared
		auc = CV_env.evaluate_cv(metrics.roc_auc_score, 'probs', drop_extremes=True)  
		
		# registering auc
		auc_avgs.append(auc['avg'])
		auc_stds.append(auc['std'])
		reps_avg = np.sum(auc_avgs)/(rep+1)
		reps_std = np.sum(auc_stds)/(rep+1)
		rounds_results[rep] = auc
		
		print 'round {0} AUC: {1:.4f} | average so far: {2:.4f} +- {3:.4f}'.format(rep, auc['avg'], reps_avg, reps_std) 	
	
	return {'loss': 1 - reps_avg,
			'results': rounds_results,
			'auc_avg': reps_avg,
			'auc_std': reps_std,
			'status': STATUS_OK,
			'parameters': space}

### getting best model (sorting trials) ###
def get_best(trials, ind=0):
    best_ind = trials.losses().index(sorted(trials.losses())[ind])
    return trials.trials[best_ind]
