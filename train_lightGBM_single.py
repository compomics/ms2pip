import os
import sys
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import random
import operator
import pickle as pickle
import matplotlib.pyplot as plt

def load_data(vector_filename, ion_type):
	# Read file
	if vector_filename.split('.')[-1] == 'pkl':
		vectors = pd.read_pickle(vector_filename)
	elif vector_filename.split('.')[-1] == 'h5':
		#vectors = pd.read_hdf(vector_filename, key='table', stop=1000)
		vectors = pd.read_hdf(vector_filename, key='table')
	else:
		print("Unsuported feature vector format")
		exit(1)
	print("{} contains {} feature vectors".format(args.vectors, len(vectors)))

	# Extract targets for given ion type
	target_names = list(vectors.columns[vectors.columns.str.contains('targets')])
	if not 'targets_{}'.format(ion_type) in target_names:
		print("Targets for {} could not be found in vector file.".format(ion_type))
		print("Vector file only contains these targets: {}".format(target_names))
		exit(1)

	targets = vectors.pop('targets_{}'.format(ion_type))
	target_names.remove('targets_{}'.format(ion_type))
	for n in target_names:
		vectors.pop(n)

	# Get psmids
	psmids = vectors.pop('psmid')

	return(vectors, targets, psmids)


sys.stderr.write('loading data\n')

parser = argparse.ArgumentParser(description='XGBoost training')
parser.add_argument('vectors', metavar='<_vectors.pkl>',
					help='feature vector file')
parser.add_argument('type', metavar='<type>',
					help='model type')
parser.add_argument('-c', metavar='INT', action="store", dest='num_cpu', default=24,
					help='number of CPUs to use')
parser.add_argument('-t', metavar='INT', action="store", dest='num_trees', default=30,
					help='number of trees in XGBoost model')
parser.add_argument('-e', metavar='FILE', action="store", dest='vectorseval',
					help='additional evaluation file')
parser.add_argument("-p", action="store_true", dest='make_plots', default=False,
					help="output plots")
parser.add_argument("-g", action="store_true", dest='gridsearch', default=False,
					help="perform Grid Search CV to select best parameters")
args = parser.parse_args()

np.random.seed(1)

filename = "{}_{}_{}".format(args.vectors.split('.')[-2], args.num_trees, args.type)
print("Using output filename {}".format(filename))

print("Loading train and test data...")
vectors, targets, psmids = load_data(args.vectors, args.type)

print("Splitting up into train and test set...")
upeps = psmids.unique()
np.random.shuffle(upeps)
test_psms = upeps[:int(len(upeps) * 0.3)]

train_vectors = vectors[~psmids.isin(test_psms)]
train_targets = targets[~psmids.isin(test_psms)]
train_psmids = psmids[~psmids.isin(test_psms)]

test_vectors = vectors[psmids.isin(test_psms)]
test_targets = targets[psmids.isin(test_psms)]
test_psmids = psmids[psmids.isin(test_psms)]

# Rename features to understand decision tree dump
#train_vectors.columns = ['Feature' + str(i) for i in range(len(train_vectors.columns))]
#test_vectors.columns = ['Feature' + str(i) for i in range(len(test_vectors.columns))]

print(train_vectors.shape)
print("Creating LightGBM datastructures...")
data = lgb.Dataset(train_vectors, label=train_targets)
datatest = lgb.Dataset(test_vectors, label=test_targets)
sys.stderr.write('loading data done\n')

params = {}
params['objective'] = 'regression'
params['metric'] = 'l1'
params['learning_rate'] = 0.8
#params['sub_feature'] = 1
params['num_leaves'] = 10
#params['min_data'] = 50
params['max_depth'] = 3

num_round = 50
#lgb.cv(param, data, num_round, nfold=5)
bst = lgb.train(params, data, num_round, valid_sets=[datatest])
		
#bst.save_model('model.txt')
print(bst.feature_importance())
model_json = bst.dump_model()
print(model_json["tree_info"])

def parseOneTree(root, index, array_type='double', return_type='double'):
	def ifElse(node):
		if 'leaf_index' in node:
			return 'return ' + str(node['leaf_value']) + ';'
		else:
			condition = 'arr[' + str(node['split_feature']) + ']'
			if node['decision_type'] == '<=':
				condition += ' <= ' + str(node['threshold'])
			else:
				condition += ' == ' + str(node['threshold'])
			left = ifElse(node['left_child'])
			right = ifElse(node['right_child'])
			return 'if ( ' + condition + ' ) { ' + left + ' } else { ' + right + ' }'
	return return_type + ' predictTree' + str(index) + '(' + array_type + ' arr) { ' + ifElse(root) + ' }'

def parseAllTrees(trees, array_type='unsigend int*', return_type='float'):
	return '\n\n'.join([parseOneTree(tree['tree_structure'], idx, array_type, return_type) for idx, tree in enumerate(trees)]) \
		+ '\n\n' + return_type + ' score_Y(' + array_type + ' arr) { ' \
		+ 'return ' + ' + '.join(['predictTree' + str(i) + '(arr)' for i in range(len(trees))]) + ';' \
		+ '}'

with open('if.else', 'w+') as f:
	f.write(parseAllTrees(model_json["tree_info"]))

