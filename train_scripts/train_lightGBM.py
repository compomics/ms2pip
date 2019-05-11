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

np.random.seed(1)

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

	# Extract targets for given ion type
	target_names = list(vectors.columns[vectors.columns.str.contains('targets')])
	if not 'targets{}'.format(ion_type) in target_names:
		print("Targets for {} could not be found in vector file.".format(ion_type))
		print("Vector file only contains these targets: {}".format(target_names))
		exit(1)

	targets = vectors.pop('targets{}'.format(ion_type))
	target_names.remove('targets{}'.format(ion_type))
	for n in target_names:
		vectors.pop(n)

	# Get psmids
	psmids = vectors.pop('psmid')

	return(vectors, targets, psmids)

fragtype = "y"
nul_cpu = 24

print("loading train data")

vectors, targets, psmids = load_data(sys.argv[1], fragtype)

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

print("Creating LightGBM datastructures...")
data = lgb.Dataset(train_vectors, label=train_targets)
datatest = lgb.Dataset(test_vectors, label=test_targets)

valid_sets = [datatest]
vector_sets = [test_vectors]
target_sets = [test_targets]
psmid_sets = [test_psmids]

print("loading evaluation data")
for fn in sys.argv[2:]:
	vectors, targets, psmids = load_data(fn, fragtype)
	tmp = lgb.Dataset(vectors, label=targets)
	valid_sets.append(tmp)
	psmid_sets.append(psmids)
	vector_sets.append(vectors)
	target_sets.append(targets)
	
sys.stderr.write('loading data done\n')

tmp2 = pd.DataFrame()
tmp3 = pd.DataFrame()
tmp3["psmid"] = test_psmids[test_vectors["charge"]==3]
tmp3["target"] = test_targets[test_vectors["charge"]==3]
tmp4 = pd.DataFrame()
tmp4["psmid"] = test_psmids[test_vectors["charge"]==4]
tmp4["target"] = test_targets[test_vectors["charge"]==4]
for max_depth in [7,9,11]:
	for num_leaves in [50,100,200]:
		params = {}
		params['objective'] = 'regression'
		params['metric'] = 'l1'
		params['learning_rate'] = 0.8
		#params['sub_feature'] = 1
		params['num_leaves'] = num_leaves
		#params['min_data'] = 50
		params['max_depth'] = max_depth
		
		num_round = 100
		#lgb.cv(param, data, num_round, nfold=5)
		bst = lgb.train(params, data, num_round, valid_sets=valid_sets)
		
		for c in [2,3,4]:
			for i in range(len(valid_sets)):
				tmp = pd.DataFrame()
				tmp["psmid"] = psmid_sets[i][vector_sets[i]["charge"]==c]
				tmp["target"] = target_sets[i][vector_sets[i]["charge"]==c]
				tmp["prediction"] = bst.predict(vector_sets[i][vector_sets[i]["charge"]==c])		
				tmpp = tmp.groupby('psmid')[['target','prediction']].corr().iloc[0::2,-1]
				print(">>%i %i %i %i %s"%(c,i,max_depth,num_leaves," ".join([str(x) for x in np.nanpercentile(tmpp.values,[10,30,50,70,90])])))

ddd
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
			if node['decision_type'] == 'no_greater':
				condition += ' <= ' + str(node['threshold'])
			else:
				condition += ' == ' + str(node['threshold'])
			left = ifElse(node['left_child'])
			right = ifElse(node['right_child'])
			return 'if ( ' + condition + ' ) { ' + left + ' } else { ' + right + ' }'
	return return_type + ' predictTree' + str(index) + '(' + array_type + '[] arr) { ' + ifElse(root) + ' }'

def parseAllTrees(trees, array_type='double', return_type='double'):
	return '\n\n'.join([parseOneTree(tree['tree_structure'], idx, array_type, return_type) for idx, tree in enumerate(trees)]) \
		+ '\n\n' + return_type + ' predict(' + array_type + '[] arr) { ' \
		+ 'return ' + ' + '.join(['predictTree' + str(i) + '(arr)' for i in range(len(trees))]) + ';' \
		+ '}'

with open('if.else', 'w+') as f:
	f.write(parseAllTrees(model_json["tree_info"]))

