import os
import sys
import argparse
import math
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import random
import operator
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import ms2pipfeatures_pyx

fair_constant = 0.7
def fair_obj(preds, dtrain):
	labels = dtrain.get_label()
	x = (preds - labels)
	den = abs(x) + fair_constant
	grad = fair_constant * x / (den)
	hess = fair_constant * fair_constant / (den * den)
	return grad, hess

def main():

	parser = argparse.ArgumentParser(description='XGBoost training')
	parser.add_argument('vectors',metavar='<_vectors.pkl or .h5>',
	         help='feature vector file')
	parser.add_argument('type',metavar='<type>',
	         help='model type')
	parser.add_argument('-c',metavar='INT', action="store", dest='num_cpu', default=23,
	         help='number of cpu\'s to use')
	args = parser.parse_args()

	sys.stderr.write('loading data\n')

	if args.vectors.split('.')[-1] == 'pkl':
	  vectors = pd.read_pickle(args.vectors)
	elif args.vectors.split('.')[-1] == 'h5':
	  vectors = pd.read_hdf(args.vectors, 'table')
	else:
	  print "unsuported feature vector format"
	vectors.dropna(inplace=True)
	print len(vectors)
	print vectors.head()
	#print vectors['charge'].value_counts()
	#tmp = vectors.pop("charge")
	targetsB = vectors.pop("targetsB")
	targetsY = vectors.pop("targetsY")
	#psmids = vectors.pop("psmid")
	psmids = vectors["psmid"]
	#vectors = vectors[['psmid','mean_pI','mean_pI_ion_other','loc_i-1_bas','loc_i_hydro','mean_mz_ion','mean_heli_ion','mean_bas','peplen','loc_i_heli','loc_i_pI','pmz','pI_ion_other','heli_ion_other','hydro_ion','loc_i_bas','loc_1_bas','mean_hydro_ion','ionnumber','loc_i+1_heli','mean_bas_ion_other','bas_ion_other','loc_i+1_bas','loc_0_bas','loc_i+1_hydro','mz_ion','heli_ion','pI_ion','mean_pI_ion','mean_bas_ion','bas_ion','ionnumber_rel','charge']]
	#targets['target'] = targets['target'].values+targets2['target']
	#targets['target'] = np.log2(targets['target']+0.001)
	#targets['target'] = np.sqrt(targets['target'])
	#print targets.target.values

	#vectors = vectors[targets.target>0]
	#psmids = psmids[targets.target>0].PSMid
	#targets = targets[targets.target>0]

	#psmids = psmids.PSMid

	#selecting charge +2 peptides only!!
	np.random.seed(1)
	upeps = psmids.unique()
	num_psms = len(upeps)
	np.random.shuffle(upeps)

	#creating train/test split
	#numTrain = int(0.8*len(vectors))

	#train_vectors = vectors.iloc[:numTrain]
	#train_targets = targets.iloc[:numTrain]
	#test_vectors = vectors.iloc[numTrain:]
	#test_targets = targets.iloc[numTrain:]

	test_psms = upeps[:int(num_psms*0.2)]


	#with open("testpsm","w") as f:
	#	for t in test_psms:
	#		f.write("%s\n"%t)
	#dddd

	test_vectors = vectors[psmids.isin(test_psms)]
	test_targets = targetsB[psmids.isin(test_psms)]
	train_vectors = vectors[~psmids.isin(test_psms)]
	train_targets = targetsB[~psmids.isin(test_psms)]

	print len(train_targets)
	print train_targets.describe()
	#print test_targets
	##dd

	train_psmids = train_vectors.pop("psmid")
	test_psmids = test_vectors.pop("psmid")

	train_vectors = train_vectors.astype(np.float32)
	test_vectors = test_vectors.astype(np.float32)


	#train_targets.hist()
	#plt.show()

	print len(train_vectors)
	print len(test_vectors)

	sys.stderr.write('loading data done\n')

	#rename features to understand decision tree dump
	train_vectors.columns = ['Feature'+str(i) for i in range(len(train_vectors.columns))]
	test_vectors.columns = ['Feature'+str(i) for i in range(len(train_vectors.columns))]
	numf = len(train_vectors.columns.values)

	#create XGBoost datastructure
	sys.stderr.write('creating DMatrix\n')
	#xtrain = xgb.DMatrix(train_vectors, label=train_targets['y_4'])
	#xeval = xgb.DMatrix(test_vectors, label=test_targets['y_4'])
	xtrain = xgb.DMatrix(train_vectors, label=train_targets)
	xeval = xgb.DMatrix(test_vectors, label=test_targets)
	sys.stderr.write('creating DMatrix done\n')

	evallist  = [(xeval,'eval')]


	#set XGBoost parameters; make sure to tune well!
	param = {"objective":"reg:linear",
	         "nthread":int(args.num_cpu),
	         "silent":1,
	         "eta":0.6,
	         #"max_delta_step":1,
	         "max_depth":11,
			 "gamma":1,
			 "min_child_weight":100,
			 "subsample":1,
			 "colsample_bytree":1,
			 #"scale_pos_weight":num_neg/num_pos
			 #"scale_pos_weight":2
	         }
	plst = param.items()
	plst += [('eval_metric', 'rmse')]

	#train XGBoost
	#bst = xgb.cv( plst, xtrain, 200,nfold=5,callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])
	bst = xgb.train( plst, xtrain, 2000, evallist,early_stopping_rounds=10)
	#bst = xgb.train( plst, xtrain, 30, evallist)

	#save model
	bst.save_model(args.vectors+'.xgboost')

	#bst = xgb.Booster({'nthread':23}) #init model
	#bst.load_model(args.vectors+'.xgboost') # load data



	#get feature importances
	importance = bst.get_fscore()
	importance = sorted(importance.items(), key=operator.itemgetter(1))
	ll = []
	with open("importance.txt","w") as f:
		for feat,n in importance[:]:
			ll.append(feat)
			f.write(feat + "\t" + str(n) + '\n')

	sys.stderr.write("[")
	for l in ll:
		sys.stderr.write("'"+l+"',")

	predictions = bst.predict(xeval)

	tmp = pd.DataFrame()
	tmp['target'] = list(test_targets.values)
	tmp['predictions'] = predictions
	tmp['psmid'] = list(test_psmids.values)
	#tmp['charge'] = list(test_vectors.charge.values)
	#tmp['peplen'] = list(test_vectors.peplen.values)
	tmp.to_pickle('predictions.pkl')

	convert_model_to_c(bst,args,numf)

	for ch in range(8,20):
		print "len %i" % ch
		n1 = 0
		n2 = 0
		tmp3 = tmp[tmp.peplen==ch]
		for pid in tmp3.psmid.unique().values:
			tmp2 = tmp3[tmp3.psmid==pid]
			print pid
			for (t,p) in zip (tmp2.target,tmp2.predictions):
				print "%f %f" % (t,p)
			#n1 += pearsonr(tmp2.target,tmp2.predictions)[0]
			n1 += np.mean(np.abs(tmp2.target-tmp2.predictions))
			#print n1
			n2+=1
		print float(n1)/n2

	#plt.scatter(x=test_targets,y=predictions)
	#plt.show()

	#dump model to .c code

def convert_model_to_c(bst,args,numf):
	#dump model and write .c file
	bst.dump_model('dump.raw.txt')
	num_nodes = []
	mmax = 0
	with open('dump.raw.txt') as f:
		for row in f:
			if row.startswith('booster'):
				if row.startswith('booster[0]'):
					mmax = 0
				else:
					num_nodes.append(mmax+1)
					mmax = 0
				continue
			l=int(row.rstrip().replace(' ','').split(':')[0])
			if l > mmax:
				mmax = l
	num_nodes.append(mmax+1)
	forest = []
	tree = None
	b = 0
	with open('dump.raw.txt') as f:
		for row in f:
			if row.startswith('booster'):
				if row.startswith('booster[0]'):
					tree = [0]*num_nodes[b]
					b += 1
				else:
					forest.append(tree)
					tree = [0]*num_nodes[b]
					b+=1
				continue
			#if b == len(num_nodes)-10: break
			l=row.rstrip().replace(' ','').split(':')
			if l[1][:4] == "leaf":
				tmp = l[1].split('=')
				tree[int(l[0])] = [-1,float(tmp[1]),-1,-1] #!!!!
			else:
				tmp = l[1].split('yes=')
				tmp[0]=tmp[0].replace('[Features','')
				tmp[0]=tmp[0].replace('[Feature','')
				tmp[0]=tmp[0].replace(']','')
				tmp2 = tmp[0].split('<')
				if float(tmp2[1]) < 0: tmp2[1] = 1
				tmp3 = tmp[1].split(",no=")
				tmp4 = tmp3[1].split(',')
				tree[int(l[0])] = [int(tmp2[0]),int(math.ceil(float(tmp2[1]))),int(tmp3[0]),int(tmp4[0])]
		forest.append(tree)

		tmp = args.vectors.replace('.','_')
		tmp2 = tmp.split('/')
		with open(tmp+'_c.c','w') as fout:
			fout.write("static float score_"+args.type+"(unsigned int* v){\n")
			fout.write("float s = 0.;\n")
			#for tt in [0]:
			#for tt in range(len(forest)-10):
			for tt in range(len(forest)):
				fout.write(tree_to_code(forest[tt],0,1))
			fout.write("\nreturn s;}\n")

		with open(tmp+'.pyx','w') as fout:
			fout.write("cdef extern from \"" + tmp2[-1] + "_c.c\":\n")
			fout.write("\tfloat score_%s(short unsigned short[%i] v)\n\n"%(args.type,numf))
			fout.write("def myscore(sv):\n")
			fout.write("\tcdef unsigned short[%i] v = sv\n"%numf)
			fout.write("\treturn score_%s(v)\n"%args.type)

	#os.remove('dump.raw.txt')


def tree_to_code(tree,pos,padding):
	p = "\t"*padding
	if tree[pos][0] == -1:
		if tree[pos][1] < 0:
			return p+"s = s %f;\n"%tree[pos][1]
		else:
			return p+"s = s + %f;\n"%tree[pos][1]
	return p+"if (v[%i]<%i){\n%s}\n%selse{\n%s}"%(tree[pos][0],tree[pos][1],tree_to_code(tree,tree[pos][2],padding+1),p,tree_to_code(tree,tree[pos][3],padding+1))

def print_logo():
	logo = """
 _____ _____ _____ _____ _____
|_   _| __  |  _  |     |   | |
  | | |    -|     |-   -| | | |
  |_| |__|__|__|__|_____|_|___|

           """
	print logo

if __name__ == "__main__":
	print_logo()
	main()
