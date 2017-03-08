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

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'pearsonr', pearsonr(preds,labels)[0]

def main():

	parser = argparse.ArgumentParser(description='XGBoost training')    
	parser.add_argument('vectors',metavar='<_vectors.pkl>',
					 help='feature vector file')
	parser.add_argument('type',metavar='<type>',
	         help='model type: [B,Y]')
	parser.add_argument('-c',metavar='INT', action="store", dest='num_cpu', default=23,
	         help='number of cpu\'s to use')
	parser.add_argument('-t',metavar='FILE', action="store", dest='vectorseval',
	         help='additional evaluation file')
	args = parser.parse_args()

	sys.stderr.write('loading data\n')
 
	if args.vectors.split('.')[-1] == 'pkl':
	  vectors = pd.read_pickle(args.vectors)
	elif args.vectors.split('.')[-1] == 'h5':
	  vectors = pd.read_hdf(args.vectors, 'table')
	else:
	  print "unsuported feature vector format"

	if args.vectorseval:
		if args.vectorseval.split('.')[-1] == 'pkl':
		  eval_vectors = pd.read_pickle(args.vectorseval)
		elif args.vectorseval.split('.')[-1] == 'h5':
		  eval_vectors = pd.read_hdf(args.vectorseval, 'table')
		else:
		  print "unsuported feature vector format"
	
		
	#vectors = vectors[vectors.charge==2]	
	#eval_vectors = eval_vectors[eval_vectors.charge==2]	
	#vectors = vectors[vectors.peplen==10]	
	#eval_vectors = eval_vectors[eval_vectors.peplen==10]	
	#vectors = vectors[vectors.ionnumber==5]	
	#eval_vectors = eval_vectors[eval_vectors.ionnumber==5]	

	#vectors = vectors.sample(1000000,replace=False)

	print "%s contains %i feature vectors" % (args.vectors,len(vectors))
	#print "%s contains %i feature vectors" % (args.vectorseval,len(eval_vectors))
				
	psmids = vectors["psmid"]
	np.random.seed(1)
	upeps = psmids.unique()
	num_psms = len(upeps)
	np.random.shuffle(upeps)

	test_psms = upeps[:int(num_psms*0.2)]

	targetsB = vectors.pop("targetsB")
	targetsY = vectors.pop("targetsY")

	test_vectors = vectors[psmids.isin(test_psms)]
	train_vectors = vectors[~psmids.isin(test_psms)]

	if args.type == 'B':
		test_targets = targetsB[psmids.isin(test_psms)]
		train_targets = targetsB[~psmids.isin(test_psms)]
	elif args.type == 'Y':
		test_targets = targetsY[psmids.isin(test_psms)]
		train_targets = targetsY[~psmids.isin(test_psms)]
	else:
		print "Wrong model type argument (should be 'B' or 'Y')."
		exit

	if args.vectorseval:
		targetsBeval = eval_vectors.pop("targetsB")
		targetsYeval = eval_vectors.pop("targetsY")
		if args.type == 'B':
			eval_targets = targetsBeval
		elif args.type == 'Y':
			eval_targets = targetsYeval

	#eval_psmids = eval_vectors.pop("psmid")
	train_psmids = train_vectors.pop("psmid")
	test_psmids = test_vectors.pop("psmid")

	train_vectors = train_vectors.astype(np.float32)
	test_vectors = test_vectors.astype(np.float32)
	#eval_vectors = eval_vectors.astype(np.float32)

	sys.stderr.write('loading data done\n')

	#rename features to understand decision tree dump
	train_vectors.columns = ['Feature'+str(i) for i in range(len(train_vectors.columns))]
	test_vectors.columns = ['Feature'+str(i) for i in range(len(train_vectors.columns))]
	#eval_vectors.columns = ['Feature'+str(i) for i in range(len(eval_vectors.columns))]
	numf = len(train_vectors.columns.values)

	#create XGBoost datastructure
	sys.stderr.write('creating DMatrix\n')
	xtrain = xgb.DMatrix(train_vectors, label=train_targets)
	xtest = xgb.DMatrix(test_vectors, label=test_targets)
	#xeval = xgb.DMatrix(eval_vectors, label=eval_targets)
	sys.stderr.write('creating DMatrix done\n')

	evallist  = [(xtest,'test')]
	#evallist  = [(xeval,'eval'),(xtest,'test')]
	#evallist  = [(xtest,'test'),(xeval,'eval')]

	#set XGBoost parameters; make sure to tune well!
	param = {"objective":"reg:linear",
	         "nthread":int(args.num_cpu),
	         "silent":1,
	         "eta":0.7,
	         "max_delta_step":12,
	         "max_depth":7,
			 "gamma":1,	
			 "min_child_weight":1000,
			 "subsample":1,
			 "colsample_bytree":1,
			 #"scale_pos_weight":num_neg/num_pos
			 #"scale_pos_weight":2
	         }
	plst = param.items()
	plst += [('eval_metric', 'rmse')]

	#train XGBoost
	#bst = xgb.cv( plst, xtrain, 200,nfold=5,callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])
	bst = xgb.train( plst, xtrain, 500, evallist,early_stopping_rounds=10,feval=evalerror,maximize=True)
	#bst = xgb.train( plst, xtrain, 500, evallist,early_stopping_rounds=10)
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

	predictions = bst.predict(xtest)

	tmp = pd.DataFrame()
	tmp['target'] = list(test_targets.values)
	tmp['predictions'] = predictions
	tmp['psmid'] = list(test_psmids.values)
	#tmp['charge'] = list(eval_vectors.charge.values)
	#tmp['peplen'] = list(eval_vectors.peplen.values)
	#tmp.to_pickle('predictions.pkl')
	tmp.to_csv('predictions.csv',index=False)

	
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
