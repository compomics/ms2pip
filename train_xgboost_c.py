import os
import sys
import argparse
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


np.random.seed(1)


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'pearsonr', pearsonr(preds, labels)[0]


def convert_model_to_c(bst, args, numf):
    filename = "{}_{}".format(args.vectors.split('.')[-2], args.type)
    bst.dump_model("{}_dump.raw.txt".format(filename))
    num_nodes = []
    mmax = 0
    with open("{}_dump.raw.txt".format(filename)) as f:
        for row in f:
            if row.startswith('booster'):
                if row.startswith('booster[0]'):
                    mmax = 0
                else:
                    num_nodes.append(mmax + 1)
                    mmax = 0
                continue
            l = int(row.rstrip().replace(' ', '').split(':')[0])
            if l > mmax:
                mmax = l
    num_nodes.append(mmax + 1)
    forest = []
    tree = None
    b = 0
    with open("{}_dump.raw.txt".format(filename)) as f:
        for row in f:
            if row.startswith('booster'):
                if row.startswith('booster[0]'):
                    tree = [0] * num_nodes[b]
                    b += 1
                else:
                    forest.append(tree)
                    tree = [0] * num_nodes[b]
                    b += 1
                continue
            l = row.rstrip().replace(' ', '').split(':')
            if l[1][:4] == "leaf":
                tmp = l[1].split('=')
                tree[int(l[0])] = [-1, float(tmp[1]), -1, -1]  # !!!!
            else:
                tmp = l[1].split('yes=')
                tmp[0] = tmp[0].replace('[Features', '')
                tmp[0] = tmp[0].replace('[Feature', '')
                tmp[0] = tmp[0].replace('[f', '')
                tmp[0] = tmp[0].replace(']', '')
                tmp2 = tmp[0].split('<')
                if float(tmp2[1]) < 0:
                    tmp2[1] = 1
                tmp3 = tmp[1].split(",no=")
                tmp4 = tmp3[1].split(',')
                tree[int(l[0])] = [int(tmp2[0]), int(math.ceil(float(tmp2[1]))), int(tmp3[0]), int(tmp4[0])]
        forest.append(tree)

        with open('{}.c'.format(filename), 'w') as fout:
            fout.write("static float score_{}(unsigned int* v){{\n".format(args.type))
            fout.write("float s = 0.;\n")
            for tt in range(len(forest)):
                fout.write(tree_to_code(forest[tt], 0, 1))
            fout.write("\nreturn s;}\n")

        """
        with open('{}.pyx'.format(filename), 'w') as fout:
            fout.write("cdef extern from \"{}.c\":\n".format(filename))
            fout.write("\tfloat score_{}(short unsigned short[{}] v)\n\n".format(args.type, numf))
            fout.write("def myscore(sv):\n")
            fout.write("\tcdef unsigned short[{}] v = sv\n".format(numf))
            fout.write("\treturn score_{}(v)\n".format(args.type))
        """

    os.remove("{}_dump.raw.txt".format(filename))


def tree_to_code(tree, pos, padding):
    p = "\t" * padding
    if tree[pos][0] == -1:
        if tree[pos][1] < 0:
            return p + "s = s {};\n".format(tree[pos][1])
        else:
            return p + "s = s + {};\n".format(tree[pos][1])
    return p + "if (v[{}]<{}){{\n{}}}\n{}else{{\n{}}}".format(tree[pos][0], tree[pos][1], tree_to_code(tree, tree[pos][2], padding + 1), p, tree_to_code(tree, tree[pos][3], padding + 1))


def print_logo():
    logo = """
     _____ _____ _____ _____ _____
    |_   _| __  |  _  |     |   | |
      | | |    -|     |-   -| | | |
      |_| |__|__|__|__|_____|_|___|

    """
    print(logo)


if __name__ == "__main__":
    print_logo()
    print("Using XGBoost version {}".format(xgb.__version__))

    parser = argparse.ArgumentParser(description='XGBoost training')
    parser.add_argument('vectors', metavar='<_vectors.pkl>',
        help='feature vector file')
    parser.add_argument('type', metavar='<type>',
        help='model type: [B,Y,C,Z]')
    parser.add_argument('-c', metavar='INT', action="store", dest='num_cpu', default=23,
        help='number of cpu\'s to use')
    parser.add_argument('-t', metavar='FILE', action="store", dest='vectorseval',
        help='additional evaluation file')
    parser.add_argument("-p", action="store_true", dest='make_plots', default=False,
        help="output plots")
    args = parser.parse_args()

    filename = "{}_{}".format(args.vectors.split('.')[-2], args.type)
    print("Using filename {}".format(filename))

    """
    # LOAD DATA
    """
    print("Loading data...")

    if args.type not in ['B', 'Y', 'C', 'Z']:
        print("Wrong model type argument (should be 'B', 'Y', 'C' or 'Z').")
        sys.exit()

    if args.vectors.split('.')[-1] == 'pkl':
        vectors = pd.read_pickle(args.vectors)
    elif args.vectors.split('.')[-1] == 'h5':
        vectors = pd.read_hdf(args.vectors, 'table')
    else:
        print("Unsuported feature vector format")
        sys.exit()
    print("{} contains {} feature vectors".format(args.vectors, len(vectors)))

    if args.vectorseval:
        if args.vectorseval.split('.')[-1] == 'pkl':
            eval_vectors = pd.read_pickle(args.vectorseval)
        elif args.vectorseval.split('.')[-1] == 'h5':
            eval_vectors = pd.read_hdf(args.vectorseval, 'table')
        else:
            print("Unsuported feature vector format")
            sys.exit()
        print("{} contains {} feature vectors".format(args.vectorseval, len(eval_vectors)))

    # Extract targets
    targetsB = vectors.pop('targetsB')
    targetsY = vectors.pop('targetsY')
    if 'targetsC' in vectors.columns:
        targetsC = vectors.pop('targetsC')
        targetsZ = vectors.pop('targetsZ')
    if 'targets2B' in vectors.columns:
        vectors.drop(['targets2B', 'Targets2Y'], axis=0, inplace=True)
    if 'Targets2C' in vectors.columns:
        vectors.drop(['Targets2C', 'Targets2Z'], axis=0, inplace=True)

    # Split data into test and train set
    psmids = vectors['psmid']
    upeps = psmids.unique()
    np.random.shuffle(upeps)
    test_psms = upeps[:int(len(upeps) * 0.1)]

    test_vectors = vectors[psmids.isin(test_psms)]
    train_vectors = vectors[~psmids.isin(test_psms)]

    test_psmids = test_vectors.pop('psmid')
    train_psmids = train_vectors.pop('psmid')

    if args.type == 'B':
        test_targets = targetsB[psmids.isin(test_psms)]
        train_targets = targetsB[~psmids.isin(test_psms)]
    elif args.type == 'Y':
        test_targets = targetsY[psmids.isin(test_psms)]
        train_targets = targetsY[~psmids.isin(test_psms)]
    elif args.type == 'C':
        test_targets = targetsC[psmids.isin(test_psms)]
        train_targets = targetsC[~psmids.isin(test_psms)]
    elif args.type == 'Z':
        test_targets = targetsZ[psmids.isin(test_psms)]
        train_targets = targetsZ[~psmids.isin(test_psms)]

    train_vectors = train_vectors.astype(np.float32)
    test_vectors = test_vectors.astype(np.float32)

    if args.vectorseval:
        targetsBeval = eval_vectors.pop('targetsB')
        targetsYeval = eval_vectors.pop('targetsY')
        if 'targetsC' in vectors.columns:
            targetsCeval = eval_vectors.pop('targetsC')
            targetsZeval = eval_vectors.pop('targetsZ')

        if args.type == 'B':
            eval_targets = targetsBeval
        elif args.type == 'Y':
            eval_targets = targetsYeval
        elif args.type == 'C':
            eval_targets = targetsCeval
        elif args.type == 'Z':
            eval_targets = targetsZeval

        eval_psmids = eval_vectors.pop('psmid')
        eval_vectors = eval_vectors.astype(np.float32)

    numf = len(train_vectors.columns.values)

    # Rename features to understand decision tree dump
    train_vectors.columns = ['Feature' + str(i) for i in range(len(train_vectors.columns))]
    test_vectors.columns = ['Feature' + str(i) for i in range(len(train_vectors.columns))]
    if args.vectorseval:
        eval_vectors.columns = ['Feature' + str(i) for i in range(len(eval_vectors.columns))]

    """
    # CREATE XGBOOST DATASTRUCTURE
    """
    print("Creating DMatrix...")
    xtrain = xgb.DMatrix(train_vectors, label=train_targets)
    xtest = xgb.DMatrix(test_vectors, label=test_targets)

    if args.vectorseval:
        xeval = xgb.DMatrix(eval_vectors, label=eval_targets)
        evallist = [(xeval, 'eval'), (xtest, 'test')]
    else:
        evallist = [(xtest, 'test')]

    """
    # TRAIN XGBOOST
    """
    print("Training XGboost model...")

    # Set XGBoost parameters; make sure to tune well!
    param = {"objective": "reg:linear",
            "nthread": int(args.num_cpu),
            "silent": 1,
            "eta": 1,
            # "max_delta_step": 12,
            "max_depth": 8,
            "gamma": 1,
            "min_child_weight": 700,
            "subsample": 1,
            "colsample_bytree": 1,
            #"scale_pos_weight": num_neg/num_pos,
            #"scale_pos_weight": 2,
            "eval_metric": 'rmse'
            }
    plst = param.items()

    # Train XGBoost
    bst = xgb.train(plst, xtrain, 300, evallist, early_stopping_rounds=10, feval=evalerror, maximize=True)
    # bst = xgb.train( plst, xtrain, 500, evallist, early_stopping_rounds=10)
    # bst = xgb.train( plst, xtrain, 30, evallist)
    # bst = xgb.cv( plst, xtrain, 200, nfold=5, callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])

    # Save model
    bst.save_model("{}.xgboost".format(filename))

    # Load previously saved model here, if necessary
    # bst = xgb.Booster({'nthread':23}) #init model
    # bst.load_model(filename+'.xgboost') # load data

    """
    OUTPUT MODEL TO C-CODE
    """
    print("Writing model to C code...")
    convert_model_to_c(bst, args, numf)

    """
    MODEL ANALYSIS
    """
    # Get feature importances
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    ll = []
    with open("{}_importance.csv".format(filename), "w") as f:
        f.write("Name,F-score\n")
        for feat, n in importance[:]:
            ll.append(feat)
            f.write("{},{}\n".format(feat, str(n)))

    #  Print feature importances
    print_feature_importances = False
    if print_feature_importances:
        print('[')
        for l in ll:
            sys.stderr.write("'{}',".format(l))
        print(']')

    # Use new model to make predictions on test set and write to csv
    predictions = bst.predict(xtest)
    tmp = pd.DataFrame()
    tmp['target'] = list(test_targets.values)
    tmp['predictions'] = predictions
    tmp['psmid'] = list(test_psmids.values)
    tmp.to_csv("{}_predictions.csv".format(filename), index=False)
    # tmp.to_pickle("{}_predictions.pkl".format(filename))

    """
    for ch in range(8, 20):
        print("len {}".format(ch))
        n1 = 0
        n2 = 0
        tmp3 = tmp[tmp.peplen == ch]
        for pid in tmp3.psmid.unique().values:
            tmp2 = tmp3[tmp3.psmid == pid]
            print(pid)
            for (t, p) in zip(tmp2.target, tmp2.predictions):
                print("{} {}".format(t, p))
            # n1 += pearsonr(tmp2.target, tmp2.predictions)[0]
            n1 += np.mean(np.abs(tmp2.target - tmp2.predictions))
            # print(n1)
            n2 += 1
        print(n1 / n2)
    """

    plt.figure()
    plt.scatter(x=test_targets, y=predictions)
    plt.title('Test set')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.savefig("{}_test.png".format(filename))
    if args.make_plots:
        plt.show()

    print("Ready!")
