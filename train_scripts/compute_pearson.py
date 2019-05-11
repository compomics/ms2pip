import sys
import math
#from scipy import spatial
#import numpy as np
#from scipy.stats import pearsonr
#from sklearn.metrics import mean_absolute_error

"""
def spectral_angle(X,Y):
    epsilon = 1e-07
    true = np.array(X)
    pred = np.array(Y)
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = true_masked / math.sqrt(max(sum(true_masked**2), epsilon))
    true_norm = pred_masked / math.sqrt(max(sum(pred_masked**2), epsilon))
    return (2.*math.acos(spatial.distance.cosine(true_norm,pred_norm))/math.pi)
"""
 
def pearson(X,Y):
    Xmean = sum(X)/len(X)
    Ymean = sum(Y)/len(Y)
    
    x = [var-Xmean for var in X]
    y = [var-Ymean for var in Y]
    
    xy =[a*b for a,b in list(zip(x,y))]
    sum_xy = sum(xy)
    
    x_square = [a*a for a in x]
    y_square = [b*b for b in y]
    
    sum_x_square = sum(x_square)
    sum_y_square = sum(y_square)
    
    sum_x_square_sum_y_square = sum_x_square*sum_y_square
    sqrt_sum_x_square_sum_y_square = math.sqrt(sum_x_square_sum_y_square)
    
    tmp = sqrt_sum_x_square_sum_y_square
    if tmp == 0:
        return 0
    return sum_xy/tmp
    
    
x = []
y = []  
buf = []
prev = ""
print("Title ms2pip-pearsonr")
with open(sys.argv[1]) as f:
    f.readline()
    for row in f:
        l=row.rstrip().split(",")       
        if (l[2]=="B")&(l[3]=="1"):
            if len(x) != 0:
                #print("%s %f %f"%(prev,spectral_angle(x,y),pearson(x,y)))
                #print(">%s %f"%(prev,pearson(x,y)))
                buf.append(pearson(x,y))
                prev = l[0]
            x = []
            y = []
        if int(l[1]) < 2: continue
        if int(l[1]) > 4: continue
        if l[2] == "B2": continue
        if l[2] == "B2": continue       
        #if l[2] != "2": continue
        #if l[2] == "Y": continue
        x.append(float(l[5]))
        y.append(float(l[6]))
        #x.append(max(min(float(l[5]),1.),0))
        #y.append(max(min(float(l[6]),1.),0))
        #x.append(2**(float(l[5])+0.001))
        #y.append(2**(float(l[6])+0.001))

print(sorted(buf)[int(len(buf)/2)])
    
