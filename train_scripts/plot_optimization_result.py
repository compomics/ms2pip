import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

data=[]
with open(sys.argv[1]) as f:
	for row in f:
		l=row.replace(">>","").split()
		x = []
		for i in range(0,len(l)-5):
			x.append(float(l[i]))
		xx = x.copy()
		xx.append(float(l[-5]))
		xx.append(10)
		data.append(xx)
		xx = x.copy()
		xx.append(float(l[-4]))
		xx.append(30)
		data.append(xx)
		xx = x.copy()
		xx.append(float(l[-3]))
		xx.append(50)
		data.append(xx)
		xx = x.copy()
		xx.append(float(l[-2]))
		xx.append(70)
		data.append(xx)
		xx = x.copy()
		xx.append(float(l[-1]))
		xx.append(90)
		data.append(xx)
		
d = pd.DataFrame(data,columns=["charge","eval-set","max_depth","num_leaves","value","percentile"])
		
with PdfPages('%s.pdf'%sys.argv[1]) as pdf:
	sns.catplot(x="percentile",y="value",hue="charge",col="eval-set",data=d,kind="point",ci="sd")
	pdf.savefig()
	plt.close() 
	sns.catplot(x="percentile",y="value",hue="max_depth",col="eval-set",data=d,kind="point",ci="sd")
	pdf.savefig()
	plt.close() 
	sns.catplot(x="percentile",y="value",hue="num_leaves",col="eval-set",data=d,kind="point",ci="sd")
	pdf.savefig()
	plt.close() 

