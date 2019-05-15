import sys
import pandas as pd

tmp = []
for fn in sys.argv[1:]:
	tmp.append(pd.read_hdf(fn, key='table'))
	
tmp2 = pd.concat(tmp)
tmp2.to_hdf("merged.h5", "table")
