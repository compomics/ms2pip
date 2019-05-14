import sys
import os.path
import multiprocessing

def runM(dn,x):
	os.system("python3.5 ms2pipC.py -m 1 -c config.file -w %s.h5 -s %s %s"% (dn,dn,dn.replace("/mgf/","/peprecs/")+".peprecR"))

myPool = multiprocessing.Pool(24)

for dn in sys.argv[1:]:		
	t = myPool.apply_async(runM, args=(dn,0))

myPool.close()
myPool.join()



		



