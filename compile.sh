rm -f ms2pipfeatures_pyx.c ms2pipfeatures_pyx.so
python setup.py build_ext --inplace

#rm -f vectors_b_pkl.c vectors_b_pkl.so
#python setup_b.py build_ext --inplace
#rm -f vectors_y_pkl.c vectors_y_pkl.so
#python setup_y.py build_ext --inplace
