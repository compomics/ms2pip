rm -f ms2pipfeatures_pyx.c ms2pipfeatures_pyx.so
#rm -f ms2pipfeatures_cython.c ms2pipfeatures_cython.so
#rm -f utils_cython.c utils_cython.so

#python setup.py build_ext --inplace
#python setup2.py build_ext --inplace
python setup.py build_ext --inplace

#rm -f vectors_b_pkl.c vectors_b_pkl.so
#python setup_b.py build_ext --inplace
#rm -f vectors_y_pkl.c vectors_y_pkl.so
#python setup_y.py build_ext --inplace
