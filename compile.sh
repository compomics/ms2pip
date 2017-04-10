#rm -f ms2pipfeatures_pyx_CID.c ms2pipfeatures_pyx_CID.so
#python setup_CID.py build_ext --inplace

#rm -f ms2pipfeatures_pyx_HCD.c ms2pipfeatures_pyx_HCD.so
#python setup_HCD.py build_ext --inplace

rm -f ms2pipfeatures_pyx_HCDiTRAQ4.c ms2pipfeatures_pyx_HCDiTRAQ4.so
python setup_HCDiTRAQ.py build_ext --inplace

#rm -f ms2pipfeatures_pyx_HCDiTRAQ4phospho.c ms2pipfeatures_pyx_HCDiTRAQ4phospho.so
#python setup_HCDiTRAQphospho.py build_ext --inplace
