#rm -f ms2pipfeatures_pyx_CID.c ms2pipfeatures_pyx_CID.so
#python3 setup_CID.py build_ext --inplace

#rm -f ms2pipfeatures_pyx_HCD.c ms2pipfeatures_pyx_HCD.so
#python3 setup_HCD.py build_ext --inplace

#rm -f ms2pipfeatures_pyx_HCDiTRAQ4.c ms2pipfeatures_pyx_HCDiTRAQ4.so
#python3 setup_HCDiTRAQ.py build_ext --inplace

#rm -f ms2pipfeatures_pyx_HCDiTRAQ4phospho.c ms2pipfeatures_pyx_HCDiTRAQ4phospho.so
#python3 setup_HCDiTRAQphospho.py build_ext --inplace

#rm -f ms2pipfeatures_pyx_ETD.c ms2pipfeatures_pyx_ETD.so
#python3 setup_ETD.py build_ext --inplace

rm -f ms2pipfeatures_pyx_HCDch2.c ms2pipfeatures_pyx_HCDch2.so
python3 setup_HCDch2.py build_ext --inplace
