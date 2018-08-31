#rm -f ms2pipfeatures_pyx_CID.c ms2pipfeatures_pyx_CID.so
#python3 setup.py build_ext --inplace CID

rm -f ms2pipfeatures_pyx_HCD.c ms2pipfeatures_pyx_HCD.so
python3 setup.py build_ext --inplace HCD

#rm -f ms2pipfeatures_pyx_HCDiTRAQ4.c ms2pipfeatures_pyx_HCDiTRAQ4.so
#python3 setup.py build_ext --inplace HCDiTRAQ

#rm -f ms2pipfeatures_pyx_HCDiTRAQ4phospho.c ms2pipfeatures_pyx_HCDiTRAQ4phospho.so
#python3 setup.py build_ext --inplace HCDiTRAQphospho

#rm -f ms2pipfeatures_pyx_ETD.c ms2pipfeatures_pyx_ETD.so
#python3 setup.py build_ext --inplace ETD

rm -f ms2pipfeatures_pyx_HCDch2.c ms2pipfeatures_pyx_HCDch2.so
python3 setup.py build_ext --inplace HCDch2

rm -f ms2pipfeatures_pyx_HCDTMT.c ms2pipfeatures_pyx_HCDTMT.so
python3 setup.py build_ext --inplace HCDTMT

rm -f ms2pipfeatures_pyx_TTOF5600.c ms2pipfeatures_pyx_TTOF5600.so
python3 setup.py build_ext --inplace TTOF5600
