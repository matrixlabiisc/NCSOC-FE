#ifdef DFTFE_MINIMAL_COMPILE
template class kohnShamDFTOperatorCUDAClass<2,2>;
template class kohnShamDFTOperatorCUDAClass<3,3>;
template class kohnShamDFTOperatorCUDAClass<4,4>;
template class kohnShamDFTOperatorCUDAClass<5,5>;
template class kohnShamDFTOperatorCUDAClass<6,6>;
template class kohnShamDFTOperatorCUDAClass<6,7>;
template class kohnShamDFTOperatorCUDAClass<6,8>;
template class kohnShamDFTOperatorCUDAClass<6,9>;
template class kohnShamDFTOperatorCUDAClass<7,7>;
#else
template class kohnShamDFTOperatorCUDAClass<1,1>;
template class kohnShamDFTOperatorCUDAClass<1,2>;
template class kohnShamDFTOperatorCUDAClass<2,2>;
template class kohnShamDFTOperatorCUDAClass<2,3>;
template class kohnShamDFTOperatorCUDAClass<2,4>;
template class kohnShamDFTOperatorCUDAClass<3,3>;
template class kohnShamDFTOperatorCUDAClass<3,4>;
template class kohnShamDFTOperatorCUDAClass<3,5>;
template class kohnShamDFTOperatorCUDAClass<3,6>;
template class kohnShamDFTOperatorCUDAClass<4,4>;
template class kohnShamDFTOperatorCUDAClass<4,5>;
template class kohnShamDFTOperatorCUDAClass<4,6>;
template class kohnShamDFTOperatorCUDAClass<4,7>;
template class kohnShamDFTOperatorCUDAClass<4,8>;
template class kohnShamDFTOperatorCUDAClass<5,5>;
template class kohnShamDFTOperatorCUDAClass<5,6>;
template class kohnShamDFTOperatorCUDAClass<5,7>;
template class kohnShamDFTOperatorCUDAClass<5,8>;
template class kohnShamDFTOperatorCUDAClass<5,9>;
template class kohnShamDFTOperatorCUDAClass<5,10>;
template class kohnShamDFTOperatorCUDAClass<6,6>;
template class kohnShamDFTOperatorCUDAClass<6,7>;
template class kohnShamDFTOperatorCUDAClass<6,8>;
template class kohnShamDFTOperatorCUDAClass<6,9>;
template class kohnShamDFTOperatorCUDAClass<6,10>;
template class kohnShamDFTOperatorCUDAClass<6,11>;
template class kohnShamDFTOperatorCUDAClass<6,12>;
template class kohnShamDFTOperatorCUDAClass<7,7>;
template class kohnShamDFTOperatorCUDAClass<7,8>;
template class kohnShamDFTOperatorCUDAClass<7,9>;
template class kohnShamDFTOperatorCUDAClass<7,10>;
template class kohnShamDFTOperatorCUDAClass<7,11>;
template class kohnShamDFTOperatorCUDAClass<7,12>;
template class kohnShamDFTOperatorCUDAClass<7,13>;
template class kohnShamDFTOperatorCUDAClass<7,14>;
template class kohnShamDFTOperatorCUDAClass<8,8>;
template class kohnShamDFTOperatorCUDAClass<8,9>;
template class kohnShamDFTOperatorCUDAClass<8,10>;
template class kohnShamDFTOperatorCUDAClass<8,11>;
template class kohnShamDFTOperatorCUDAClass<8,12>;
template class kohnShamDFTOperatorCUDAClass<8,13>;
template class kohnShamDFTOperatorCUDAClass<8,14>;
template class kohnShamDFTOperatorCUDAClass<8,15>;
template class kohnShamDFTOperatorCUDAClass<8,16>;
#endif
