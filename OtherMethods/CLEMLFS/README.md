# CLEMLFS
Multi-label feature selection based on correlation label enhancement

Document Description:

CLEMLFS.m is the algorithmic function of the paper

mainMLKNNAllParamter_CLEMLFS.m implements how to call the CLEMLFS algorithm

MLKNN_train and MLKNN_test functions can be found on Zhang Minling's homepage at https://palm.seu.edu.cn/zhangml/ (Resources, Multi-label lazy learning approach). 

Tip: The features of the input data for CLEMLFS need to be normalized using max-min normalization (such as data=normalize(data, 'range');) to prevent the feature values from being too small, leading to ineffective label enhancement.

The citation for the paper is below:

He Z, Lin Y, Wang C, et al. Multi-label feature selection based on correlation label enhancement[J]. Information Sciences, 2023, 647: 119526.
