function [ telco_sample ] = Standard( data )
%数据标准化
telco_sample=mapminmax(data',0,1); %将矩阵的每一行处理成[-1,1]区间，数据应该是每一列是一个样本，每一行是多个样本的同一维，即对于一个M*N的矩阵来说，样本的维度是M，样本数目是N，一共N列N个样本
telco_sample=telco_sample';
end

