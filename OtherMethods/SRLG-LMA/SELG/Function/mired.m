function [red] = mired(attrqM,selfMatrixs1,n,k)
%MIRED 此处显示有关此函数的摘要
%   此处显示详细说明
% attrqM=calculateSimilarity(data(:,k),type1);
temp = 0;


selm = selfMatrixs1;

f=min(attrqM,selm);
hcq=-1/n*sum(log2(sum(f,2)./sum(attrqM,2)));
hL= -1/n*sum(log2(sum(selm,2)/n));
mi=hL-hcq;

red = mi;
end

