function [count,fitc] = mifitc(attrqM,selfMatrixs1,n,labelsM,k)
%MIFITC 此处显示有关此函数的摘要
%   此处显示详细说明
% attrqM=calculateSimilarity(data(:,k),type1);
temp = 0;
temp1 = 0;fitc=0;
% [n,~]=size(data);
[~,~,d] = size(labelsM);
[~,num] = size(selfMatrixs1);
count = 0;

selm = selfMatrixs1;
f2 = min(attrqM,selm);
f1 = min(f2,labelsM);
f3 = min(selm,labelsM);
fadi = 1/n*sum(log2((sum(f1,2).*sum(selm,2))./(sum(f2,2).*sum(f3,2))));

f=min(selm,labelsM);
hcq=-1/n*sum(log2(sum(f,2)./sum(selm,2)));
hL= -1/n*sum(log2(sum(labelsM,2)/n));
mi=hL-hcq;
mi = mi/num;

temp2 = fadi-mi;
if temp2 > 0
    fitc = temp2;
end



