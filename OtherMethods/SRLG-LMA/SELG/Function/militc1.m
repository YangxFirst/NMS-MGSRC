function [litc] = militc1(attrqM,labelMatrixs,n,leny1,labelMatrixs1,k)
%MILITC 此处显示有关此函数的摘要
%   此处显示详细说明
% attrqM=calculateSimilarity(data(:,k),type1);
temp = 0;
temp1 = 0;litc=0;

for i=1:leny1

    f2 = min(attrqM,labelMatrixs1(:,:,i));
    f1 = min(f2,labelMatrixs(:,:,i));
    f3 = min(attrqM,labelMatrixs(:,:,i));
    f4 = min(labelMatrixs(:,:,i),labelMatrixs1(:,:,i));
    fadi = 1/n*sum(log2((sum(f1,2).*sum(labelMatrixs(:,:,i),2))./(sum(f3,2).*sum(f4,2))));
    temp = temp + fadi;
      
end
litc = temp ;
end