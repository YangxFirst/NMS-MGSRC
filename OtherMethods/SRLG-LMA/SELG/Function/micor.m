function [cor] = micor(attrqM,labelMatrixs,n,leny1,k)
%MICOR 此处显示有关此函数的摘要
% 此处显示详细说明

temp = 0;

for i=1:leny1 
    f=min(attrqM,labelMatrixs(:,:,i));
    hcq=-1/n*sum(log2(sum(f,2)./sum(attrqM,2)));
    hL= -1/n*sum(log2(sum(labelMatrixs(:,:,i),2)/n));
    mi=hL-hcq;
    temp = temp+mi;
end
cor = temp;
end

