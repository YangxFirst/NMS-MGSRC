function [selectind,time] = SRLG(data,label,type1,type2,numK)
%MLFL 此处显示有关此函数的摘要
%   此处显示详细说明
tic
[n,leny1] = size(label);
[~,leny2] = size(data);
selectind = [];

unSelect = zeros(1,leny2); 

labelMatrixs = zeros(n,n,leny1);
labelMatrixs1 = zeros(n,n,leny1);
for i = 1:leny1
    labelMatrixs(:,:,i) = calculateSimilarity(label(:,i),6);
    label1 = label;
    label1(:,i) = [];
    labelMatrixs1(:,:,i) = calculateSimilarity(label1,6);
end

corlist = zeros(1,leny2);
litclist = zeros(1,leny2);
labelsM=calculateSimilarity2(label,type2);
for i = 1:leny2
    featMatrixs = calculateSimilarity(data(:,i),type1);
    corlist(:,i) = micor(featMatrixs,labelMatrixs,n,leny1,i);
    litclist(:,i) = militc1(featMatrixs,labelMatrixs,n,leny1,labelMatrixs1,i);
end


[~,f1] = max(corlist);

selfMatrixs2 = ones(n,n);
for k = 1:numK
    mit = zeros(1,leny2);
    if k == 35
        mm = 0;
    end
    if k == 1
        selectind = [selectind,f1];
        unSelect(1,f1) = 1;
    else
        for i = 1:leny2 
            if unSelect(i) == 0
                featMatrixs = calculateSimilarity(data(:,i),type1);
                red = mired(featMatrixs,selfMatrixs2,n,i);
                [count,fitc] = mifitc(featMatrixs,selfMatrixs2,n,labelsM,i);
                total = corlist(i)/leny1 - red/(k-1) + fitc/leny1 + litclist(i)/leny1;
                mit(1,i) = total;
            end

        end
        if max(mit) ~= 0
            [~,f2] = max(mit);
            selectind = [selectind,f2];
            unSelect(1,f2) = 1;
        else
            mit = abs(mit);
            ret = min(mit(mit ~= 0));
            [~,f2] = find(mit == ret);
            selectind = [selectind,f2];
            unSelect(1,f2) = 1;
        end
    end
    selfMatrixs1 = calculateSimilarity(data(:,selectind(k)),type1);
    selfMatrixs2 = min(selfMatrixs2,selfMatrixs1);
end
time=toc;
end
