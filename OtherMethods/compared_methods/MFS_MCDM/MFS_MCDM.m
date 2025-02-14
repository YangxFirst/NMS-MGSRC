% MFS-MCDM: Multi-label feature selection using multi-criteria decision making-2020
function S1 = MFS_MCDM(dataTrain,dataTrainLabels)

[~, feature_num] = size( dataTrain);
Lambda = 1;
num_class = size(dataTrainLabels,2);
D = pinv(dataTrain'*dataTrain + Lambda*eye(feature_num)) * dataTrain'*dataTrainLabels;
for t = 1:num_class
    w(t) = entropy(dataTrainLabels(:,t));
end
W = w./sum(w);
rCloseness = Topsis(D,ones(1,feature_num),W);
[~, S1] = sort(-rCloseness);
end
