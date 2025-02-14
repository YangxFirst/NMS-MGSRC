function [Xun_Means1,Xun_Stds1] = Totall_LRLSF(dataset)

dataset_name = dataset;
load(dataset);
addpath('LRLSF');
addpath('basic classifier');
addpath('Evaluation');

labeled_rate = 0.7; % 0.7, 0.5, 0.2
modelparameter.round              = 5;
%% Importing Data
if exist('train_data','var')==1
    data=[train_data;test_data];
    target=[train_target;test_target];
    clear train_data test_data train_target test_taget
end
if exist('dataset','var')==1
    data = dataset;
    target = labels;
    clear dataset label
end

data     = double(data);
% 归一化
%minMaxd = mapminmax(data');
%data = minMaxd';
target = double(target>0);
num_data = size(data,1);

num_test = ceil(num_data*0.3);
num_train = num_data - num_test;
num_label = ceil(num_train*labeled_rate);



%% Optimization Parameters
lambda1s = [0.1,1];
lambda2s = [0.01,1];
lambda3s = [1,10];
lambda4s = [0.01,1];

optmParameter.gamma   = 3;

optmParameter.maxIter           = 1; % 最大迭代次数
optmParameter.minimumLossMargin = 0.001; % 两次迭代的最小损失间距  0.0001
optmParameter.bQuiet            = 1;


%% 参数
lambda1_num = length(lambda1s);
lambda2_num = length(lambda2s);
lambda3_num = length(lambda3s);
lambda4_num = length(lambda4s);
Para_num =  lambda1_num * lambda2_num * lambda3_num * lambda4_num;
Result_NEW  = zeros(6,50);
Avg_Means1 = zeros(6,Para_num);
Avg_Stds1 = zeros(6,Para_num);
Xun_Means1 = zeros( 6,Para_num);
Xun_Stds1 = zeros( 6,Para_num);

l1  = 0;
l2  = 0;
l3 = 0;
l4 = 0;
k = 1;
m  = 1;
n = 1;

%% 网格调参
for lambda1 = lambda1s
    optmParameter.lambda1 = lambda1;
    l1 = l1 + 1;
    for lambda2 = lambda2s
        optmParameter.lambda2 = lambda2;
        l2 = l2 + 1;
        for lambda3 = lambda3s
            optmParameter.lambda3 = lambda3;
            l3 = l3 + 1;
            for lambda4 = lambda4s
                optmParameter.lambda4 = lambda4;
                l4 = l4 +1;
                while (m <= modelparameter.round)
                    fprintf('LRLSF Running %s lambda1 - %d/%d lambda2 - %d/%d lambda1 - %d/%d lambda2 - %d/%d \n',dataset_name,l1,lambda1_num,l2,lambda2_num,l3,lambda3_num,l4,lambda4_num);
                    randorder = randperm(num_data);
                    train_index = randorder(1:num_train);
                    test_index = randorder(num_train + 1:num_data);
                    randorder = train_index(randperm(num_train));
                    label_index = randorder(1:num_label);
                    unlabel_index = randorder((num_label+1):num_train);
                    
                    label_data = data(label_index,:); % 有标记训练样本
                    label_target = target(:,label_index); % 有标记训练样本的标签
                    % unlabel_data = data(unlabel_index,:); % 无标记训练样本
                    test_data = data(test_index,:); % 测试样本
                    test_target = target(:,test_index); % 测试样本的标签
                                
                    index=label_target==-1;
                    label_target(index)=0;
                    X = label_data;
                    Y= label_target';
                    [~,num_feature] = size(X);
                    
                    %% Training
                    [W] = LRLSF( X, Y, optmParameter);
                    
                    [~, feature_idx] = sort(sum(W,2),'descend');
                    % time = etime(clock, t0);
    
                    MLKNN_train_data = label_data;
                    MLKNN_test_data = test_data;
                    MLKNN_train_label = label_target;
                    MLKNN_test_label = test_target;
                    MLKNN_train_label(MLKNN_train_label == 0) = -1;
                    MLKNN_test_label(MLKNN_test_label == 0) = -1;
                    
                    %% Begin MLKNN
                    Num=10;
                    Smooth=1;
    %                 [~,num_feature]=size(MLKNN_train_data);
    
                    for i = 1:50
                        fprintf('Running the program with the selected features - %d/%d \n',i,num_feature);
                        f=feature_idx(1:i);
                        [Prior,PriorN,Cond,CondN]=MLKNN_train(MLKNN_train_data(:,f),MLKNN_train_label,Num,Smooth); % Invoking the training procedure
                        [Outputs,Pre_Labels]=MLKNN_test(MLKNN_train_data(:,f),MLKNN_train_label,MLKNN_test_data(:,f),MLKNN_test_label,Num,Prior,PriorN,Cond,CondN);
                        %% Evaluation of NEW
                        Result_NEW(:,i) = EvaluationAll(Pre_Labels,Outputs,MLKNN_test_label);%参数均为转置
                    end
                    Avg_Means1(1:6,k) = mean(Result_NEW,2);%平均值 2代表行
                    Avg_Stds1(1:6,k) = std(Result_NEW,1,2);%标准差
                    
                    X_Means1(:,k) = Avg_Means1(1:6,k);
                    X_Stds1(:,k)  = Avg_Stds1(1:6,k);
                    k = k + 1;
                    m = m + 1;
                end
                m = 1;
                Xun_Means1(1:6,n) = mean(X_Means1,2);
                Xun_Stds1(1:6,n)  = std(X_Stds1,1,2);
                Xun_Means1(7,n) = lambda1;
                Xun_Means1(8,n) = lambda2;
                Xun_Means1(9,n) = lambda3;
                Xun_Means1(10,n) = lambda4; 
                n = n+1;
            end
            l4 = 0;
        end
        l3 = 0;
    end
    l2 = 0;
end

Xun_Means1 = Xun_Means1';
Xun_Stds1 = Xun_Stds1';

rmpath('LRLSF');
rmpath('basic classifier');
rmpath('Evaluation');


end