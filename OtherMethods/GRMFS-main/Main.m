function [Xun_Means,Xun_Stds] = Main(dataset)
dataset_name = dataset;
load(dataset);
addpath(genpath('.'));

labeled_rate = 0.7; % 0.7, 0.5, 0.2
modelparameter.round              = 5;
%% Importing Data
if exist('train_data','var')==1
    data=[train_data;test_data];
    target=[train_targets;test_targets];
    clear train_data test_data train_target test_taget
end
if exist('dataset','var')==1
    data = dataset;
    target = labels;
    clear dataset labels
end

data     = double(data);
target = double(target>0);
num_data = size(data,1);

num_test = ceil(num_data*0.3);
num_train = num_data - num_test;
num_label = ceil(num_train*labeled_rate);

Result_NEW  = zeros(6,50);
Avg_Means = zeros(6,1);
Avg_Stds = zeros(6,1);
Xun_Means = zeros( 6,1);
Xun_Stds = zeros( 6,1);
m = 1;
j = 1;
n = 1;


while (m <= modelparameter.round)
    randorder = randperm(num_data);
    train_index = randorder(1:num_train);
    test_index = randorder(num_train + 1:num_data);
    randorder = train_index(randperm(num_train));
    label_index = randorder(1:num_label);
    unlabel_index = randorder((num_label+1):num_train);
    
    label_data = data(label_index,:); % 有标记训练样本
    label_target = target(:,label_index); % 有标记训练样本的标签
    unlabel_data = data(unlabel_index,:); % 无标记训练样本
    test_data = data(test_index,:); % 测试样本
    test_target = target(:,test_index); % 测试样本的标签
    
    train_data = Standard(label_data);
    train_target = label_target;
    
    X_train = train_data;
    Y_train = train_target';
    
    %% feature selection
    lammda = 1.2;
    k = 3;
    
    [Rank, Re]= GRMFS(X_train, Y_train, lammda, k);
    
    
    MLKNN_train_data = label_data;
    MLKNN_test_data = test_data;
    MLKNN_train_label = label_target;
    MLKNN_test_label = test_target;
    MLKNN_train_label(MLKNN_train_label == 0) = -1;
    MLKNN_test_label(MLKNN_test_label == 0) = -1;
    
     %% Begin MLKNN
    fprintf('Start features selections......\n');
    Num=10;
    Smooth=1;
    [~,num_feature]=size(MLKNN_train_data);
    for i = 1:50
        fprintf('Running the program with the selected features - %d/%d \n',i,num_feature);
        f=Rank(1:i);
        [Prior,PriorN,Cond,CondN]=MLKNN_train(MLKNN_train_data(:,f),MLKNN_train_label,Num,Smooth); % Invoking the training procedure
        [Outputs,Pre_Labels]=MLKNN_test(MLKNN_train_data(:,f),MLKNN_train_label,MLKNN_test_data(:,f),MLKNN_test_label,Num,Prior,PriorN,Cond,CondN);
        %% Evaluation of NEW
        Result_NEW(:,i) = EvaluationAll(Pre_Labels,Outputs,MLKNN_test_label);%参数均为转置
    end
    Avg_Means(1:6,j) = mean(Result_NEW,2);%平均值 2代表行
    Avg_Stds(1:6,j) = std(Result_NEW,1,2);%标准差
    
    X_Means(:,j) = Avg_Means(1:6,j);
    X_Stds(:,j)  = Avg_Stds(1:6,j);
    j = j + 1;
    m = m + 1;
end
Xun_Means(1:6,n) = mean(X_Means,2);
Xun_Stds(1:6,n)  = std(X_Stds,1,2);
Xun_Means = Xun_Means';
Xun_Stds = Xun_Stds';







