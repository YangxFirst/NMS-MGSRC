
clear;clc;close all;

addpath(genpath('./'))
str = "scene";

labeled_rate = 0.7; % 0.7, 0.5, 0.2
for ii = 1:length(str)
    load(str(ii) + '.mat')
    modelparameter.round              = 5;
    dataset_name = char(str(ii) + '.mat');
    %% Importing Data
    if exist('train_data','var')==1
        data=[train_data;test_data];
        target=[train_target;test_target];
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

    lambda1s = [0.0001,0.001,0.005,0.01,0.05];
    lambda2s = [0.01,0.1,1,10,100];
    lambda1_num = length(lambda1s);
    lambda2_num = length(lambda2s);
    Para_num =  lambda1_num * lambda2_num;
    Result_NEW  = zeros(6,50);
    Avg_Means = zeros(6,Para_num);
    Avg_Stds = zeros(6,Para_num);
    Xun_Means = zeros(6,Para_num);
    Xun_Stds = zeros(6,Para_num);
    
    i = 1;
    k = 1;
    m  = 1;
    n = 1;
    L1 = 0;
    L2 = 0;

    for lambda1 = lambda1s
        optmParameter.lambda1 = lambda1;
        L1 = L1 + 1;
        for lambda2 = lambda2s
            optmParameter.lambda2 = lambda2;
            L2 = L2 + 1;
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
                
                    train_data = label_data;
                    train_target = label_target;
                    Y = label_target;
                %     Y(train_target<0) = 0;
                    feature_slct = SCNMF(train_data,Y',lambda1,lambda2);
        %             feature_slct = thirdPaperMethod2(train_data,Y');
                    MLKNN_train_data = train_data;
                    MLKNN_test_data = test_data;
                    MLKNN_train_label = train_target;
                    MLKNN_test_label = test_target;
                    MLKNN_train_label(MLKNN_train_label == 0) = -1;
                    MLKNN_test_label(MLKNN_test_label == 0) = -1;
        
                    Num=10;
                    Smooth=1;
                    [~,num_feature]=size(MLKNN_train_data);
                    for i = 1:50
                        fprintf('Running the program with the selected features - %d/%d \n',i,num_feature);
                        f=feature_slct(1:i);
                        [Prior,PriorN,Cond,CondN]=MLKNN_train(MLKNN_train_data(:,f),MLKNN_train_label,Num,Smooth); % Invoking the training procedure
                        [Outputs,Pre_Labels]=MLKNN_test(MLKNN_train_data(:,f),MLKNN_train_label,MLKNN_test_data(:,f),MLKNN_test_label,Num,Prior,PriorN,Cond,CondN);
                        %% Evaluation of NEW
                        Result_NEW(:,i) = EvaluationAll(Pre_Labels,Outputs,MLKNN_test_label);%参数均为转置
                    end
                    Avg_Means(1:6,k) = mean(Result_NEW,2);%平均值 2代表行
                    Avg_Stds(1:6,k) = std(Result_NEW,1,2);%标准差
                    
                    X_Means(:,k) = Avg_Means(1:6,k);
                    X_Stds(:,k)  = Avg_Stds(1:6,k);
                    k = k + 1;
                    m = m + 1;

                end
                m = 1;
                Xun_Means(1:6,n) = mean(X_Means,2);
                Xun_Stds(1:6,n)  = std(X_Stds,1,2);
                Xun_Means(7,n) = lambda1;
                Xun_Means(8,n) = lambda2;
                n = n + 1;
    
        end
        L2=0;
    end
    Xun_Means = Xun_Means';
    Xun_Stds = Xun_Stds';


    pathname = 'D:\Code\table\SCNMF-2024\results\';
    filename = char(dataset_name + "_0.7_result.mat"); 
    save([pathname,filename],'Xun_Means','Xun_Stds');

%         clear train_data test_data train_target test_taget Avg_Means Avg_Stds

end

