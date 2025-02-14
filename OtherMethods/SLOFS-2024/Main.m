function [Xun_Means,Xun_Stds] = Main(dataset)
dataset_name = dataset;
load(dataset);
addpath(genpath('.'));

labeled_rate = 0.7; % 0.7, 0.5, 0.2
modelparameter.round              = 5;
%% Importing Data
if exist('train_data','var')==1
    data=[train_data;test_data];
    target=[train_target;test_taget];
    clear train_data test_data train_target test_taget
end
if exist('dataset','var')==1
    data = dataset;
    target = labels;
    clear dataset labels
end


Result_NEW  = zeros(6,50);
alphas=[0.01,0.1];
betas=[0.9,1]; 
lamda1s=[0.1,1]; 
lamda2s=[0.1,0.3]; 
deltas=[0.01,1]; 
gammas=0.9;

Avg_Means = zeros(6,1);
Avg_Stds = zeros(6,1);
Xun_Means = zeros( 6,1);
Xun_Stds = zeros( 6,1);
m = 1;
k = 1;
n = 1;
a = 0;
b= 0;
d = 0;
g = 0;
l1 = 0;
l2 = 0;

for alpha = alphas
    optmParameter.alpha = alpha;
    a = a + 1;
    for lamda1 = lamda1s
        optmParameter.lamda1 = lamda1;
        l1 = l1 + 1;
        for beta = betas
            optmParameter.beta = beta;
            b = b + 1;
            for delta = deltas
                optmParameter.delta = delta;
                d = d +1;
                for lamda2 = lamda2s
                    optmParameter.lamda2 = lamda2;
                    l2 = l2 +1;
                    for gamma = gammas
                        optmParameter.gamma = gamma;
                        g = g +1;
                        while (m <= modelparameter.round)
                            data     = double(data);
                            target = double(target>0);
                            num_data = size(data,1);
                            
                            num_test = ceil(num_data*0.3);
                            num_train = num_data - num_test;
                            num_label = ceil(num_train*labeled_rate);
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
                            
                            X_train = label_data;
                            Y_train = label_target';
                            X_test = test_data;
                            
                            [num_train, num_label] = size(Y_train); [num_test, num_feature] = size(X_test);
                            pca_remained = round(num_feature*0.95);
                            
                            % Performing PCA
                            all = [X_train; X_test];
                            ave = mean(all);
                            all = (all'-concur(ave', num_train + num_test))';
                            covar = cov(all); covar = full(covar);
                            [u,s,v] = svd(covar);
                            t_matrix = u(:, 1:pca_remained)';
                            all = (t_matrix * all')';
                            X_train = all(1:num_train,:); X_test = all((num_train + 1):(num_train + num_test),:);
                            fea=X_train; gnd=Y_train;
                            % Parameter
                            nClass1=length(unique(gnd));
                            
                            tic
                            [W,obj] =SLOFS(fea,gnd,nClass1,alpha,beta,lamda1,lamda2,delta);
                            tt=toc;
                            score= sqrt(sum(W.*W,2));
                            [res, feature_idx] = sort(score,'descend');
                            
                            MLKNN_train_data = label_data;
                            MLKNN_test_data = test_data;
                            MLKNN_train_label = label_target;
                            MLKNN_test_label = test_target;
                            MLKNN_train_label(MLKNN_train_label == 0) = -1;
                            MLKNN_test_label(MLKNN_test_label == 0) = -1;
                            
                             %% Begin MLKNN
                            Num=10;
                            Smooth=1;
                            [~,num_feature]=size(MLKNN_train_data);
                            for i = 1:50
                                fprintf('Running the program with the selected features - %d/%d \n',i,num_feature);
                                f=feature_idx(1:i);
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
                        Xun_Means(7,n) = alpha;
                        Xun_Means(8,n) = beta;
                        Xun_Means(9,n) = gamma;
                        Xun_Means(10,n) = delta; 
                        Xun_Means(11,n) = lamda1; 
                        Xun_Means(12,n) = lamda2;
                        n = n + 1;
                                
                    end
                    g = 0;
                end
                l2 = 0;
            end
            d = 0;
        end
        b = 0;
    end
    l1 = 0;
end

Xun_Means = Xun_Means';
Xun_Stds = Xun_Stds';
end


