warning off % #ok<WNOFF>
clear all
clc

addpath('function');
addpath('Evaluation');
addpath('data');
addpath('basic classifier');

datasets = "scene"; % 数据集
datasets_num = length(datasets);
for i = 1:datasets_num
    dataset_name = char(datasets(i) + '.mat');
    [Xun_Means,Xun_Stds] = Main(dataset_name);
    
    pathname = 'D:\Code\table\GRMFS-main\results\';
    filename = char(datasets(i) + "_0.7_result.mat"); 
    save([pathname,filename],'Xun_Means','Xun_Stds');
   

end