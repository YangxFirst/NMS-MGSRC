%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off % #ok<WNOFF>
clear all
clc

addpath('function');
addpath('Evaluation');
addpath('centered_data');
addpath('basic classifier');

datasets = "scene"; % Êý¾Ý¼¯
datasets_num = length(datasets);

for i = 1:datasets_num
    dataset_name = char(datasets(i) + '.mat');
    [Xun_Means,Xun_Stds] = Main(dataset_name);
    pathname = 'D:\Code\NMS-MGSRC\results\';
    filename = char(datasets(i) + "_result.mat"); 
    save([pathname,filename],'Xun_Means','Xun_Stds');
end