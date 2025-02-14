%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
warning('off')
addpath(genpath(pwd));

datasets = "scene"; % Êý¾Ý¼¯
datasets_num = length(datasets);


for i = 1:datasets_num
    dataset_name = char(datasets(i) + '.mat');
    [Xun_Means,Xun_Stds] = Main(dataset_name);
    pathname = 'D:\Code\table\SRLG-LMA-2024\SELG\results\';
    filename = char(datasets(i) + "_0.7_result.mat"); 
    save([pathname,filename],'Xun_Means','Xun_Stds');



end