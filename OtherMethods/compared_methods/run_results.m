clear;clc; addpath(genpath('.\'))

addpath('centered_data');

datasets = "scene"; % 数据集; % 数据集

datasets_num = length(datasets);

pathname1 = 'D:\Code\table\compared_methods\MFS_MCDM\results\';
pathname2 = 'D:\Code\table\compared_methods\D2F\results\';
pathname3 = 'D:\Code\table\compared_methods\FIMF\results\';
pathname4 = 'D:\Code\table\compared_methods\PMU\results\';



for i = 1:datasets_num
    dataset_name = char(datasets(i) + ".mat");
    filename = char(datasets(i) + "_0.7_result.mat");
    % [Xun_Means1,Xun_Stds1] = Total_MFS_MCDM(dataset_name);
    % save([pathname1,filename],'Xun_Means1',"Xun_Stds1");
     [Xun_Means2,Xun_Stds2] = Total_D2F(dataset_name);
    save([pathname2,filename],'Xun_Means2',"Xun_Stds2");
    % [Xun_Means3,Xun_Stds3] = Total_FIMF(dataset_name);
    % save([pathname3,filename],'Xun_Means3',"Xun_Stds3");
    %  [Xun_Means4,Xun_Stds4] = Total_PMU(dataset_name);
    % save([pathname4,filename],'Xun_Means4',"Xun_Stds4");
    
end