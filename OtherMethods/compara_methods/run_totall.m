clear;clc; addpath(genpath('.\'))

addpath('centered_data');
datasets = "scene"; % 数据集

datasets_num = length(datasets);

pathname1 = 'D:\Code\table\compara_methods\LRLSF\results\';
pathname2 = 'D:\Code\table\compara_methods\MRDM\results\';



for i = 1:datasets_num
    dataset_name = char(datasets(i) + ".mat");
    filename = char(datasets(i) + "_0.7_result.mat");
      [Xun_Means1,Xun_Stds1] = Totall_LRLSF(dataset_name);
    save([pathname1,filename],'Xun_Means1',"Xun_Stds1");
      [Xun_Means2,Xun_Stds2] = Totall_MRDM(dataset_name);
    save([pathname2,filename],'Xun_Means2',"Xun_Stds2");
    
end