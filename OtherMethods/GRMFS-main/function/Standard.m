function [ telco_sample ] = Standard( data )
%���ݱ�׼��
telco_sample=mapminmax(data',0,1); %�������ÿһ�д����[-1,1]���䣬����Ӧ����ÿһ����һ��������ÿһ���Ƕ��������ͬһά��������һ��M*N�ľ�����˵��������ά����M��������Ŀ��N��һ��N��N������
telco_sample=telco_sample';
end

