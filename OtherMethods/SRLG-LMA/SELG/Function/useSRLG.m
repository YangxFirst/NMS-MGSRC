function [res,time] = useSRLG( train_data, train_target,type1,type2,numK)
% train_data = discretization( train_data, 2 );
tic;
[res,time] = SRLG(train_data,train_target,type1,type2,numK) ;
%time=toc;
end


