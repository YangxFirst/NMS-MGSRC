function [S1, S2, S3]=GRMFSentropy(M1, M2, M3)
[a,~] = size(M1); 
K1 = 0;
K2 = 0;
K3 = 0;
for i=1:a
    S1=-(1/a)*log2(sum(M1(i,1)));
    S2=-(1/a)*log2(sum(M2(i,1)));
    S3=-(1/a)*log2(sum(M3(i,1)));
    K1 = K1 + S1;
    K2 = K2 + S2;
    K3 = K3 + S3;
end
S1 = K1;
S2 = K2;
S3 = K3;
end
