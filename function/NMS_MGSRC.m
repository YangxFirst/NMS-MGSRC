function [W_entire,M_entire] = NMS_MGSRC(label_data,unlabel_data,label_target,LPparam,optmParameter)
%% optimization parameters
lambda1          = optmParameter.lambda1;  % 
alpha            = optmParameter.alpha;    % 
beta             = optmParameter.beta;     % 
gamma            = optmParameter.gamma;    % 
lambda2          = optmParameter.lambda2;  % 
sigma            = optmParameter.sigma;
miu              = optmParameter.miu;
maxIter          = optmParameter.maxIter;
miniLossMargin   = optmParameter.minimumLossMargin;

%% Label Propagation Parameters
lambda_l = LPparam.lambda_l;
lambda_u = LPparam.lambda_u;
K = LPparam.K;

%% LP initializtion
Y_temp = label_target';
[num_label,~] = size(label_data);
[~,c] = size(Y_temp);
[num_unlabel,~] = size(unlabel_data);
Y_unlabel = zeros(num_unlabel,c);
num_train = num_label + num_unlabel;
y_label = zeros(num_label,1);
y_unlabel = ones(num_unlabel,1);
Y_temp=[Y_temp y_label];
Y_unlabel = [Y_unlabel y_unlabel];
X = [label_data;unlabel_data];
Y = [Y_temp;Y_unlabel];

%% Graph Construction
G = pdist2(X,X);
for i =1:num_train
    temp = G(i,:);
    Gs =sort(temp);
    temp  = (temp <=  Gs(K));
    G(i,:) = temp;
end
dd = sum(G,2);
D = diag(dd);
G_ = D\G;

%% Multi-label label Propagation
Y_ = Y ./ sum(Y,2);
Y_l = Y_(1:num_label,:);
Y_u = Y_((num_label+1):num_train,:);
% G_ll = W_(1:num_label,1:num_label);
% G_lu = W_(1:num_label,(num_label+1):num_train);
G_ul = G_((num_label+1):num_train,1:num_label);
G_uu = G_((num_label+1):num_train,(num_label+1):num_train);
I = eye(num_unlabel);
I_lu = lambda_u * I;
Fl = Y_l(:,1:c);
Fu = (I - I_lu*G_uu)\(I_lu*G_ul*Y_l + (I-I_lu)*Y_u);
Fu = Fu(:,1:c);

%% label correlation
RR = pdist2(label_target+eps,label_target+eps, 'cosine' );%R 得到的只是距离，距离与相似度是成反比的
C = 1 - RR;%用1—RR得到的就是相似度矩阵
%C = abs(C);
Fu_ = Fu * C;
F_ = [Fl;Fu_];
F_ = F_';

%% 计算Z
X_temp = X';
[m,n]=size(X_temp);
[U,S,V]=svd(X_temp);
q=diag(S);
t=sum(q>0);%t=rank(Xtrn)
U1=U(:,1:t);
U2=U(:,t+1:m);
V1=V(:,1:t);
V2=V(:,t+1:n);
% miu=param.miu; %max trace(HX'PP'XHL) s.t. P'(miuXX'+(1-miu)I)P=I
Sigma=S(1:t,1:t);
A1=Sigma/(miu*(Sigma^2)+(1-miu)*eye(t,t));
A=A1^(1/2)*Sigma^(-1/2);
one=ones(n,1);
H=eye(n,n)-(1/n)*(one*one');
B=F_*H*V1*Sigma*A;
[P1,Lamtha,P2]=svd(B);
Z=P2'*A*Sigma*V1';

ZZ=mapminmax(Z,0,1);
cludata=ZZ;
K_range = min(size(cludata,1), 10);
eva_DBI_1= evalclusters(cludata,'kmeans','silhouette','KList',1:K_range);
kkk=eva_DBI_1.OptimalK;

temp_Z = Z';
% cen_Z = temp_Z - repmat(mean(temp_Z,1),size(temp_Z,1),1);
% if sum(sum(isnan(temp_Z)))>0
%     temp_Z = Z'+eps;
%     cen_Z = temp_Z - repmat(mean(temp_Z,1),size(temp_Z,1),1);
% end

cen_Z=temp_Z;
%Clu=zscore(cen_Z');
Clu=cen_Z';

W_entire=zeros(m,t);
M_entire=zeros(m,t);

%% Z clustering by column
kk=kkk;

if kk>1
    idx = eva_DBI_1.OptimalY; 
    id=cell(kk,1);
    loss=cell(kk,1);
    for i=1:kk
    id{i,1}=find(idx==i);
    end
else
     id{kk,1}=(1:1:t)';
end

% for i=1:kk
%     [row,~]=size(id{i,1});
%     if row==1
%        error('grouping error');
%     end
% end

for i=1:kk
    cen_Zi=cen_Z(:,id{i,1});
    %% optimization initializtion
    num_dim = size(X,2);
    % XT = X';
    XTX = X'*X;
    XTZ = X'*cen_Zi;
    % 1范数约束的W
    W = (XTX + sigma*eye(num_dim)) \ (XTZ);
    W_1 = W;
    % 2，1范数约束的M
    M = (XTX + sigma*eye(num_dim)) \ (XTZ);
    M_1 = M;
    
    %% instance correlation
    L1 = diag(sum(G,2)) - G;

    %% Iterative  
    iter = 1; 
    oldloss = 0;
    tk = 1;
    tk_1 = 1;

    % HSIC用到的参数
    n1=size(W,1);
    I1=eye(n1,n1);
    Wh = W';
    [~,n2] = size(Wh);
    one=ones(n2,1);
    H1 = eye(n2,n2)-(1/n2)*(one*one');

    % 低维流形相似性
    L2 = label_correlation(cen_Zi);

    %% 计算关于f(W)和f(M)的Lipschitz constants
    A = gradL21(M);
    varepsilon = 0.01;
    Lf_W = sqrt(norm(XTX)^2 + alpha*norm(L2)^2+ gamma*norm(X'*L1*X)^2);
    Lf_M = sqrt(norm(XTX)^2 + beta*norm(A)^2 + alpha*norm(L2)^2);

    %% s-proximal gradient(S-APG)
    while iter <= maxIter
        A = gradL21(M);
        XTX = X'*X;
        XTZ = X'*cen_Zi;
        
      %solve W
        Zeta_Wk  = W + (tk_1 - 1)/tk * (W - W_1);
    
        % calculate the Lipschitz constant of F_mu(W)
        s1_W = varepsilon*norm(M)*sqrt(2/num_dim);
        s2_W = lambda1*norm(M)*sqrt(num_dim/2)+sqrt(Lf_W*varepsilon);
        mu_W = s1_W/s2_W; 
        Lip_W = Lf_W+(lambda1*norm(M)^2)/mu_W;
    
        % calculate the graid of F_mu(W) 
        grad_W_F_1=XTX*Zeta_Wk+XTX*M-XTZ+alpha*Zeta_Wk*L2+gamma*(X'*L1*X*Zeta_Wk+X'*L1*X*M);
        grad_W_q=(1/mu_W)*M*(Zeta_Wk'*M-softthres(Zeta_Wk'*M,mu_W))';
        grad_W_p=(1/mu_W)*M*(M'*Zeta_Wk-softthres(M'*Zeta_Wk,mu_W));
        grad_W_F=grad_W_F_1+(lambda1/2)*(grad_W_q+grad_W_p);
             
        % calculate W(k)
        r1=(1/Lip_W);
        Wk=Zeta_Wk-r1*grad_W_F;
        
%     % updata tk，W, M
%     tk_1=tk;
%     tk=(1 + sqrt(4*tk^2 + 1))/2;
%        
%     W_1=W;
%     M_1=M;
    
%     % calculate W^k
%     q1=lambda2/Lip_W;   
%     W = softthres(Wk,q1);
    
    %solve M
      Zeta_Mk  = M + (tk_1 - 1)/tk * (M - M_1);
    
      % calculate the Lipschitz constant of F_mu(M)
      s1_M = varepsilon*norm(W)*sqrt(2/num_dim);
      s2_M = lambda1*norm(W)*sqrt(num_dim/2)+sqrt(Lf_M*varepsilon);
      mu_M = s1_M/s2_M; 
      Lip_M = Lf_M+(lambda1*norm(W)^2)/mu_M;

      % calculate the graid of F_mu(M) 
      grad_M_F_1=XTX*W+XTX*Zeta_Mk-XTZ+beta*A*Zeta_Mk+alpha*Zeta_Mk*L2;
      grad_M_q=(1/mu_M)*W*(Zeta_Mk'*W-softthres(Zeta_Mk'*W,mu_M))';
      grad_M_p=(1/mu_M)*W*(W'*Zeta_Mk-softthres(W'*Zeta_Mk,mu_M));
      grad_M_F=grad_M_F_1+(lambda1/2)*(grad_M_q+grad_M_p);

      % calculate M(k)  
      r2=(1/Lip_M);
      Mk=Zeta_Mk-r2*grad_M_F;

      % updata tk，W, M
      tk_1=tk;
      tk=(1 + sqrt(4*tk^2 + 1))/2;
       
      W_1=W;
      M_1=M;
    
      % calculate W^k, M^k
      q2=gamma/Lip_M; 
      M = (q2*X'*L1*X+I1)\(Mk-q2*X'*L1*X*W);
      q1=lambda2/Lip_W;   
      W = softthres(Wk,q1);

      % Calculate the value of the objective function
      %Q = M+W; 

      %% 开始计算损失函数的值
      O1 = (X*(M+W) - cen_Zi);
      DiscriminantLoss = (1/2)*trace(O1'* O1);
      WM_RedundantLoss = (lambda1/2)*(norm(W'*M,1)+norm(M'*W,1));
      L_correlationLoss = (alpha/2)*(trace(M*L2*M')+trace(W*L2*W'));
      sparsity1    = lambda2*norm(W,1);
      sparsity2    = (beta/2)*trace(M'*A*M);
      sample_correlationLoss = (gamma/2)*trace((X*(W+M))'*L1*(X*(W+M)));

      totalloss = DiscriminantLoss + WM_RedundantLoss + sparsity1 + sparsity2 + L_correlationLoss + sample_correlationLoss;
       
      loss{i,1}(iter) = totalloss;
      if abs(oldloss - totalloss) <= miniLossMargin
          %本次迭代的结果与上次的结果相差少于预订的最小损失间距时，结束循环
          break;
      elseif totalloss <=0
          break;
      else
          oldloss = totalloss;
      end
    
      iter=iter+1;
    end
    %%Convergence graphs
%     xlabel=1:1:numel(loss{i,1});
%     subplot(kk,1,i)
%     plot(xlabel,loss{i,1})
%       W = W(:,1:r);
%       M = M(:,1:r);
 W_entire(:,id{i,1})=W;
 M_entire(:,id{i,1})=M;
end
end



%% label correlation
function L = label_correlation(Y)
    R = pdist2( Y'+eps, Y'+eps, 'cosine' );
    C = 1 - R;
    C = abs(C);
    L = diag(sum(C,2)) - C;
end


%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end

function A = gradL21(P)
num = size(P,1);
A = zeros(num,num);
for i=1:num
    temp = norm(P(i,:),2);
    if temp~=0
        A(i,i) = 1/temp;
    else
        A(i,i) = 0;
    end
end
end
