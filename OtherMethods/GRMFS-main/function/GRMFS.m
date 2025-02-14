function [FRank, RE] = GRMFS(data, targets, lammda, k)
[row, column] = size(data);
[~, label] = size(targets);
all_f = ones(row);
all_d = ones(row);
Edg=zeros(column, column);
Sig = [];
%%%Calculate the parameterized fuzzy similarity relation matrix%%% 
for i = 1 : column
    temp1 = data(:, i);
    sigma = std(temp1)/lammda;
    col = i;
    eval(['ssr' num2str(col) '=[];']);
    r = 1- abs(repmat(temp1, 1, row) - repmat(temp1.', row, 1));
    r(r<=sigma) = 0;
    eval(['ssr' num2str(col) '=r;']);
    all_f = min(all_f, r);
end
%%%Calculate the parameterized multi-neighborhood fuzzy decision%%%
for i = 1 : row
    for j = 1 : label
        index = find(targets(:, j) == 1);
        fd(i, j) = sum(all_f(i, index), 2)/sum(all_f(i, :));
    end
end
for i = 1 : label
    temp2 = fd(:, i);
    rd = 1 - abs(repmat(temp2, 1, row) - repmat(temp2.', row, 1));
    all_d = min(all_d, rd);
end
%%%Calculate the sample similarity relation matrix under label set%%%
for i = 1 : row
    temp3 = corr(repmat(fd(i, :).', 1, row), fd.', 'type', 'Pearson');
    sim(i, :) = temp3(i, :);
end
sim(logical(eye(size(sim)))) = 1;
sim(isnan(sim)) = 0;
%%%Calculate initial weights for vertices in the graph%%%
fprintf('Start assigning node weights......\n');
for i = 1 : row
    [K1, K2] = sort(sim(i, :), 'ascend'); 
    [~, temp4] = find(K1>=0); 
    if length(temp4) < k
        Differ = K2( temp4(:) );
    else
        Differ = K2( temp4(1: k) );     
    end
    DT{i,1} = Differ;
end
for i = 1 : column
    sig = 0;
    r = eval(['ssr' num2str(i)]);
    for j = 1 : row
        lower = sum(1 - r(j, DT{j, 1}), 2)/k;
        sig = sig + lower/row;
    end
    Sig = [Sig sig];
end
fprintf('The node weight assignment is complete......\n');
%%%Calculate initial weights for edges in the graph%%%
fprintf('Start assigning edge weights......\n');
for i = 1 : column
    rf1 = eval(['ssr' num2str(i)]);
    rf1d = min(all_d, rf1);
    for j = 1 : column
        rf2 = eval(['ssr' num2str(j)]);
        rf12 = min(rf1, rf2);
        rf12d = min(rf12, all_d);
        temp1 = sum(rf12, 2);
        temp2 = sum(rf12d, 2);
        temp3 = sum(rf1, 2);
        int = (temp1.*sum(rf1d, 2))./(temp2.*temp3);
        red = (temp3.*sum(rf2, 2))./(row*temp1);
        ass = (temp1.*sum(all_d, 2))./(temp2*row); 
        [ASS, INT, RED] = GRMFSentropy(ass, int, red);    
        Edg(i, j) = ASS + INT - RED;
    end
end
fprintf('The edge weight assignment is complete......\n');
%%%Start constructing graph%%%
Edg(logical(eye(size(Edg)))) = 0;
temp = Edg;
temp(temp<0) = 0;
[~, index2] = find(sum(temp, 1) == 0);
if (~isempty(index2))
    Edg = Standard(Edg);
    Edg(logical(eye(size(Edg)))) = 0;
    Edg = Edg./repmat(sum(Edg, 1), column, 1);
else
    Edg(logical(eye(size(Edg)))) = 0;
    Edg = Edg./repmat(sum(Edg, 1), column, 1);
end
[aa, bb] = find( Edg > 0 );
w1 = Edg(Edg>0);
G = digraph(aa,bb,w1);
%%%Assign weights to vertices%%%
[rank, RE]= PageRank(G, Sig);
[g, w2] = sort(rank, 'descend');
FRank=w2.';
end





