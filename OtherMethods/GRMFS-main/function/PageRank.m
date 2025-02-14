function [c, RE] = PageRank(g, Sig)
%UNTITLED 此处提供此函数的摘要
%   此处提供详细说明        
tol = 1e-4;
maxit = 100;
damp = 0.85;
n = numnodes(g);
At = adjacency(g,  g.Edges.Weight);
d = full(sum(At, 1))';
snks = d == 0;
d(snks) = 1; % do 0/1 instead of 0/0 in first term of formula
cnew = Sig.';
for ii=1:maxit
    c = cnew;
    cnew = damp*At*(c./d) + damp/n*sum(c(snks)) + (1 - damp)/n;
    RE(ii) = norm(c - cnew, inf);
    if norm(c - cnew, inf) <= tol
        break;
    end
end
if ~(norm(c - cnew, inf) <= tol)
    warning(message('MATLAB:graphfun:centrality:PageRankNoConv'));
end
c = cnew;        
end