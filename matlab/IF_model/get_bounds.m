function [lb,ub] = get_bounds(x0)
lb = ones(length(x0),1)*-Inf;
ub = ones(length(x0),1)*Inf;

lb(end-5) = 0;
ub(end-4) = 0;
ub(end-3) = 0;
lb(end-1) = 0;
lb(end) = 0;