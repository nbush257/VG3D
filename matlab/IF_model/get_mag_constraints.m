function [c,ceq] = get_mag_constraints(x0)
% constrain all K vectors to be normal
nfilts=3;
K = reshape(x0(1:end-6),nfilts,[]);
N = zeros(nfilts,1);
for ii=1:nfilts
    N(ii) = norm(K(ii,:));
end
ceq=sum(abs(N)-1);
c=[];