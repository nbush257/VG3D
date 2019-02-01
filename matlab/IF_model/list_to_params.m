function params = list_to_params(x0,nfilts)
params = struct();

params.K=x0(1:end-6);
params.K = reshape(params.K,nfilts,[]);
params.tau=x0(end-5);
params.A0=x0(end-4);
params.A1=x0(end-3);
params.a=x0(end-2);
params.I0=x0(end-1);
params.scale=x0(end);

