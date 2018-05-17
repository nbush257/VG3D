function free_params = init_free_params(X,nfilts)
free_params = struct();

free_params.K=random('uniform',-1,1,nfilts,size(X,2));
free_params.tau=2;
free_params.A0=-1e2;
free_params.A1=-1e2;
free_params.a=1e-3;
free_params.I0=1;
free_params.scale=2e3;

