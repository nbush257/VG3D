function Run_fitIF(fname)
PC_tgl=false;
load(fname)
npcs = 5;
[p_save,basename] =fileparts(fname);
prefix = basename(1:10);
outname = [p_save '/'  prefix  'fit_matlab_IF.mat'];
if PC_tgl
    Xpc = zeros(size(X,1),npcs);
    temp = pca(X(cbool,:)');
    Xpc(cbool,:) = temp(:,1:npcs);
    X = Xpc;
end
X = X./nanstd(X);


const_params = init_const_params();
free_params = init_free_params(X,const_params.nfilts);
x0 = params_to_list(free_params);

f = @(x0)fit_IF(x0,X,y,const_params,cbool);

[lb,ub] = get_bounds(x0);
options = optimoptions('patternsearch','Display','iter','CompletePoll','off', 'UseParallel',true);

[x,fval] = patternsearch(f,x0,[],[],[],[],lb,ub,[],options);
%%
fit_params = list_to_params(x,const_params.nfilts);

[~,yhat,Vhat] = fit_IF(x,X,y,const_params,cbool);
save(outname)