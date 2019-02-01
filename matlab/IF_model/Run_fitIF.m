function Run_fitIF(fname)
PC_tgl=true;
load(fname)
npcs = 9;
[p_save,basename] =fileparts(fname);
prefix = basename(1:10);
outname = [p_save '/'  prefix  'fit_matlab_IF.mat'];
if exist(outname)
	error('File already exists')
end
parpool('local',20)

if PC_tgl
    Xpc = zeros(size(X,1),npcs);
    temp = pca(X(cbool,:)');
    if npcs>size(temp,2)
        npcs=size(temp,2);
    end
    Xpc(cbool,1:npcs) = temp(:,1:npcs);
    X = Xpc;
end
X = X./repmat(nanstd(X(cbool,:)),size(X,1),1);
X(isnan(X))=0;

const_params = init_const_params();
free_params = init_free_params(X,const_params.nfilts);
x0 = params_to_list(free_params);

f = @(x0)fit_IF(x0,X,y,const_params,cbool,true);

[lb,ub] = get_bounds(x0);
options = optimoptions('patternsearch','Display','iter','CompletePoll','off', 'UseParallel',true);

[x,fval] = patternsearch(f,x0,[],[],[],[],lb,ub,[],options);
%%
fit_params = list_to_params(x,const_params.nfilts);

[~,yhat,Vhat] = fit_IF(x,X,y,const_params,cbool,false);
save(outname)
