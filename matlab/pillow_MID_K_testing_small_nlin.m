function pillow_MID_K_testing_small_nlin(fname)
% This script is looking at how the number of K vectors (the dimensionality
% of the projection space) affects the model results. It still uses 5 basis
% functions for the nonlinearity
%% ====  Set model parameters ==== %
nkt = 1; % number of time bins to use for filter
neye = 1;
b_nlin = 4;
ncos = 0;
% This is a a parameter that gets changed...:  nfilts = 3; % number of filters to recover
nlin_basis_funcs = 2; % When fitting nonlinearity, how many basis functions to use?
K=10; %number of kfolds
%% ============== Set Up inputs ============== %
load(fname)
disp(fname)
outname = [fname(1:end-4) 'fitted_pillow_MID_K_testing_small.mat'];
if exist(outname)
	fprintf('%s already fit',outname)
    return
end

disp(size(X))
y = y(:);
X(~cbool,:)=0;
y(~cbool)=0;
X = X./repmat(nanstd(X),[size(X,1),1]);
X(~cbool,:)=0;
X(any(isnan(X),2),:)=0;
output = struct();
K_dimensionality = [1,2,4,8,16]
%% =============== Run with and add K vectors ==================== %
for ii=1:length(K_dimensionality)
    nfilts=K_dimensionality(ii);
     
    Stim_tr = X;
    sps_tr = y;
    
    RefreshRate = 1000;
    slen_tr = size(Stim_tr,1);   % length of training stimulus / spike train
    nsp_tr = sum(sps_tr);   % number of spikes in training set
    
    %% == 3. Set up temporal basis for stimulus filters (for ML / MID estimators)
    
    % Compute STA and STC
    [sta,stc,rawmu,rawcov] = simpleSTC(Stim_tr,sps_tr,nkt);  % compute STA and STC
    
    % init param struct
    pp0=struct();
    
    % == Set up temporal basis for representing filters
    ktbasprs.neye = neye; % number of "identity"-like basis vectors
    ktbasprs.ncos = ncos; % number of raised cosine basis vectors
    ktbasprs.kpeaks = [0 nkt/2+4]; % location of 1st and last basis vector bump
    ktbasprs.b = b_nlin; % determines how nonlinearly to stretch basis (higher => more linear)
    [ktbas, ktbasis] = makeBasis_StimKernel(ktbasprs, nkt); % make basis
    filtprs_basis = (ktbas'*ktbas)\(ktbas'*sta);  % filter represented in new basis
    sta_basis = ktbas*filtprs_basis;
    
    % Insert filter basis into fitting struct
    pp0.k = sta_basis; % insert sta filter
    pp0.kt = filtprs_basis; % filter coefficients (in temporal basis)
    pp0.ktbas = ktbas; % temporal basis
    pp0.ktbasprs = ktbasprs;  % parameters that define the temporal basis
    pp0.RefreshRate = RefreshRate;
    pp0.mask=[];
    pp0.nlfun=@expfun;
    pp0.dc=0;
    pp0.model='LNP';
    pp0.ktype='linear';
    %% == 4. ML/MID 1:  ML estimator for LNP with CBF (cylindrical basis func) nonlinearity
    
    % Set parameters for cylindrical basis funcs (CBFs) and initialize fit
    fstructCBF.nfuncs = nlin_basis_funcs; % number of basis functions for nonlinearity
    fstructCBF.epprob = [.01, 0.99]; % cumulative probability outside outermost basis function peaks
    fstructCBF.nloutfun = @logexp1;  % log(1+exp(x)) % nonlinear stretching function
    fstructCBF.nlfuntype = 'cbf';
    
    % Fit the model (iteratively adding one filter at a time)
    % loop over crossvalidations
    yhat = zeros(size(sps_tr));
    idx=crossvalind('Kfold',size(Stim_tr,1),K);
    for kk = 1:K
        
        [ppcbf_temp,negLcbf] = fitLNP_multifiltsCBF_ML(pp0,Stim_tr(idx~=kk,:),sps_tr(idx~=kk),nfilts,fstructCBF);
        if kk==1
            ppcbf_all=ppcbf_temp;
        else
            ppcbf_all.k = cat(1,ppcbf_all.k,ppcbf_temp.k);
            ppcbf_all.kt = cat(1,ppcbf_all.kt,ppcbf_temp.kt);
            ppcbf_all.fprs = cat(2,ppcbf_all.fprs,ppcbf_temp.fprs);
        end
        temp_inj=[]
        for jj=1:nfilts
            temp_inj= [temp_inj,sameconv(Stim_tr(idx==kk,:),squeeze(ppcbf_temp.k(:,:,jj)))];
        end
        yhat(idx==kk) = ppcbf_temp.nlfun(temp_inj);
        % need to average over all K?
    end
    [ppcbf,negLcbf] = fitLNP_multifiltsCBF_ML(pp0,Stim_tr,sps_tr,nfilts,fstructCBF);
    
    ppcbf_avg = ppcbf_all;
    ppcbf_avg.k = mean(ppcbf_avg.k,1);
    ppcbf_avg.kt = mean(ppcbf_avg.kt,1);
    ppcbf_avg.fprs = mean(ppcbf_avg.fprs,2);
    ndims = size(ppcbf_all.k,2);
    % Run with average filter
    temp_inj=[]
    for jj=1:nfilts
        temp_inj= [temp_inj,sameconv(Stim_tr,squeeze(ppcbf_avg.k(:,:,jj)))];
    end
    yhat_avg= ppcbf_avg.nlfun(temp_inj);
    
    %%
    %r = smoothts(double(y)','g',length(y),16);
    %R = corrcoef(r(cbool),yhat(cbool));
    %fprintf('Correlation between rate and prediction: %f\n',R(1,2))
    %% ===================== Put data in struct ============= %
    K_name = sprintf('K_%02d',ii);
    output.(K_name).yhat = yhat;
    output.(K_name).yhat_avg = yhat_avg;
    output.(K_name).ppcbf = ppcbf;
    output.(K_name).ppcbf_avg = ppcbf_avg;
    output.(K_name).fstructCBF = fstructCBF;
    output.(K_name).pp0 = pp0;
    %output.(K_name).R = R;
end
%%
save(outname)
end
