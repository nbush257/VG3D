function pillow_MID_sim_hist(fname)
%% ====  Set model parameters ==== %
disp(fname)
nkt = 1; % number of time bins to use for filter 
neye = 1;
neye_hist = 1;
b_nlin = 1;
ncos = 0;
ncos_hist = 2;
nkt_hist =10;
nfilts = 3; % number of filters to recover
nlin_basis_funcs = 5; % When fitting nonlinearity, how many basis functions to use?
n_sims = 1000;
K=10; %number of kfolds
%% ============== Set Up inputs ============== %

%strip middle derivatives, add spike history dimension, and set all non
%contact to 0
load(fname)
disp(fname)
disp(size(X))
nstim_dims = size(X,2);
y = y(:);


% == Set up temporal basis for representing history filters 
ktbasprs_hist.neye = neye_hist; % number of "identity"-like basis vectors
ktbasprs_hist.ncos = ncos_hist; % number of raised cosine basis vectors
ktbasprs_hist.kpeaks = [0 nkt_hist/2+4]; % location of 1st and last basis vector bump
ktbasprs_hist.b = b_nlin; % determines how nonlinearly to stretch basis (higher => more linear)
[ktbas_hist, ktbasis_hist] = makeBasis_StimKernel(ktbasprs_hist, nkt_hist); % make basis
yhist=zeros(size(y,1),size(ktbas_hist,2));
for ii =1:size(ktbas_hist,2)
    yhist(2:end,ii) = sameconv(y(1:end-1),ktbas_hist(:,ii));
end
X = [X yhist];

X(~cbool,:)=0;
y(~cbool)=0;
X = X./repmat(nanstd(X),[size(X,1),1]);
X(~cbool,:)=0;
X(any(isnan(X),2),:)=0;



% Get some values needed in the pillow code
Stim_tr = X;
sps_tr = y;


RefreshRate = 1000;
slen_tr = size(Stim_tr,1);   % length of training stimulus / spike train
nsp_tr = sum(sps_tr);   % number of spikes in training set

%% == 3. Set up temporal basis for stimulus filters (for ML / MID estimators)


% == Set up temporal basis for representing history filters 
ktbasprs.neye = neye; % number of "identity"-like basis vectors
ktbasprs.ncos = ncos; % number of raised cosine basis vectors
ktbasprs.kpeaks = [0 nkt/2+4]; % location of 1st and last basis vector bump
ktbasprs.b = b_nlin; % determines how nonlinearly to stretch basis (higher => more linear)
[ktbas, ktbasis] = makeBasis_StimKernel(ktbasprs, nkt); % make basis
% Compute STA and STC
[sta,stc,rawmu,rawcov] = simpleSTC(Stim_tr,sps_tr,nkt);  % compute STA and STC

% init param struct
pp0=struct();
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
yhat = zeros(size(y));
I_inj = zeros(size(y,1),nfilts);
idx=crossvalind('Kfold',size(X,1),K);
for kk = 1:K

    [ppcbf_temp,negLcbf] = fitLNP_multifiltsCBF_ML(pp0,Stim_tr(idx~=kk,:),sps_tr(idx~=kk),nfilts,fstructCBF);
    if kk==1
        ppcbf_all=ppcbf_temp;
    else
        ppcbf_all.k = cat(1,ppcbf_all.k,ppcbf_temp.k);
        ppcbf_all.kt = cat(1,ppcbf_all.kt,ppcbf_temp.kt);
        ppcbf_all.fprs = cat(2,ppcbf_all.fprs,ppcbf_temp.fprs);
    end
    temp_inj=[];
    
    for ii=1:nfilts
        temp_inj= [temp_inj,sameconv(X(idx==kk,:),squeeze(ppcbf_temp.k(:,:,ii)))];

    end
    I_inj(idx==kk,:) = temp_inj;
    yhat(idx==kk) = ppcbf_temp.nlfun(temp_inj);
    % need to average over all K?
end
[ppcbf,negLcbf] = fitLNP_multifiltsCBF_ML(pp0,Stim_tr,sps_tr,nfilts,fstructCBF);

ppcbf_avg = ppcbf_all;
ppcbf_avg.k = mean(ppcbf_avg.k,1);
ppcbf_avg.kt = mean(ppcbf_avg.kt,1);
ppcbf_avg.fprs = mean(ppcbf_avg.fprs,2);
ndims = size(ppcbf_all.k,2);
temp_inj=[]
for ii=1:nfilts
	temp_inj= [temp_inj,sameconv(X,squeeze(ppcbf_avg.k(:,:,ii)))];
end
yhat_avg= ppcbf_avg.nlfun(temp_inj);
%%  =============== Simulate ============ %
% get stim_currs
h_ = squeeze(ppcbf.k(:,end-size(ktbas_hist,2)+1:end,:));
h = ktbas_hist*h_;
nTimePts = size(X,1);
g = zeros(nTimePts+length(h),nfilts,n_sims);
ysim = zeros(nTimePts,n_sims); % initialize response vector (pad with zeros in order to convolve with post-spike filter)
rsim = zeros(nTimePts+length(h)-1,n_sims); % firing rate (output of nonlinearity)
hcurr = zeros(size(g));
h_len = size(h,1);

% get stim currents
I_inj=[];
for ii=1:nfilts
	I_inj= [I_inj,sameconv(X(:,1:nstim_dims),squeeze(ppcbf_avg.k(:,1:nstim_dims,ii)))];
end
stim_curr = I_inj;

%simulate runs
for runNum=1:n_sims
    fprintf('Run %i\n',runNum)
    g(:,:,runNum) = [stim_curr; zeros(h_len,nfilts)]; % injected current includes DC drive
    for t=1:nTimePts
        if ~cbool(t)
            continue
        end
        rsim(t,runNum) = ppcbf.nlfun(g(t,:,runNum));
        if rand<(1-exp(-rsim(t,runNum)/RefreshRate))
             ysim(t,runNum) = 1;
             g(t+1:t+h_len,:,runNum) = g(t+1:t+h_len,:,runNum) + flipud(h); % add post-spike filter
             hcurr(t+1:t+h_len,:,runNum) = hcurr(t+1:t+h_len,:,runNum) + flipud(h);
        end
    end
end

hcurr = hcurr(1:nTimePts,:);  % trim zero padding
rsim = rsim(1:nTimePts,:);  % trim zero padding


%%
r = smoothts(double(y)','g',length(y),16);
R = corrcoef(r(cbool),rsim(cbool));
fprintf('Correlation between rate and prediction: %f\n',R(1,2))
%%
save([fname(1:end-4) 'fitted_pillow_MID_sim.mat'])
end

