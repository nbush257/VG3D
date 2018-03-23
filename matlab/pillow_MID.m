function pillow_MID(fname)
%% ====  Set model parameters ==== %
nkt = 75; % number of time bins to use for filter 
neye = 3;
b_nlin = 4;
ncos = 6;
nfilts = 3; % number of filters to recover
nlin_basis_funcs = 5; % When fitting nonlinearity, how many basis functions to use?
n_sims = 10;
%% ============== Set Up inputs ============== %

%strip middle derivatives, add spike history dimension, and set all non
%contact to 0
load(fname)
y = y(:);
X = [X,[0;y(1:end-1)]];
X(~cbool,:)=0;

% split into train/test sets?
test_idx = randsample(size(X,1),round(size(X,1)/10));
idx_bool = false(size(X,1),1);
idx_bool(test_idx)=1;

% Get some values needed in the pillow code
Stim_tr = X(~idx_bool,:);
sps_tr = y(~idx_bool);
Stim_tst = X(idx_bool,:);
sps_tst = y(idx_bool);
RefreshRate = 1000;
slen_tr = size(Stim_tr,1);   % length of training stimulus / spike train
slen_tst = size(Stim_tst,1); % length of test stimulus / spike train
nsp_tr = sum(sps_tr);   % number of spikes in training set
nsp_tst = sum(sps_tst); % number of spikes in test set

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
[ppcbf,negLcbf] = fitLNP_multifiltsCBF_ML(pp0,Stim_tr,sps_tr,nfilts,fstructCBF);

INJ = [];
for ii=1:nfilts
    INJ= [INJ,sameconv(X,squeeze(ppcbf.k(:,:,ii)))];
end
yhat = ppcbf.nlfun(INJ);

ndims = size(ppcbf.k,2);
for ii=1:ndims
    subplot(4,5,ii)
    plot(squeeze(ppcbf.k(:,ii,:)))
end
%%  =============== Simulate ============ %

% get stim_currs
h = squeeze(ppcbf.k(:,end,:));
nTimePts = size(X,1);
g = zeros(nTimePts+length(h),nfilts,n_sims);
ysim = zeros(nTimePts,n_sims); % initialize response vector (pad with zeros in order to convolve with post-spike filter)
rsim = zeros(nTimePts+length(h)-1,n_sims); % firing rate (output of nonlinearity)
hcurr = zeros(size(g));
h_len = size(h,1);

% get stim currents
stim_curr = zeros(size(X,1),nfilts);
for ii = 1:nfilts
    stim_curr(:,ii) = sameconv(X(:,1:end-1),squeeze(ppcbf.k(:,1:end-1,ii)));
end

%simulate runs
for runNum=1:n_sims
    fprintf('Run %i\n',runNum)
    g(:,:,runNum) = [stim_curr; zeros(length(h),nfilts,1)]; % injected current includes DC drive
    for t=1:nTimePts
        if ~cbool(t)
            continue
        end
        rsim(t,runNum) = ppcbf.nlfun(g(t,:,runNum));
        if rand<(1-exp(-rsim(t,runNum)/RefreshRate))
             ysim(t,runNum) = 1;
             g(t:t+h_len-1,:,runNum) = g(t:t+h_len-1,:,runNum) + flipud(h); % add post-spike filter
             hcurr(t:t+h_len-1,:,runNum) = hcurr(t:t+h_len-1,:,runNum) + flipud(h);
        end
    end
end

hcurr = hcurr(1:nTimePts,:);  % trim zero padding
rsim = rsim(1:nTimePts,:);  % trim zero padding


%%
r = smoothts(double(y)','g',length(y),16);
R = corrcoef(r(cbool),mean(rsim(cbool,:),2));
fprintf('Correlation between rate and prediction: %f\n',R(1,2))
%%
save([fname(1:end-4) 'fitted_pillow_MID.mat'])
end

