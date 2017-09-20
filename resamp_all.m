function resamp_all(fname_in)
% This function saves the important data for E3D and neural, and samples
% everything at the desired rate (suggested 1K, but flexible if needed:
sgolay_span = 11;
hampel_k = 3;
new_sr = 1000; %in Hz
nan_gap = 20; % in frames [333 ms for 300fps, 200ms for 500 fps
outname = [fname_in(1:end-12) '1K.mat'];
%% DO NOT EDIT BELOW THIS LINE IF YOU CHANGE PARAMETERS
% ===================================================
%% Load Data
load(fname_in,'M','F','PHIE','TH','Rcp','THcp','PHIcp','spt','sr','frame*','C','PT');
%%
vars.M = M;
vars.F = F;
vars.PHIE = PHIE;
vars.TH = TH;
vars.Rcp = Rcp;
vars.THcp = THcp;
vars.PHIcp = PHIcp;
varnames = fieldnames(vars);

%% get filt vars
filtvars = vars;


for ii = 1:length(varnames)
    filtvars.(varnames{ii}) = filtervars(vars.(varnames{ii}),...
        'nan_gap',nan_gap,...
        'sgolay_span',sgolay_span,...
        'hampel_k',hampel_k...
    );
end

%% resample vars
rawvars = vars;
for ii = 1:length(varnames)
    vars.(varnames{ii}) = resamp(vars.(varnames{ii}),spt{1},sr,frametimes,new_sr);
    filtvars.(varnames{ii}) = resamp(filtvars.(varnames{ii}),spt{1},sr,frametimes,new_sr);
end

% resample the spike times
for ii = 1:length(spt)
    [~,~,spt_upsamp{ii}] = resamp(rawvars.M(:,1),spt{ii},sr,frametimes,new_sr);
end
% resample the contact var
C_raw=C;
C = resamp(C,spt{1},sr,frametimes,new_sr);
%% save the output
sp = spt_upsamp;
save(outname,'vars','rawvars','filtvars','sp','sr','C','C_raw','PT');


    
     