function resamp_all2D(fname_in)
% This function saves the important data for E3D and neural, and samples
% everything at the desired rate (suggested 1K, but flexible if needed:
sgolay_span = 7;
hampel_k = 5;
new_sr = 1000; %in Hz
nan_gap = 10; % in frames [333 ms for 300fps, 200ms for 500 fps
mad_thresh = 10;
outname = [fname_in(1:end-12) '1K.mat'];
%% DO NOT EDIT BELOW THIS LINE IF YOU CHANGE PARAMETERS
% ===================================================
%% Load Data
load(fname_in,'M','F*','TH','spt','sr','frame*','C','PT','spikes','outliers','use_flags','CP*');
%%
vars.M = M;
vars.FX = FX;
vars.FY = FY;
vars.TH = TH;
vars.CP = CP;
vars.CPm = CPm;
varnames = fieldnames(vars);

%% get filt vars
filtvars = vars;

for ii = 1:length(varnames)
    filtvars.(varnames{ii}) = filtervars(vars.(varnames{ii}),...
        'nan_gap',nan_gap,...
        'sgolay_span',sgolay_span,...
        'hampel_k',hampel_k,...
        'mad_thresh',mad_thresh,...
        'py_outlier_detect',true,...
        'py_outliers',outliers...
    );
end

%% resample vars
rawvars = vars;
rawfiltvars = filtvars;
for ii = 1:length(varnames)
    vars.(varnames{ii}) = resamp(vars.(varnames{ii}),spt{1},sr,frametimes,new_sr);
    filtvars.(varnames{ii}) = resamp(filtvars.(varnames{ii}),spt{1},sr,frametimes,new_sr);
end

% resample the spike times
for ii = 1:length(spt)
    [~,~,spt_upsamp{ii}] = resamp(rawvars.M(:,1),spt{ii},sr,frametimes,new_sr);
end
% resample the contact vars
C_raw=C;
C = resamp(C,spt{1},sr,frametimes,new_sr);
use_flags = resamp(use_flags(:),spt{1},sr,frametimes,new_sr);
%% save the output
sp = spt_upsamp;
save(outname,'vars','rawvars','filtvars','rawfiltvars','sp','sr','C','C_raw','PT','spikes','outliers','use_flags');


    
     