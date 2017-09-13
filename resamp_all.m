function resamp_all(fname_in)
% This function saves the important data for E3D and neural, and samples
% everything at 1K
smooth_span = 11;
new_sr = 1000; %in Hz
nan_gap = 20; % in frames [333 ms for 300fps, 200ms for 500 fps
outname = [fname_in(1:end-12) '1K.mat'];
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
    for jj =1:size(filtvars.(varnames{ii}),2)
        filtvars.(varnames{ii})(:,jj) = InterpolateOverNans(filtvars.(varnames{ii})(:,jj),nan_gap);
    end
    idx_mask = isnan(filtvars.(varnames{ii}));
    
    filtvars.(varnames{ii}) = hampel(filtvars.(varnames{ii}));
    
    for jj = 1:size(filtvars.(varnames{ii}),2)
        filtvars.(varnames{ii})(:,jj) = smooth(filtvars.(varnames{ii})(:,jj),smooth_span,'sgolay');
    end
    filtvars.(varnames{ii})(idx_mask) = NaN;
end

%% resample vars
rawvars = vars;
for ii = 1:length(varnames)
    vars.(varnames{ii}) = resamp(vars.(varnames{ii}),spt{1},sr,frametimes,new_sr);
    filtvars.(varnames{ii}) = resamp(filtvars.(varnames{ii}),spt{1},sr,frametimes,new_sr);
end
for ii = 1:length(spt)
    [~,~,spt_upsamp{ii}] = resamp(rawvars.M(:,1),spt{ii},sr,frametimes,new_sr);
end

C_raw=C;
C = resamp(C,spt{1},sr,frametimes,new_sr);
%%
sp = spt_upsamp;
save(outname,'vars','rawvars','filtvars','sp','sr','C','C_raw','PT');


    
     