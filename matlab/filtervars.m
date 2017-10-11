function filtvar = filtervars(var,varargin)
%% function filtvar = filtervars(var)
% This function is the implementation of the filtering I am using for E3D.
% You can do different types of filtering by editing the inputs, or by
% adding sections.
% ==============================================
% INPUTS: var - a NxM array of a variable you want to filter. N is the
%   number of timepoints, M is the number of dimensions
% OUTPUTS: filtvar - the filtered version of the input you gave.
% ================================================
% NEB 20170913
% =============================================================
%  if you want to expand the functionlity at any point, use the option
%  parser and add LOCAL functions
% ================================================================
%% input handling
 % This is my first attempt at the options parser.
p = inputParser();
p.KeepUnmatched = true;
p.addRequired('var');
p.addOptional('hampel_tgl', true);
p.addOptional('sgolay_tgl', true);
p.addOptional('interp_nan_tgl',true);
p.addOptional('hampel_k', 3); % neighbors on either side to look at
p.addOptional('hampel_sigma', 3); % threshold of the hampel filter
p.addOptional('sgolay_span', 11); % number of beighbors on either side of the sgolay filter
p.addOptional('sgolay_degree', 2); % Default is 2-- Nick has not understood how this affects the data. Suggested not to change
p.addOptional('nan_gap', 15); % in samples
p.addOptional('mad_outlier_detection_tgl',true);
p.addOptional('mad_thresh',10); % numper to multiply times the IQR


parse(p,var,varargin{:});
vv = p.Results;
%%
filtvar = var;
%%
if vv.mad_outlier_detection_tgl %http://www.sciencedirect.com/science/article/pii/S0022103113000668
    med = nanmedian(filtvar);
    dev = (filtvar-med)./mad(filtvar);
    idx = any(abs(dev)>vv.mad_thresh,2);
    filtvar(idx,:)=nan;
    
end
    
        
if vv.interp_nan_tgl
    for jj =1:size(filtvar,2)
        filtvar(:,jj) = InterpolateOverNans(filtvar(:,jj),vv.nan_gap);
    end
end
% This prevents our filters from interpolating data points we dnot want.
% Specifically during the onset of contact
idx_mask = isnan(filtvar);

% Implements a hampel (thresholded median) filter.
if vv.hampel_tgl
    filtvar = hampel(filtvar,vv.hampel_k,vv.hampel_sigma);
end

% implements an sgolay filter on all the columns of the data
if vv.sgolay_tgl
    for jj = 1:size(filtvar,2)
        filtvar(:,jj) = smooth(filtvar(:,jj),vv.sgolay_span,'sgolay',vv.sgolay_degree);
    end
end
% Mask out to NaNs where the filters have added data points.
filtvar(idx_mask) = NaN;
end
