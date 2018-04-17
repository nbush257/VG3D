function [idx,CPout] = CPonWhisker(CP,xw2d,yw2d)
%% function [idx,CPout] = CPonWhisker2D(CP,w)
% ==============================================
% Takes a 2D CP and makes sure that the CP is exactly on a node of the
% whisker. Particularly helpful if the CP has been smoothed and may no
% longer correspond to a node on the whisker
% INPUTS:
%
% OUTPUTS
% ===============================================
% Nick Bush 2016_04_28
%%
CPout = CP;
idx = nan(length(xw2d),1);
for ii = 1:length(xw2d)
    if any(isnan(CP(ii,:)))
        continue
    end
    if isempty(xw2d{ii}) || length(xw2d{ii})<10
        continue
    end
    d = (CP(ii,1)-xw2d{ii}).^2+(CP(ii,2)-yw2d{ii}).^2; %calculate the squared distance because we don't actually care how far, just the min.
    [~,idx(ii)] = min(d);
    idx(ii) = round(idx(ii));
    if nargout == 2
        CPout(ii,:) = [xw2d{ii}(idx(ii)) yw2d{ii}(idx(ii))];
    end
end

