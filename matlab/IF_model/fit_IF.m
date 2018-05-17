function [cost,yhat,V] = fit_IF(x0,X,y,const_params,C)
if iscolumn(y)
    y = y';
end

free_params = list_to_params(x0,const_params.nfilts);
I_inj = LOCAL_calcI(X,free_params);
cc = convertContact(C);
idx = randi(size(cc,1),50,1);
% while isempty(D)
%     for ii=1:length(idx)
%         [~,yhat] = runIF(I_inj(cc(idx(ii),1):cc(idx(ii),2)),free_params,const_params);
%         if sum(yhat)~=0
%             d = vanRossumPW([y(cc(idx(ii),1):cc(idx(ii),2));yhat],const_params.van_rossum_tau);
%             d = d(1,2);
%             D = [D d];
%         end
% d = vanRossumPW([y(cc(idx(ii),1):cc(idx(ii),2));yhat],const_params.van_rossum_tau);
% d = d(1,2);
% D = [D d];
%
%     end
[V,yhat] = runIF(I_inj,free_params,const_params);
cost = van_rossum_NEB(y,yhat,const_params.van_rossum_tau);

end




function I_inj = LOCAL_calcI(X,free_params)
I_inj = free_params.K*X';
I_inj = sum(abs(I_inj),1);
I_inj = (I_inj*free_params.I0)./(abs(I_inj)+free_params.I0);
I_inj = I_inj.*free_params.scale;
end