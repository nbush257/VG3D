function corrs = get_corrs(output,y,sigmas)

models = fieldnames(output);
corrs = nan(length(sigmas),length(models));
for ii = 1:length(sigmas)
    sigma = sigmas(ii);
    r = smoothts(double(y)','g',length(y),sigma);
    for jj = 1:length(models)
        model = models{jj};
        temp = corrcoef(output.(model).yhat,r);
        corrs(ii,jj) = temp(1,2);
        
    end
    
   
end