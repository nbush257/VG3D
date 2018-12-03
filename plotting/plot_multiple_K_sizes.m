%% plot the pearson correlation of a model with different numbers of linears spaces
function plot_multiple_K_sizes(fit_model,p_save);
load(fit_model)
[~,cellname] = fileparts(fname);
cellname = regexp(cellname,'\d{6}[A-E]\dc\d','match');
sigmas = [2,4,8,16,32,64,128,256,512];
models = fieldnames(output);
corrs= get_corrs(output,y,sigmas);
%%
figure;
set(groot, 'DefaultAxesColorOrder',gray(length(models)+2))
%semilogx(sigmas,corrs,'o-','linewidth',2)
subplot(121)
plot(sigmas,corrs,'o-','linewidth',2)
grid on
box off
axy(0,1);
line([16,16],[0,1],'color',[.7,.2,.2],'linewidth',2,'linestyle','--')
for ii =1:length(models)
    text(sigmas(end)+10,corrs(end,ii),sprintf('K = %d',K_dimensionality(ii)),'FontSize',12);
end

set(gca,'FontSize',12)
xlabel('\sigma (ms)')
ylabel('Correlation (R)')
title(cellname)
%%
subplot(122)
plot(corrs(4,:),'o--','linewidth',2)
xticklabels(string(K_dimensionality))
xlabel('K dimensionality')
ylabel('Correlation (R)')
title('\sigma = 16ms')
axy(0,1)
set(gca,'FontSize',12)
grid on
box off
print([p_save '\' cellname{1} '_Kcompare'],'-dpdf')

