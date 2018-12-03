function compare_nlin(p_load,prefix,p_save)
sigmas = [2,4,8,16,32,64,128,256,512];
%% get teh data
d = dir([p_load '/*' prefix '*K*.mat']);
if length(d)~=2
    error('Two files with K not found')
end
load([p_load '/' d(1).name])
corrs_big = get_corrs(output,y,sigmas);
nfuncs_big = ppcbf.fstruct.nfuncs;
load([p_load '/' d(2).name])
corrs_small = get_corrs(output,y,sigmas);
nfuncs_small = ppcbf.fstruct.nfuncs;
models = fieldnames(output);
%% plot the data
longfig;
set(groot, 'DefaultAxesColorOrder',gray(length(models)+2))
%% All vals for sigma and K; small nonlinearity
subplot(131)
plot(sigmas,corrs_small,'*-','linewidth',2)
ax = gca;
ax.YGrid = 'on';box off
axy(0,1);
axx(0,150)

line([16,16],[0,1],'color',[.7,.2,.2],'linewidth',2,'linestyle','--')
% for ii =1:length(models)
%     text(sigmas(6)+10,corrs_small(6,ii),sprintf('K = %d',K_dimensionality(ii)),'FontSize',12);
% end
temp = {};
for ii=1:length(models)
    temp = [temp sprintf('K = %d',K_dimensionality(ii))];
end
legend(temp)
set(gca,'FontSize',12)
xlabel('\sigma (ms)')
ylabel('Correlation (R)')
title(sprintf('%d parameter nonlinearity',nfuncs_small))
%% All vals for sigma and K; small nonlinearity
subplot(132)
plot(sigmas,corrs_big,'o-','linewidth',2)
ax = gca;
ax.YGrid = 'on';box off
axy(0,1);
axx(0,150)
line([16,16],[0,1],'color',[.7,.2,.2],'linewidth',2,'linestyle','--')
% for ii =1:length(models)
%     text(sigmas(6)+10,corrs_big(6,ii),sprintf('K = %d',K_dimensionality(ii)),'FontSize',12);
% end
legend(temp)
set(gca,'FontSize',12)
xlabel('\sigma (ms)')
ylabel('Correlation (R)')
title(sprintf('%d parameter nonlinearity',nfuncs_big))
%% 
subplot(133)
plot(corrs_small(4,:),'*-','color',[.7,.2,.2],'linewidth',2)
hold on
plot(corrs_big(4,:),'o--','linewidth',2,'color',[.7,.2,.2])

legend({sprintf('%d param nlin',nfuncs_small),sprintf('%d param nlin',nfuncs_big)})
xticks([1:length(models)]);
xticklabels(string(K_dimensionality))

xlabel('K dimensionality')
ylabel('Correlation (R)')
title('\sigma = 16ms')
axy(0,1)
set(gca,'FontSize',12)
ax = gca;
ax.YGrid = 'on';
box off
%%
suptitle(prefix)
%%
print([p_save '\' cellname{1} '_Kcompare_big_small'],'-dpdf')

