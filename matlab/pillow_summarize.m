% Pillow MID results
p_load = 'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\pillowX\';
d = dir([p_load '*MID.mat']);
kernel_sizes = [2,4,8,16,32,64,128,256,512];
DF = table()
for ii=1:length(d)
    load([p_load d(ii).name],'y','ysim','rsim','cbool')
    
    for jj=1:length(kernel_sizes)
        fprintf('Smoothing at %i\n',kernel_sizes(jj))
        r{jj} = smoothts(double(y)','g',length(y),kernel_sizes(jj));
        tempR = corrcoef(r{jj}(cbool),mean(rsim(cbool,:),2));
        R_conditional(jj) = tempR(1,2);
        tempR = corrcoef(r{jj}(cbool),mean(ysim(cbool,:),2));
        R_sim(jj) = tempR(1,2);
    end
    subDF = table(R_conditional',R_sim',repmat(d(ii).name(1:10),9,1),kernel_sizes','VariableNames',{'R_conditional','R_sim','id','kernel_sizes'});
    DF = [DF;subDF];
end
