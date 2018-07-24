d = dir('C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\pillowX\best_smoothing_deriv\all_arclengths\*.mat');
population = struct();
inputs = [];
C = [];
for ii=1:length(d)
    load(d(ii).name,'output','X','cbool')
    cell_name = ['cell_' d(ii).name(1:10)];
    disp(cell_name)
    
    K = squeeze(output.Full.ppcbf_avg.k);
    nlfun = output.Full.ppcbf_avg.nlfun;
    fprs = output.Full.ppcbf_avg.fprs;
    fstruct = output.Full.ppcbf_avg.fstruct;
        
    population.(cell_name).K = K;
    population.(cell_name).nlfun = nlfun;
    population.(cell_name).fprs = fprs;
    population.(cell_name).fstruct = fstruct;
    inputs = [inputs;X];
    C = [C;cbool(:)];
        
end
save('population_data.mat','population','inputs','C','-v7.3')

