function allT = batch_compare_K_size(spec)
%% function allT = batch_compare_K_size(spec)
% returns a table of all the correlations of neuron with different 
% dimensionality of the K weights.
%% 
file_list = dir(spec);
allT = table;

for ii = 1:length(file_list)
    file_name = file_list(ii).name;
    [~,cellname] = fileparts(file_name);
    cellname = regexp(cellname,'\d{6}[A-E]\dc\d','match');
    sigmas = [2,4,8,16,32,64,128,256,512];
    models = fieldnames(output);
    corrs= get_corrs(output,y,sigmas);  
        
    T = array2table(corrs,'VariableNames',sprintfc('K%d',K_dimensionality));
    T = [T table(repmat(cellname,length(corrs),1),'VariableNames',{'id'})];
    T = [T table(sigmas','VariableNames',{'sigma'})];

    allT = [allT;T];
end

    
