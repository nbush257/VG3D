mkdir not_matched
d = dir('*PROC.mat')
mkdir 'not_matched'
p_neural = 'E:\VG3D\neural\sorted\';
for ii = 1 :length(d)
    token = regexp(d(ii).name,'^rat\d{4}_\d{2}_[A-Z]{3}\d\d_VG_[A-Z]\d_t\d\d','match');token = token{1};
    disp(token)
    neural_filename = [p_neural token '_sorted.mat'];
    if ~exist(neural_filename,'file')
        warning('%s does not match',token)
        continue
    end
    load(neural_filename)
    load(d(ii).name,'C')
    if ~exist('ctrig')
        ctrig = [];
    end
    
    save(d(ii).name,'-append','frame*','ndata','nfilt','sp*','sr','time','ctrig')
    if length(C)~=length(framesamps)
        movefile(neural_filename,'./not_matched')
        movefile(d(ii).name,'./not_matched')
    end
    
    clearvars -except d ii p_neural
    
end
