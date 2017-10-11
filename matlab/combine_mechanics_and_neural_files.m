d = dir('*2017_02*PROC.mat')
for ii = 1 :length(d)
    token = regexp(d(ii).name,'^rat\d{4}_\d{2}_[A-Z]{3}\d\d_VG_[A-Z]\d_t\d\d','match');token = token{1};
    disp(token)
    neural_filename = [token '_sorted.mat'];
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
    
    clearvars -except d ii
    
end
