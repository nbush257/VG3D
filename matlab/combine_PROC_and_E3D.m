
d = dir('*DATA*.mat');
for ii = 1:length(d)
    load(d(ii).name,'PT')
    disp(PT.TAG)
    PT_d = dir(['Vg*' PT.TAG '*.mat']);
    for jj = 1:length(PT_d)
        load(PT_d(jj).name)
    end
    outfile = [PT.TAG '_E3D_PROC.mat'];

    save(outfile,'-regexp','^(?!(PT_d|d|ii|outfile|jj)$).')
    clearvars -except d ii
end