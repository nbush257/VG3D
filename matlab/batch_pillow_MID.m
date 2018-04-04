function batch_pillow_MID(p_load)
d = dir([p_load '/*X.mat']);
for ii = 1:length(d)
    fname = d(ii).name;
    try
        pillow_MID([p_load '\' fname]);
    catch
        fprintf('fail at %i',ii)
    end
end
