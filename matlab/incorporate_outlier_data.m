function incorporate_outlier_data(p,raw_spec,outlier_spec)
% Use this to combine the python found outliers with the E3D elastica data.
d_raw = dir([p '\*' raw_spec '*.mat']);
d_outlier = dir([p '\*' outlier_spec '*.mat']);
cd(p)
for ii=73:length(d_raw)
    
    root = regexp(d_raw(ii).name,'rat\d{4}_\d{2}_[A-Z]{3}\d\d_VG_[A-Z]\d_t\d\d','match');
    disp(root)
    if ~isempty(strfind(d_raw(ii).name,outlier_spec))
        continue
    end
    idx = strfind({d_outlier.name},root);
    idx = find(~cellfun(@isempty,idx));
    if isempty(idx)
        continue
    end
    load(d_outlier(idx).name)
    outliers = outliers(:);
    use_flags = use_flags(:);
    save(d_raw(ii).name,'use_flags','outliers','-append')
    clear outliers use_flags
end
