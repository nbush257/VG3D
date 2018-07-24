function reshape_arclength_model_output(fname_in)
load(fname_in)
%%
try
    output = rmfield(output,'inputs');
end
try
    output = rmfield(output,'arclength');
end

names = fieldnames(output);
%%
for ii=1:length(names)
    
    idx = regexp(names{ii},'(Distal)|(Proximal)|(all)');
    output.(names{ii}).inputs=names{ii}(1:idx-2);
    output.(names{ii}).arclengths=names{ii}(idx:end);
    try
        output.(names{ii}) = rmfield(output.(names{ii}),'iarclength');
    end
end
%%
save(fname_in,'-append','output')
