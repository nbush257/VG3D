function ignore_bool = unpack_arclength_bool(arclengths,arclength_used)
ignore_bool = zeros(size(arclengths.Proximal));
arclength_names = fieldnames(arclengths);
for ii=1:length(arclength_names)
    if ~strcmp(arclength_names{ii},lower(arclength_used))
        ignore_bool(arclengths.(arclength_names{ii}))=1;
    end
end
ignore_bool = logical(ignore_bool(:));