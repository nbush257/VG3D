function x0=params_to_list(free_params)
x0=[];
names = fieldnames(free_params);

for ii = 1:length(names)
    name = names{ii};
    if name=='K'
        temp = free_params.(name);
        temp = temp(:);
        x0 = [x0;temp];
    else
        x0 = [x0;free_params.(name)(:)];
    end
end

            
            