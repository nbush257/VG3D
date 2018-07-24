function lambda = calc_pop_response(inputs,population)
cell_list = fieldnames(population);
lambda = zeros(size(inputs,1),length(cell_list));

for ii = 1:length(cell_list)
    K = population.(cell_list{ii}).K;
    nl = population.(cell_list{ii}).nlfun;
    fprs = population.(cell_list{ii}).fprs;
    fstruct = population.(cell_list{ii}).fstruct;
    lambda(:,ii) = evalCBFnlin(inputs*K,fstruct,fprs);
end