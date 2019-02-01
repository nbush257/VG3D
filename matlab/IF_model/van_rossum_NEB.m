function d = van_rossum_NEB(y1,y2,tau)
t = [1:tau*10];
%% kernel on first train
spt1 = find(y1);
Y1 = zeros(1,length(y1));
kernel = [1 exp(-t/tau)];
if ~isempty(spt1)
    for ii =1:length(spt1)
        starts = spt1(ii);
        stops = spt1(ii)+t(end);
        if stops>length(Y1)
            stops=length(Y1);
            k_end = stops-starts+1;
        else
            k_end = length(kernel);
        end
        Y1(starts:stops) =  Y1(starts:stops)+kernel(1:k_end);
    end
end

%% Kernel on second train
spt2 = find(y2);
Y2 = zeros(1,length(y2));
kernel = [1 exp(-t/tau)];
if ~isempty(spt2)
    for ii =1:length(spt2)
        starts = spt2(ii);
        stops = spt2(ii)+t(end);
        if stops>length(Y1)
            stops=length(Y1);
            k_end = stops-starts+1;
        else
            k_end = length(kernel);
        end
        Y2(starts:stops) =  Y2(starts:stops)+kernel(1:k_end);
    end
end
%% calc cost
d = (1/tau).*sum((Y1-Y2).^2);

