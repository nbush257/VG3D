% unpack block
concat_TGL=1;
for ii= 1:length(block.segments)
    seg = block.segments{ii};
    for jj= 1:length(seg.spiketrains)
        sp{ii,jj} = seg.spiketrains{jj}.times;
    end
    M{ii} = seg.analogsignals{1}.signal;
    F{ii} = seg.analogsignals{2}.signal;
    PHIE{ii} = seg.analogsignals{3}.signal;
    TH{ii} = seg.analogsignals{4}.signal;
    Rcp{ii} = seg.analogsignals{5}.signal;
    THcp{ii} = seg.analogsignals{6}.signal;
    PHIcp{ii} = seg.analogsignals{7}.signal;
    
    cc{ii} = [seg.epochs{1}.times(:) seg.epochs{1}.times(:)+seg.epochs{1}.durations(:)];
end
clear ii jj seg

if concat_TGL
    last_time = length(M{1}(:,1));
    for ii = 2:length(cc)
        cc{ii} = cc{ii}+last_time;
        for jj = 1:size(sp,2)
            sp{ii,jj} = sp{ii,jj}+last_time;
        end
        
        last_time = last_time+ length(M{ii}(:,1));
        
    end
    
    M = vertcat(M{:});
    F = vertcat(F{:});
    PHIcp = vertcat(PHIcp{:});
    PHIE = vertcat(PHIE{:});
    Rcp = vertcat(Rcp{:});
    TH = vertcat(TH{:});
    THcp = vertcat(THcp{:});
    cc = vertcat(cc{:});
    for ii = 1:size(sp,2)
        temp{ii} = [sp{:,ii}];
    end
    sp = temp;clear temp
    for ii = 1:length(sp)
        spbool{ii} = zeros(size(M,1),1);
        spbool{ii}(sp{ii})=1;
    end
end

    
    
        
    