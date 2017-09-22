% unpack block
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
    