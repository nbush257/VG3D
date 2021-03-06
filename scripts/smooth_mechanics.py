import neoUtils
import neo
import sys
import numpy as np
import os
if __name__=='__main__':
    fname = sys.argv[1]
    new_filename = os.path.splitext(fname)[0]+'_smoothed.h5'
    print('Old file:{}'.format(fname))
    print('new file:{}'.format(new_filename))
    fid = neo.io.NixIO(fname,mode='ro')
    fid2 = neo.io.NixIO(new_filename)
    blk = fid.read_block()
    root = neoUtils.get_root(blk,0)[:-2]
    print('Working on {}'.format(root))

    smoothing_windows = range(5,101,10)

    for seg in blk.segments:
        sig_list =  [x.name for x in seg.analogsignals]
        print(sig_list)
        for sig in seg.analogsignals:
            if sig.name.find('smoothed')!=-1:
                continue
            if '{}_smoothed'.format(sig.name) in sig_list:
                continue


            print('working on {}'.format(sig.name))
            sig_smoothed = np.array([neoUtils.smooth_var_lowess(sig, x) for x in smoothing_windows])
            sig_smoothed = np.moveaxis(sig_smoothed,[0,1,2],[2,0,1])
            sig_smoothed = neo.AnalogSignal(sig_smoothed,
                                             units=sig.units,
                                             sampling_rate=sig.sampling_rate,
                                             name='{}_smoothed'.format(sig.name))
            seg.analogsignals.append(sig_smoothed)
    fid2.write_block(blk)
    fid.close()
    fid2.close()

