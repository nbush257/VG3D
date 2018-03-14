import neo
import neoUtils
import glob
import sys
import os
if __name__=='__main__':
    p = sys.argv[1]
    for fname in glob.glob(p):
        fid = neo.io.NixIO(fname)
        fname2 =os.path.splitext(fname)[0]+'_smooth_dat.h5'
        print(fname2)
        blk = fid.read_block()
        blk2 = neo.Block()
        for seg in blk.segments:
            seg2 = neo.Segment()
            sig_list =  [x.name for x in seg.analogsignals]
            for sig in seg.analogsignals:
                if sig.name.find('smoothed')!=-1:
                    seg2.analogsignals.append(sig)
            blk2.segments.append(seg2)
        fid2 = neo.io.NixIO(fname2)
        fid2.write_block(blk2)
        fid2.close()
