import glob
import pandas as pd
import os
import numpy as np
import neoUtils

recording_length = []
frame_length = []
root = []
p_save = r'C:\Users\guru\Box Sync\___hartmann_lab\papers\VG3D\summary_data_used'
for f in glob.glob(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\*.h5')):
    print('Working on {}'.format(os.path.basename(f)))
    blk = neoUtils.get_blk(f)
    M = neoUtils.get_var(blk)
    t = M.t_stop.magnitude
    recording_length.append(M.t_stop)
    root.append(neoUtils.get_root(blk,0))
    year = neoUtils.get_root(blk,0)[:4]
    if year =='2017':
        frames = int(np.round((t*1000)/(1000./500.)))
    else:
        frames = int(np.round((t*1000)/(1000./300.)))
    frame_length.append(frames)

df = pd.DataFrame()
df['id']=root
df['Time (s)']=recording_length
df['Number of Frames'] = frame_length
df.to_csv(os.path.join(p_save,'recording_lengths.csv'),index=False)
print('Average time: {}\nAverage number of frames: {}'.format(df['Time (s)'].mean(),
                                                              df['Number of Frames'].mean()))
