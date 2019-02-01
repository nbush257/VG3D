import worldGeometry
import collections
import neoUtils
import pandas as pd
import numpy as np
import glob
import os

p_load = os.path.join(os.environ['BOX_PATH'],r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO')
p_save = os.path.join(os.environ['BOX_PATH'],r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results')

DF = pd.DataFrame()
for f in glob.glob(os.path.join(p_load,'*.h5')):
    blk = neoUtils.get_blk(f)
    print('Working on {}'.format(f))
    num_units = len(blk.channel_indexes[-1].units)
    dir_idx,med_angle,projection_angle = worldGeometry.get_contact_direction(blk,False)
    for unit_num in range(num_units):
        id = neoUtils.get_root(blk,unit_num)
        df = pd.DataFrame()
        df['Direction_Group'] = dir_idx
        df['Group_Angle'] = [med_angle[x] for x in dir_idx]
        df['Contact_Angle'] = projection_angle
        df['id'] = id
        DF = DF.append(df)
DF.to_csv(os.path.join(p_save,'Direction_data.csv'),index=False)


master_dict=collections.OrderedDict()
for cell in DF.id.unique():
    df = DF[DF.id==cell]
    field = 'cell_{}'.format(cell)
    contact_angle = df['Contact_Angle'].as_matrix()
    direction_group = df['Direction_Group'].as_matrix()
    group_angle = df['Group_Angle'].as_matrix()
    master_dict[field] = {}
    master_dict[field]['contact_angle']=contact_angle
    master_dict[field]['direction_group']=direction_group
    master_dict[field]['group_angle']=group_angle
    master_dict[field]['contact_idx']=np.arange(len(group_angle))


