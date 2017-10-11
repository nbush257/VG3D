from scipy.io import loadmat
from scipy.signal import butter,filtfilt
from scipy.io import wavfile
import numpy as np
import re
import os
import sys
import subprocess
import shutil
import glob




def create_audiofile(matfile,audiofile_path=None,audiofile_name=None,filter_tgl=False,low_cut=300.,high_cut=8000.):
    if audiofile_path==None:
        audiofile_path = os.path.split(matfile)[0]
    if audiofile_name==None:
        audiofile_name = gather_params_from_filename(matfile)[0] + '_neural.wav'

    dat = loadmat(matfile, variable_names=['ndata', 'sr', 'framesamps'])
    audio_data = dat['ndata'].squeeze()
    framesamps = dat['framesamps'].squeeze()
    sr = float(dat['sr'].squeeze())
    audio_data = audio_data[framesamps[0]:framesamps[-1]]

    if filter_tgl:
        Wn = [low_cut / (sr / 2), high_cut / (sr / 2)]
        b, a = butter(1, Wn, btype='bandpass')
        audio_data = filtfilt(b, a, audio_data.squeeze()).squeeze()

    scale = 1250  # hardcoded because that was the digital gain of the system
    audio_data /= scale
    wavfile.write(os.path.join(audiofile_path,audiofile_name), sr, audio_data)


def gather_params_from_filename(filename):
    ratnum = re.search('(?<=rat)\d{4}_\d{2}', filename).group()
    rec_date = re.search('(?<=_)[A-Z]{3}\d{2}(?=_VG)', filename).group()
    whisker_id = re.search('(?<=VG_)[A-Z]\d(?=_)', filename).group()
    trial_num = re.search('(?<=_)t\d{2}(?=_)', filename).group()
    root =  'rat{}_{}_VG_{}_{}'.format(ratnum,rec_date,whisker_id,trial_num)
    return(root,[ratnum,rec_date,whisker_id,trial_num])

def create_file_info_dict(matfile,p_vid_load,p_vid_save=None,audiofile_path=None,quality=1,vcodec='mpeg4',acodec='libvo_aacenc'):
    out = {}
    # gather data parameters
    root,rat_id_params = gather_params_from_filename(matfile)
    # set optionals
    if rat_id_params[0][:4]=='2017':
        FPS=500
    else:
        FPS=300

    if p_vid_save==None:
        p_vid_save = p_vid_load
    if audiofile_path==None:
        audiofile_path = os.path.split(matfile)[0]



    out['tv_name'] = root + '_Top'
    out['fv_name'] = root + '_Front'
    out['p_vid_save'] = p_vid_save
    out['p_vid_load'] = p_vid_load
    out['quality'] = str(quality)
    out['vcodec'] = vcodec
    out['FPS'] = str(FPS)
    out['audiofile_name'] = root + '_neural.wav'
    out['audiofile_path'] = audiofile_path
    out['acodec'] = acodec
    out['f_vid_save'] = root + '_neural_vid.mp4'
    return(out)


def FFMPEG_calls(file_info_dict):
    tv_name = file_info_dict['tv_name']
    fv_name = file_info_dict['fv_name']
    p_vid_load = file_info_dict['p_vid_load']
    p_vid_save = file_info_dict['p_vid_save']
    quality = file_info_dict['quality']
    vcodec = file_info_dict['vcodec']
    FPS = file_info_dict['FPS']
    audiofile_name = file_info_dict['audiofile_name']
    audiofile_path = file_info_dict['audiofile_path']
    acodec = file_info_dict['acodec']
    f_vid_save = file_info_dict['f_vid_save']

    fv_name = glob.glob(os.path.join(p_vid_load, fv_name+'*'))
    tv_name = glob.glob(os.path.join(p_vid_load, tv_name+'*'))

    if len(fv_name)==0 or len(tv_name)==0:
        print('Requested videos were not found')
        return -1
    else:
        fv_name = fv_name[0]
        tv_name = tv_name[0]


    subprocess.call(['ffmpeg',
                     '-r',str(FPS),
                     '-i',tv_name,
                     '-i',fv_name,
                     '-c:v','libx265',
                     '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]',
                     '-map', '[vid]',
                     '-crf','22',
                     '-y',
                     os.path.join(p_vid_save,f_vid_save)
                     ]
                    )

    subprocess.call(['ffmpeg',
                     '-r',str(FPS),
                     '-i',os.path.join(p_vid_save,f_vid_save),
                     '-i',os.path.join(audiofile_path,audiofile_name),
                     '-c:v','copy',
                     '-c:a',acodec,
                     '-y',
                     os.path.join(p_vid_save,'temp'+f_vid_save)
                     ]
                    )
    return 0
def main(argv=None):
    if argv is None:
        argv = sys.argv

    matfile = argv[1]
    p_vid_load = argv[2]
    p_vid_save=None
    if len(argv)>3:
        p_vid_save = argv[3]


    file_info_dict = create_file_info_dict(matfile, p_vid_load, p_vid_save=p_vid_save, audiofile_path=p_vid_save)
    f_vid_save = file_info_dict['f_vid_save']
    if os.path.isfile(os.path.join(p_vid_save,f_vid_save)):
        print('Target File Found, skipping...')
        return 0


    create_audiofile(matfile,audiofile_path=p_vid_save)

    ffmpeg_flag = FFMPEG_calls(file_info_dict)
    if ffmpeg_flag!=0:
        return 0
    os.remove(os.path.join(p_vid_save, f_vid_save))
    os.remove(os.path.join(file_info_dict['audiofile_path'], file_info_dict['audiofile_name']))
    shutil.move(os.path.join(p_vid_save,'temp'+f_vid_save),os.path.join(p_vid_save,f_vid_save))

if __name__=='__main__':
    sys.exit(main())