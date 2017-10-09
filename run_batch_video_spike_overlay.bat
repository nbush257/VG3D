SET MAT_PATH=K:\VG3D\_E3D_PROC\_deflection_trials\*sorted.mat
SET VID_PATH=D:\VG3D\COMPRESSED
SET SAVE_PATH=K:\VG3D\neural_overlay_vids
echo %MAT_PATH%

for %%i in (%MAT_PATH%) do (
	echo %%i
	
	ipython overlay_spike_audio_video.py %%i %VID_PATH% %SAVE_PATH%
	)



