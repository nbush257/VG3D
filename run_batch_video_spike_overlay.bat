cd /D L:\Users\guru\Documents\hartmann_lab\proc\VG3D
SET MAT_PATH=C:\Users\guru\Desktop\test\neural\*.mat
SET VID_PATH=C:\Users\guru\Desktop\test\video
SET SAVE_PATH=C:\Users\guru\Desktop\test\save
echo %MAT_PATH%

for %%i in (%MAT_PATH%) do (
	echo %%i
	
	ipython overlay_spike_audio_video.py %%i %VID_PATH% %SAVE_PATH%
	)
cd /D C:\Users\guru\Desktop


