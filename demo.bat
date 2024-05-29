set BA_DIR=%YD_DIR%\Projects\BA\docs\20240411_SM-case\hall2_cams

: video_mornL:	hall-1 morning	from the left
python evaluate.py	--video %BA_DIR%/8.C_20220604-073015--20220604-090015.avi

: video_mornR:	hall-1 morning	from the right
python evaluate.py	--video %BA_DIR%/14.C_20220604-073015--20220604-090015.avi	--rotate

: video_dayL:	hall-1 day 	from the left
rem ==== Russian letters are not supported
::python evaluate.py	--video %BA_DIR%/8.Õ_ﬁ.¿¡«_ œœ_“3_“2_“1__20220603-153015--20220603-162015.avi
python evaluate.py	--video %BA_DIR%/8.C_20220603-153015--20220603-162015.avi
rem ==== The same file has been renamed
rem Add the parameter "out_video" to save results to the given video path
: python evaluate.py	--video %BA_DIR%/8.C_20220603-153015--20220603-162015.avi --out_video ./result_mask_coded.mp4

: video_dayR:	hall-1 day	from the right
python evaluate.py	--video %BA_DIR%/14.C_20220603-153015--20220603-162015.avi	--rotate
