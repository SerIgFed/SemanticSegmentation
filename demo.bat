set BA_DIR=C:\Users\serge\YandexDisk\Projects\BA\docs\20240411_SM-case\hall2_cams

: video-1
python evaluate_images.py		--video %BA_DIR%/8.C_20220604-073015--20220604-090015.avi

: video-2
rem ==== Russian letters are not supported
:::python demo.py			--video %BA_DIR%/8.Õ_ﬁ.¿¡«_ œœ_“3_“2_“1__20220603-153015--20220603-162015.avi
:python evaluate_images.py		--video %BA_DIR%/8.C_20220603-153015--20220603-162015.avi --out_video ./result_mask_coded.mp4
 python evaluate_images.py		--video %BA_DIR%/8.C_20220603-153015--20220603-162015.avi
rem ==== The same file has been ranamed

: video-3
python evaluate_images.py --rotate	--video %BA_DIR%/14.C_20220603-153015--20220603-162015.avi

: video-4
python evaluate_images.py --rotate	--video %BA_DIR%/14.C_20220604-073015--20220604-090015.avi
