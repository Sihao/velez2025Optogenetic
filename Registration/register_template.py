import os

# Step 2: Upload template/atlas .tiff files to template directory and invoke -> get registration
antscmd = '/home/creagor/install/bin/antsRegistrationSyNQuick.sh'
atlas = 'T_AVG_HuCH2BGCaMP2-tg_ch0.tiff'
template = 'AvgTemplate_template0.tiff'
cmd = f'{antscmd} -d 3 -n 32 -e 42 -o Registration_ -f {atlas} -m {template}'

os.system(cmd)
