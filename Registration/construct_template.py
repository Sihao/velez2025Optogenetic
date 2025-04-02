import os

# Step 1: Invoke from directory containing all raw HR images in .nii format -> get template
antscmd = '/home/creagor/install/bin/antsMultivariateTemplateConstruction.sh'
cmd = f'{antscmd} -d 3 -b 1 -c 2 -g 0.1 -i 10 -j 32 -n 0 -r 1 -y 0 -o AvgTemplate_ *.nii'

os.system(cmd)
