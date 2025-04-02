import os, glob

# Step 3: Upload ROI .csv files named for each fish to template directory and invoke -> get ROIs in atlas space
antscmd = '/home/creagor/install/bin/antsApplyTransformsToPoints'
a1_tf = '-t [Registration_0GenericAffine.mat, 1]'
w1_tf = '-t Registration_1InverseWarp.nii.gz'
names = sorted([fn.split('.')[0] for fn in glob.glob('*.csv')])
for name in names:
    a0 = glob.glob(f'*{name}*Affine.txt')[0]
    w0 = glob.glob(f'*{name}*InverseWarp.nii.gz')[0]
    cmd = f'{antscmd} -d 3 -i {name}.csv -o {name}_Warped.csv -t [{a0}, 1] -t {w0} {a1_tf} {w1_tf}'
    os.system(cmd)
