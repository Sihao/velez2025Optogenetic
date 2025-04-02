# Generate Subjects List

# Holland Brown

# Updated 2023-05-26
# Created 2023-05-08

# THIS SCRIPT: generates sublist.txt, a list of subject ID's necessary to run highres_reg.py, and
#              fnlist.txt, a list of the input filenames for reference

# ----------------------------------------------------------------------
# %% Setup
import glob



destination = "" # directory where you want sublist.txt to be created
datadir = "/highres_tiffs" # main data directory
sublist_fn = "sublist_test1.txt" # sublistFileName.txt

# Get paths to highres TIF files
highres_paths = glob.glob(f'{datadir}/*/*.tiff') # tiff file paths
subdirs=glob.glob(f'{datadir}/*') # subject directories

# %% Create sublist text file from directories
sublist=open(f'{destination}/{sublist_fn}','w')

# CREATE SUBS LIST FROM PATHS
for hdir in highres_paths:
    hr_ls = hdir.split('/')
    fn = hr_ls[-2]
    fn_ls = fn.split('_')
    sub = fn_ls[0] # get subject num
    runnum = fn_ls[-2] # get run num
    f=fn.strip("_HR") # filename without HR or extension
    # sublist.write(f'{sub}\n')
    sublist.write(f'{f}\n')
sublist.close()
# %%
