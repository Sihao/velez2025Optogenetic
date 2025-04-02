# Quick script to rename granger files in a directory

# input_dir/
#   YYYYMMDD/
#       Fish_A/
#       Fish_B/
#       Fish_C/
#       ...

# List fish paths
input_dir=$1
fish_paths=$(find $input_dir -maxdepth 2 -type d -name "Fish_*")


# Run cli python script on each fish
for fish_path in $fish_paths
do
    echo "Processing $fish_path"
    python ../cli_rename_granger_files.py --input_path $fish_path
done