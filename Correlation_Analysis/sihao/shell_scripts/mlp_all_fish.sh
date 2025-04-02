# bash script to run granger causality analysis on all fish for a given date (cli_granger_single_fish.py)
# Assuming structure of data is:

# input_dir/
#   YYYYMMDD/
#       fish_A/
#       fish_B/
#       fish_C/
#       ...
# Usage: bash svm_all_fish.sh <input_dir> <overwrite>
# List fish paths
input_dir=$1
fish_paths=$(find $input_dir -maxdepth 2 -type d -name "Fish_*")

echo "Running granger causality analysis on all fish in $input_dir"

# List paths processed
echo "Fish paths to be processed:"
echo "$fish_paths"

# Run python cli script on each fish
for fish_path in $fish_paths
do
    echo "Processing $fish_path"
    # If overwrite flag is set, pass it to the python script
    if [ -z "$2" ]
    then
        python ../cli_MLP_single_fish.py --input_path $fish_path
    else
        python ../cli_MLP_single_fish.py --input_path $fish_path --overwrite
    fi
done

