# bash script to run granger causality analysis on all fish (cli_granger_single_fish.py)
# Assuming structure of data is:

# input_dir/
#   YYYYMMDD/
#       Fish_A/
#       Fish_B/
#       Fish_C/
#       ...

# List fish paths
input_dir=$1
alpha=$2
fish_paths=$(find $input_dir -maxdepth 2 -type d -name "Fish_*")

echo "Generating responsive cluster plots for all fish in $input_dir"

# List paths processed
echo "Fish paths to be processed:"
echo "$fish_paths"

# Run python cli script on each fish
for fish_path in $fish_paths
do
    echo "Processing $fish_path"
    python ../cli_plot_responsive_clusters.py --input_dir $fish_path --alpha $alpha
done

