# bash script to run granger causality analysis on all fish for a given date (cli_granger_single_fish.py)
# Assuming structure of data is:

#       input_dir/
#           Fish_A/
#           Fish_B/
#           ...

# List paths processed
echo "Fish path to be processed:"

# List fish paths
input_dir=$1
fish_paths=$(find $input_dir -maxdepth 1 -type d -name "Fish_*")

# Check if alpha is provided
if [ -z "$2" ]
then
    alpha=0.01
else
    alpha=$2
fi

# List paths processed
echo "Fish paths to be processed:"
echo "$fish_paths"


# Run python cli script on each fish
for fish_path in $fish_paths
do
    echo "Processing $fish_path"
    python ../cli_plot_responsive_clusters.py --input_dir $fish_path --alpha $alpha
done


