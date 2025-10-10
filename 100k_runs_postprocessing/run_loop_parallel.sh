#!/bin/bash
# run_loop.sh
# Runs the 100k_run_postprocessing_parallel.py script for multiple DIR_ID values

# Activate your conda environment (adjust path if needed)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate ssp_la

# Loop over the directory IDs you want to process
for i in {47..48}
do
    echo "==========================================="
    echo "Running postprocessing for DIR_ID = $i"
    echo "==========================================="

    # Run the Python script (use --dir-id argument)
    python 100k_run_postprocessing_parallel.py --dir-id "$i" --workers 8

    # Check exit code
    if [ $? -ne 0 ]; then
        echo "⚠️  Error running DIR_ID=$i. Stopping loop."
        break
    fi

    echo "✅ Finished DIR_ID=$i"
    echo ""
done

echo "All runs completed."
