#!/bin/bash
# run_loop.sh
# Runs the 100k_run_postprocessing.py script for DIR_ID values from 0 to 50

# # Activate your conda environment
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate ssp_la

# Loop from 0 to 50 (inclusive) NOTE: can change to whatever range you want
for i in {1..2}
do
    echo "==========================================="
    echo "Running postprocessing for DIR_ID = $i"
    echo "==========================================="
    
    # Run the Python script
    python 100k_run_postprocessing.py $i
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "⚠️  Error running DIR_ID=$i. Stopping loop."
        break
    fi

    echo "✅ Finished DIR_ID=$i"
    echo ""
done

echo "All runs completed."
