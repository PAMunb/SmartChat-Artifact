#!/bin/bash

SCRIPTDIR=$(dirname $0)
OUTDIR=$(realpath $SCRIPTDIR/../output)
ROOT=$(realpath $SCRIPTDIR/..)
BENCHDIR=$(realpath $SCRIPTDIR/../benchmarks)
B2_DIR="$BENCHDIR/B1"
TEST_DIR="$BENCHDIR/b1_test"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <iterN>"
    exit
fi

# Get list of directories in B5_test
TEST_DIRS=$(find $TEST_DIR -mindepth 1 -maxdepth 1 -type d)

# Check if results directory exists
if ls $OUTDIR/result-B1-compare 1> /dev/null 2>&1; then
    echo "$OUTDIR/result-B1-compare exists, please remove it."
    exit 1
fi

# Create results directory
mkdir -p $OUTDIR/result-B1-compare

# Process each directory in B2_test
for test_dir in $TEST_DIRS; do
    # Get directory name
    dir_name=$(basename $test_dir)

    RELATIVE_PATH=$(realpath --relative-to="$BENCHDIR" "$test_dir")
    rm $B2_DIR
    ln -sf $RELATIVE_PATH $B2_DIR
    touch $BENCHDIR/x
    rm $BENCHDIR/x
    touch $BENCHDIR/x


    pushd $ROOT
    docker build .
    popd
    
    # Create results directory for this test
    mkdir -p $OUTDIR/result-B1-compare/$dir_name
    
    # Run experiment for the specified number of iterations
    for i in $(seq $1); do
        python $SCRIPTDIR/run_experiment.py B1 smartian 3600 "--uselllmseeds --nosdfa --noddfa --withbuggain"
    done
    # Check if output directories exist
    if ls $OUTDIR/B1-smartian-* 1> /dev/null 2>&1; then
       # Move results to the appropriate directory
       mv $OUTDIR/B1-smartian-* $OUTDIR/result-B1-compare/$dir_name/
    fi    
done
