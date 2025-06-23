#!/bin/bash

SCRIPTDIR=$(dirname $0)
OUTDIR=$(realpath $SCRIPTDIR/../output)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <iterN>"
    exit
fi

if ls $OUTDIR/B2-confuzius-* 1> /dev/null 2>&1; then
    echo "$OUTDIR/B2-confuzius-* exists, please remove it."
    exit 1
fi

if ls $OUTDIR/result 1> /dev/null 2>&1; then
    echo "$OUTDIR/result exists, please remove it."
    exit 1
fi

mkdir -p $OUTDIR/result

for i in $(seq $1); do
    python $SCRIPTDIR/run_experiment.py B2 confuzius 60
done
mkdir -p $OUTDIR/result/confuzius
mv $OUTDIR/B2-confuzius-* $OUTDIR/result/confuzius

