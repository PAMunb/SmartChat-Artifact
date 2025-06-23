#!/bin/bash

# Arg1 : Time limit
# Arg2 : Source file
# Arg3 : Bytecode file
# Arg4 : ABI file
# Arg5 : Main contract name
# Arg6 : Optional argument to pass
# Arg7 : Seed dir
# Arg8 : Solcv
mkdir -p /home/test/output

python3 /home/test/tools/confuzzius/ConFuzzius/fuzzer/main.py \
    -r /home/test/output/log.json \
    -s $2 \
    -c $5 \
    --solc $8 \
    --evm byzantium \
    -t $1 > /home/test/output/stdout.txt 2>&1