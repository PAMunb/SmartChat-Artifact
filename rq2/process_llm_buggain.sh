#!/bin/bash

for filename in output_B1_gpt4.1mini_*_gpt_buggain/result-dfa-impact/llmseeds_no_dynamic_no_static_buggain/B1-smartian-* output_B1_Llama3.3-70B_*_*_buggain/result-dfa-impact/llmseeds_no_dynamic_no_static_buggain/B1-smartian-*; do
    # Skip if no files match
    [[ -e "$filename" ]] || continue

    identifier=$(echo "$filename" | awk -F'/' '{print $1}' | awk -F'_' '{print $2"_"$3"-"$4}')

    temperature=$(echo "$filename" | awk -F'/' '{print $1}' | awk -F'_' '{print $4}')    

    # Extract final character of "B1-smartian-*"
    smartian_suffix=$(basename "$filename" | awk -F'-' '{print substr($NF, length($NF), 1)}')

    # Run the Python script and process the output
    result=$(python2.7 ../scripts/plot_b1_cve.py "$filename" | grep 60m | cut -c 6-)

    # Print the output in comma-separated format
    echo "$identifier,\"$temperature\",\"$result\",$smartian_suffix"
done
