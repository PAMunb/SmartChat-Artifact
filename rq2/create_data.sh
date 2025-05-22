#!/bin/bash

# Check if parameters are provided, otherwise use defaults
model=${1:-"gpt4omini"}
base_dir=${2:-"."}  # Default to current directory if not specified
output_prefix=${3:-"bug_b1"}  # Allow customizing the output file prefix

echo "Using model: $model"
echo "Looking in base directory: $base_dir"
echo "Using output prefix: $output_prefix"

# Define an array of temperature values
temps=(0.0 0.2 0.4 0.6 0.8 1.0 1.2)

# Loop through each temperature value
for temp in "${temps[@]}"; do
  echo "Processing temperature $temp..."
  
  # Find the matching directory using a regex pattern with the model parameter
  dir=$(find "$base_dir" -maxdepth 1 -type d -name "output_B1_${model}_${temp}_10_4_*" | grep -v "_buggain$" | head -1)  
  
  # Check if directory was found
  if [ -z "$dir" ]; then
    echo "Warning: No matching directory found for $model with temperature $temp in $base_dir"
    continue
  fi
  
  # Run the command with the found directory
  python2.7 ../scripts/plot_b1_cve.py $dir/result-dfa-impact/llmseeds_no_dynamic_no_static/B1-smartian-* > ${output_prefix}/${model}_llm_only_$temp
  
  echo "Completed temperature $temp"
done




echo "All processing complete!"