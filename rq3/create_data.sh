#!/bin/bash

# Find all directories matching the patterns
for dir in output_b2_*/result-*/*/ output_b2_*/result/*/; do
    if [[ -d "$dir" ]]; then
        # Extract the type from the main directory name
        type=$(echo "$dir" | sed 's|output_b2_||; s|_llm||; s|_5times.*||')
        
        # Extract model info from the final directory name
        basename_dir=$(basename "$dir")
        
        # Generate output name based on pattern
        if [[ "$type" == "smartian" ]]; then
            output_name="smartian"
        elif [[ "$type" == "confuzzius" ]]; then
            output_name="confuzzius"
        else
            # Extract model name from basename - everything after B2*_ and before _0.4
            if [[ "$basename_dir" =~ B2.*_([^_]+)_0\.4 ]]; then
                model="${BASH_REMATCH[1]}"
            else
                # Fallback: extract between underscores
                model=$(echo "$basename_dir" | sed 's|.*_\([^_]*\)_0\.4.*|\1|')
            fi
            
            output_name="${type}_${model}_0.4"
        fi
        
        echo "Processing: $dir -> $output_name"
        
        python3.8 ../../scripts/plot_b2_bug.py "$dir"/* > "${output_name}.bug"
        python3.8 ../../scripts/plot_b2_cov.py "$dir"/* > "${output_name}.cov"
    fi
done
