#!/bin/bash
# Function to display usage
usage() {
    echo "Usage: $0 <base_directory> <output_prefix>"
    echo "  base_directory: Directory to search in"
    echo "  output_prefix: Prefix for output files"
    echo "Example: $0 /path/to/data bug_b1"
    exit 1
}

# Function to extract identifier from filename
extract_identifier() {
    local filename="$1"
    echo "$filename" | awk -F'/' '{print $1}' | awk -F'_' '{print $2"_"$3"-"$4}'
}

# Function to extract smartian suffix
extract_smartian_suffix() {
    local filename="$1"
    basename "$filename" | awk -F'-' '{print substr($NF, length($NF), 1)}'
}

# Function to merge all model CSV files into one final CSV
merge_csv_files() {
    local output_prefix="$1"
    local final_csv="${output_prefix}/${output_prefix}.csv"
    
    echo "Merging CSV files..."
    > "$final_csv"
    
    csv_files=()
    while IFS= read -r -d '' file; do
        csv_files+=("$file")
    done < <(find "${output_prefix}" -name "*_detailed_results.csv" -print0)
    
    if [ ${#csv_files[@]} -eq 0 ]; then
        echo "Warning: No CSV files found to merge"
        return 1
    fi
    
    total_lines=0
    for csv_file in "${csv_files[@]}"; do
        if [ -f "$csv_file" ]; then
            lines_added=$(cat "$csv_file" | wc -l)
            cat "$csv_file" >> "$final_csv"
            total_lines=$((total_lines + lines_added))
        fi
    done
    
    if [ -f "$final_csv" ]; then
        echo "Merged CSV created: $final_csv ($total_lines lines)"
        return 0
    else
        echo "ERROR: Failed to create merged CSV"
        return 1
    fi
}

# Parameter validation
if [ $# -ne 2 ]; then
    echo "Error: 2 parameters required"
    usage
fi

base_dir="$1"
output_prefix="$2"

if [ -z "$base_dir" ] || [ -z "$output_prefix" ]; then
    echo "Error: Parameters cannot be empty"
    usage
fi

if [ ! -d "$base_dir" ]; then
    echo "Error: Directory '$base_dir' does not exist"
    exit 1
fi

if [ ! -d "$output_prefix" ]; then
    mkdir -p "$output_prefix" || { echo "Error: Cannot create output directory"; exit 1; }
fi

# Discover models
models=()
while IFS= read -r -d '' dir; do
    dirname=$(basename "$dir")
    if [[ "$dirname" =~ ^output_B1_.*_[0-9]+\.[0-9]+_10_4_ ]]; then
        model=$(echo "$dirname" | sed 's/output_B1_\(.*\)_[0-9]\+\.[0-9]\+_10_4_.*/\1/')
        if [[ ! " ${models[@]} " =~ " ${model} " ]] && [[ -n "$model" ]]; then
            models+=("$model")
        fi
    fi
done < <(find "$base_dir" -maxdepth 1 -type d -name "output_B1_*_*_*_*" -print0)

if [ ${#models[@]} -eq 0 ]; then
    echo "Error: No model directories found"
    exit 1
fi

echo "Processing ${#models[@]} models: ${models[*]}"

temps=(0.0 0.2 0.4 0.6 0.8 1.0 1.2)
created_csv_files=()

# Process each model
for model in "${models[@]}"; do
    echo "Processing $model..."
    
    csv_file="${output_prefix}/${model}_detailed_results.csv"
    created_csv_files+=("$csv_file")
    
    for temp in "${temps[@]}"; do
        dir=$(find "$base_dir" -maxdepth 1 -type d -name "output_B1_${model}_${temp}_10_4_*" | head -1)
        
        if [ -z "$dir" ] || [ ! -d "$dir" ]; then
            continue
        fi
        
        if [ ! -d "$dir/result-dfa-impact" ]; then
            continue
        fi
        
        file_count=0
        for filename in $dir/result-dfa-impact/*/B1-smartian-*; do
            [[ -e "$filename" ]] || continue
            file_count=$((file_count + 1))
            
            identifier=$(extract_identifier "$(basename "$dir")")
            smartian_suffix=$(extract_smartian_suffix "$filename")
            
            result=$(python2.7 ../scripts/plot_b1_cve.py "$filename" 2>/dev/null | grep 60m | cut -c 6-)
            
            if [ ! -z "$result" ]; then
                echo "B1_${model}-${temp},\"${temp}\",\"${result}\",${smartian_suffix}" >> "$csv_file"
            fi
        done
        
        # Create merged file for temperature
        merged_file="${output_prefix}/merged_all_${output_prefix}_${temp}"
        
        if ls $dir/result-dfa-impact/*/B1-smartian-* 1> /dev/null 2>&1; then
            if [ ! -f "$merged_file" ]; then
                touch "$merged_file"
            fi
            
            python2.7 ../scripts/plot_b1_cve.py $dir/result-dfa-impact/*/B1-smartian-* >> "$merged_file" 2>/dev/null
            echo "-----------------------------------" >> "$merged_file"
        fi
    done
done

echo "Individual processing complete"
merge_csv_files "$output_prefix"

echo "Summary: ${#models[@]} models processed across ${#temps[@]} temperatures"
