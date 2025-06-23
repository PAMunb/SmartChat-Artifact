import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def vargha_delaney_a12(x, y):
    """
    Calculate Vargha and Delaney's A12 effect size.
    
    A12 represents the probability that a randomly selected observation 
    from group x will be larger than a randomly selected observation from group y.
    
    Args:
        x: First group (e.g., model performance)
        y: Second group (e.g., baseline performance)
        
    Returns:
        A12 value between 0 and 1:
        - 0.5: No difference (random chance)
        - > 0.5: Group x tends to be larger than group y
        - < 0.5: Group x tends to be smaller than group y
        - 1.0: All values in x are larger than all values in y
        - 0.0: All values in x are smaller than all values in y
    """
    x = np.array(x)
    y = np.array(y)
    
    m = len(x)
    n = len(y)
    
    # Count how many times each x[i] is greater than each y[j]
    greater_count = 0
    equal_count = 0
    
    for xi in x:
        for yj in y:
            if xi > yj:
                greater_count += 1
            elif xi == yj:
                equal_count += 1
    
    # A12 formula: (greater_count + 0.5 * equal_count) / (m * n)
    a12 = (greater_count + 0.5 * equal_count) / (m * n)
    
    return a12

def interpret_a12_effect_size(a12):
    """
    Interpret A12 effect size according to Vargha & Delaney (2000) guidelines.
    
    Returns:
        Tuple of (magnitude, interpretation)
    """
    # Convert to absolute difference from 0.5 for magnitude classification
    abs_diff = abs(a12 - 0.5)
    
    if abs_diff < 0.06:  # 0.44 < A12 < 0.56
        magnitude = "Negligible"
    elif abs_diff < 0.14:  # 0.36 < A12 < 0.44 or 0.56 < A12 < 0.64
        magnitude = "Small"
    elif abs_diff < 0.21:  # 0.29 < A12 < 0.36 or 0.64 < A12 < 0.71
        magnitude = "Medium"
    else:  # A12 < 0.29 or A12 > 0.71
        magnitude = "Large"
    
    # Interpretation
    if a12 > 0.5:
        interpretation = f"Group 1 superior ({a12:.3f} probability)"
    elif a12 < 0.5:
        interpretation = f"Group 2 superior ({1-a12:.3f} probability)"
    else:
        interpretation = "No difference (equal performance)"
    
    return magnitude, interpretation

def load_and_clean_data(file_path):
    """
    Load and clean the experiment data from a CSV file.
    The file has columns for model_temp, temperature, bugs found, and run number.
    Assumes no header in the CSV file.
    
    Expected format: B1_gpt4.1mini-0.0,"0.0","50.0",1
    """
    # Load data with explicit column names (no header)
    column_names = ['model_temp', 'temperature', 'bugs_found', 'run']
    
    try:
        df = pd.read_csv(file_path, names=column_names, header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        raise
    
    print(f"Loaded {len(df)} rows from {file_path}")
    
    # Clean up quoted values and convert to proper types
    def clean_value(value):
        """Remove quotes and convert to appropriate type"""
        if isinstance(value, str):
            cleaned = value.strip().replace('"', '').replace("'", "")
            try:
                return float(cleaned)
            except ValueError:
                return cleaned
        return value
    
    # Apply cleaning to all columns
    for col in df.columns:
        df[col] = df[col].apply(clean_value)
    
    # Convert numeric columns
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['bugs_found'] = pd.to_numeric(df['bugs_found'], errors='coerce')
    df['run'] = pd.to_numeric(df['run'], errors='coerce')
    
    # Extract model name from model_temp column
    def extract_model_name(model_temp_str):
        if pd.isna(model_temp_str) or not isinstance(model_temp_str, str):
            return "Unknown"
        
        # Remove the B1_ prefix if present
        if model_temp_str.startswith('B1_'):
            model_temp_str = model_temp_str[3:]
        
        # Split by '-' and take all parts except the last one (which should be temperature)
        parts = model_temp_str.split('-')
        if len(parts) > 1:
            # Rejoin all parts except the last one
            model_name = '-'.join(parts[:-1])
        else:
            model_name = parts[0]
        
        return model_name
    
    df['model'] = df['model_temp'].apply(extract_model_name)
    
    # Check for any parsing issues
    print(f"\nData after cleaning:")
    print(f"Models found: {sorted(df['model'].unique())}")
    print(f"Temperature values: {sorted(df['temperature'].unique())}")
    print(f"Temperature data type: {df['temperature'].dtype}")
    print(f"Bugs found data type: {df['bugs_found'].dtype}")
    
    # Check for any NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        print(f"\nWarning: Found NaN values:")
        print(nan_counts[nan_counts > 0])
    
    # Remove any rows with NaN values in critical columns
    initial_rows = len(df)
    df = df.dropna(subset=['model', 'temperature', 'bugs_found'])
    final_rows = len(df)
    
    if initial_rows != final_rows:
        print(f"Removed {initial_rows - final_rows} rows with missing values")
        
    return df

def load_baseline_data(baseline_file_path):
    """
    Load baseline data from CSV file.
    Expected format: method_name,bugs_found,run
    Example: dfa,44.0,1
    """
    try:
        # Use same column names as main data
        baseline_column_names = ['method_name', 'bugs_found', 'run']
        baseline_df_raw = pd.read_csv(baseline_file_path, names=baseline_column_names, header=None)
        
        # Clean the baseline data the same way
        def clean_baseline_value(value):
            if isinstance(value, str):
                cleaned = value.strip().replace('"', '').replace("'", "")
                try:
                    return float(cleaned)
                except ValueError:
                    return cleaned
            return value
        
        # Apply cleaning to baseline data
        for col in baseline_df_raw.columns:
            baseline_df_raw[col] = baseline_df_raw[col].apply(clean_baseline_value)
        
        # Convert bugs_found to numeric
        baseline_df_raw['bugs_found'] = pd.to_numeric(baseline_df_raw['bugs_found'], errors='coerce')
        baseline_df_raw['run'] = pd.to_numeric(baseline_df_raw['run'], errors='coerce')
        
        # Extract just the bugs_found values
        baseline_data = baseline_df_raw['bugs_found'].dropna()
        baseline_method_name = baseline_df_raw['method_name'].iloc[0] if len(baseline_df_raw) > 0 else "Baseline"
        
        print(f"Baseline method: {baseline_method_name}")
        print(f"Baseline data: {len(baseline_data)} measurements")
        print(f"Baseline statistics:")
        print(f"  Mean: {baseline_data.mean():.2f}")
        print(f"  Median: {baseline_data.median():.2f}")
        print(f"  Std Dev: {baseline_data.std():.2f}")
        print(f"  Range: {baseline_data.min():.1f} - {baseline_data.max():.1f}")
        print(f"  Values: {baseline_data.tolist()}")
        
        return baseline_data, baseline_method_name
        
    except Exception as e:
        print(f"Error loading baseline data: {e}")
        print("Expected format: method_name,bugs_found,run")
        print("Example: dfa,44.0,1")
        raise

def compare_with_baseline_mannwhitney(df, baseline_data, baseline_name="Baseline"):
    """
    Compare each model and temperature combination with baseline data using Mann-Whitney U test
    and Vargha-Delaney A12 effect size.
    
    Args:
        df: DataFrame containing experiment data
        baseline_data: Series or array of baseline measurements
        baseline_name: Name for the baseline method
        
    Returns:
        DataFrame with comparison results
    """
    models = sorted(df['model'].unique())
    temp_values = sorted(df['temperature'].unique())
    
    results = []
    
    print(f"\nComparing each model-temperature combination with {baseline_name} using Mann-Whitney U test...")
    print(f"Baseline sample: {len(baseline_data)} measurements, mean={np.mean(baseline_data):.2f}")
    print(f"Effect size: Vargha-Delaney A12 (probability of superiority)")
    
    for model in models:
        for temp in temp_values:
            model_bugs = df[(df['model'] == model) & (df['temperature'] == temp)]['bugs_found']
            
            if len(model_bugs) > 0:
                # Perform Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(model_bugs, baseline_data, alternative='two-sided')
                
                # Calculate Vargha-Delaney A12 effect size
                a12 = vargha_delaney_a12(model_bugs, baseline_data)
                a12_magnitude, a12_interpretation = interpret_a12_effect_size(a12)
                
                # Calculate descriptive statistics
                mean_bugs = model_bugs.mean()
                baseline_mean = np.mean(baseline_data)
                diff_from_baseline = mean_bugs - baseline_mean
                percent_diff = (diff_from_baseline / baseline_mean) * 100
                
                # Determine significance and performance
                is_significant = "Yes" if p_value < 0.05 else "No"
                if is_significant == "Yes":
                    performance = "Better" if mean_bugs > baseline_mean else "Worse"
                else:
                    performance = "No significant difference"
                
                results.append({
                    'Model': model,
                    'Temp': temp,
                    'Mean_Bugs': round(mean_bugs, 1),
                    'Baseline_Mean': round(baseline_mean, 1),
                    'Difference': round(diff_from_baseline, 1),
                    'Percent_Diff': round(percent_diff, 1),
                    'A12': round(a12, 3),
                    'A12_Magnitude': a12_magnitude,
                    'p_value': round(p_value, 4),
                    'Significant': is_significant,
                    'Performance': performance
                })
                
                print(f"  {model} @ temp {temp}: mean={mean_bugs:.1f}, A12={a12:.3f} ({a12_magnitude}), p={p_value:.4f}, {performance}")
    
    return pd.DataFrame(results)

def create_baseline_comparison_plots(df, baseline_data, baseline_name="Baseline"):
    """
    Create visualizations comparing all models to the baseline.
    """
    models = sorted(df['model'].unique())
    temp_values = sorted(df['temperature'].unique())
    baseline_mean = np.mean(baseline_data)
    
    # 1. Boxplot comparison with baseline line
    plt.figure(figsize=(15, 8))
    ax = sns.boxplot(x='temperature', y='bugs_found', hue='model', data=df)
    plt.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2, 
                label=f'{baseline_name} (mean: {baseline_mean:.1f})')
    
    plt.title(f'Model Performance vs {baseline_name} Across Temperatures', fontsize=16)
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Number of Bugs Found', fontsize=12)
    
    # Enhance legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles + [plt.Line2D([0], [0], color='red', linestyle='--')], 
              labels + [f'{baseline_name} (mean: {baseline_mean:.1f})'], 
              title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('baseline_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. A12 effect size heatmap
    comparison_data = []
    for model in models:
        for temp in temp_values:
            model_bugs = df[(df['model'] == model) & (df['temperature'] == temp)]['bugs_found']
            if len(model_bugs) > 0:
                a12 = vargha_delaney_a12(model_bugs, baseline_data)
                comparison_data.append({
                    'Model': model,
                    'Temperature': temp,
                    'A12': a12
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    heatmap_data = comparison_df.pivot_table(
        values='A12', 
        index='Model', 
        columns='Temperature'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f',
               linewidths=0.5, cbar_kws={'label': 'Vargha-Delaney A12 Effect Size'},
               vmin=0, vmax=1)
    plt.title(f'Vargha-Delaney A12 Effect Size vs {baseline_name}\n(0.5=no difference, >0.5=model better, <0.5=baseline better)', fontsize=14)
    plt.tight_layout()
    plt.savefig('baseline_a12_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Bar plot showing mean differences from baseline
    comparison_data_extended = []
    for model in models:
        for temp in temp_values:
            model_bugs = df[(df['model'] == model) & (df['temperature'] == temp)]['bugs_found']
            if len(model_bugs) > 0:
                mean_bugs = model_bugs.mean()
                comparison_data_extended.append({
                    'Model': model,
                    'Temperature': temp,
                    'Mean Bugs Found': mean_bugs,
                    'Difference from Baseline': mean_bugs - baseline_mean
                })
    
    comparison_df_extended = pd.DataFrame(comparison_data_extended)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=comparison_df_extended, x="Model", y="Difference from Baseline", 
                hue="Temperature", palette="viridis")
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.title(f'Difference in Bugs Found Compared to {baseline_name} (mean: {baseline_mean:.1f})', fontsize=16)
    plt.ylabel('Difference in Bugs Found', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('baseline_difference_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results_df, baseline_name, baseline_mean):
    """
    Generate a text summary report of the analysis using A12 effect sizes.
    """
    print("\n" + "="*80)
    print("MANN-WHITNEY U TEST WITH VARGHA-DELANEY A12 EFFECT SIZE ANALYSIS")
    print("="*80)
    
    print(f"\nBaseline: {baseline_name} (mean: {baseline_mean:.2f} bugs)")
    print(f"Total model-temperature combinations tested: {len(results_df)}")
    print(f"Effect size measure: Vargha-Delaney A12 (probability of superiority)")
    
    # Count significant results
    significant_better = len(results_df[(results_df['Significant'] == 'Yes') & 
                                       (results_df['Performance'] == 'Better')])
    significant_worse = len(results_df[(results_df['Significant'] == 'Yes') & 
                                      (results_df['Performance'] == 'Worse')])
    not_significant = len(results_df[results_df['Significant'] == 'No'])
    
    print(f"\nResults Summary:")
    print(f"  • Significantly better than baseline: {significant_better}")
    print(f"  • Significantly worse than baseline: {significant_worse}")
    print(f"  • No significant difference: {not_significant}")
    
    # Best performing combinations
    print(f"\nTop 5 Best Performing Model-Temperature Combinations:")
    top_performers = results_df.nlargest(5, 'Mean_Bugs')[
        ['Model', 'Temp', 'Mean_Bugs', 'A12', 'A12_Magnitude', 'p_value', 'Performance']
    ]
    print(top_performers.to_string(index=False))
    
    # A12 magnitude distribution
    print(f"\nVargha-Delaney A12 Effect Size Distribution:")
    a12_counts = results_df['A12_Magnitude'].value_counts()
    for magnitude, count in a12_counts.items():
        print(f"  • {magnitude}: {count} combinations")
    
    # Statistical significance summary with A12
    if significant_better > 0 or significant_worse > 0:
        print(f"\nStatistically Significant Results (p < 0.05):")
        sig_results = results_df[results_df['Significant'] == 'Yes'][
            ['Model', 'Temp', 'Mean_Bugs', 'Difference', 'A12', 'A12_Magnitude', 'p_value', 'Performance']
        ].sort_values('p_value')
        print(sig_results.to_string(index=False))
    
    # A12 interpretation guide
    print(f"\nVargha-Delaney A12 Interpretation Guide:")
    print(f"  • A12 = 0.5: No difference (random chance)")
    print(f"  • A12 > 0.5: Model tends to outperform baseline")
    print(f"  • A12 < 0.5: Baseline tends to outperform model")
    print(f"  • A12 = 1.0: Model always outperforms baseline")
    print(f"  • A12 = 0.0: Baseline always outperforms model")
    print(f"  • Magnitude thresholds: Negligible (<0.56), Small (<0.64), Medium (<0.71), Large (≥0.71)")

def run_analysis(file_path, baseline_data_file):
    """
    Main function to run the Mann-Whitney U test baseline comparison analysis with A12 effect size.
    """
    print("="*80)
    print("MANN-WHITNEY U TEST WITH VARGHA-DELANEY A12 EFFECT SIZE")
    print("="*80)
    
    print("Loading and processing data...")
    df = load_and_clean_data(file_path)
    
    # Display basic data info
    print(f"\n" + "="*50)
    print("DATA OVERVIEW")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"Models found: {list(df['model'].unique())}")
    print(f"Temperature values: {sorted(df['temperature'].unique())}")
    
    print(f"\nRuns per model-temperature combination:")
    run_counts = df.groupby(['model', 'temperature']).size().reset_index(name='runs')
    run_counts_pivot = run_counts.pivot(index='model', columns='temperature', values='runs')
    print(run_counts_pivot.to_string())
    
    # Validate data consistency
    print(f"\nData validation:")
    expected_runs_per_combo = 5
    inconsistent_runs = run_counts[run_counts['runs'] != expected_runs_per_combo]
    if len(inconsistent_runs) > 0:
        print(f"WARNING: Found model-temperature combinations with != {expected_runs_per_combo} runs:")
        print(inconsistent_runs.to_string(index=False))
    else:
        print(f"✓ All model-temperature combinations have exactly {expected_runs_per_combo} runs")
    
    # Load baseline data
    print(f"\n" + "="*50)
    print("LOADING BASELINE DATA")
    print("="*50)
    baseline_data, baseline_method_name = load_baseline_data(baseline_data_file)
    
    # Perform Mann-Whitney U tests with A12
    print(f"\n" + "="*50)
    print("MANN-WHITNEY U TEST WITH A12 EFFECT SIZE ANALYSIS")
    print("="*50)
    results = compare_with_baseline_mannwhitney(df, baseline_data, baseline_method_name)
    
    # Display results in compact format
    print(f"\nMann-Whitney U Test Results with Vargha-Delaney A12:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 140)
    pd.set_option('display.max_colwidth', 15)
    print(results.to_string(index=False))
    
    # Save detailed results to CSV
    output_filename = "mannwhitney_a12_baseline_comparison_results.csv"
    results.to_csv(output_filename, index=False)
    print(f"\nDetailed results saved to '{output_filename}'")
    
    # Generate summary statistics
    print(f"\nSummary Statistics by Model and Temperature:")
    summary = df.groupby(['model', 'temperature'])['bugs_found'].agg([
        'count', 'mean', 'std', 'min', 'median', 'max'
    ]).round(2).reset_index()
    print(summary.to_string(index=False))
    
    # Save summary to CSV
    summary.to_csv("summary_statistics.csv", index=False)
    print(f"Summary statistics saved to 'summary_statistics.csv'")
    
    # Generate summary report
    generate_summary_report(results, baseline_method_name, baseline_data.mean())
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    create_baseline_comparison_plots(df, baseline_data, baseline_method_name)
    print("Visualizations saved:")
    print("  • baseline_comparison_boxplot.png")
    print("  • baseline_a12_heatmap.png")
    print("  • baseline_difference_barplot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare model performance against baseline using Mann-Whitney U test with Vargha-Delaney A12 effect size.')
    parser.add_argument('file', help='CSV file containing experiment data')
    parser.add_argument('baseline_data', help='CSV file containing baseline sample data (format: method_name,bugs_found,run)')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    run_analysis(args.file, args.baseline_data)
