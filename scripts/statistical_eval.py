import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_and_clean_data(file_path):
    """
    Load and clean the experiment data from a CSV file.
    The file has columns for model, temperature, bugs found, and run number.
    Assumes no header in the CSV file.
    """
    # Load data with explicit column names (no header)
    column_names = ['model_temp', 'temperature', 'bugs_found', 'run']
    df = pd.read_csv(file_path, names=column_names, header=None)
    
    # Handle case where parsing might have resulted in a single column
    if len(df.columns) == 1:
        # Split the single column into multiple columns
        df = pd.DataFrame([row.split(',') for row in df.iloc[:, 0]], 
                         columns=column_names)
    
    # Clean up values and convert to proper types
    # Handle both string and numeric types safely
    def clean_and_convert(column):
        if pd.api.types.is_string_dtype(column):
            return column.str.replace('"', '').astype(float)
        else:
            return column.astype(float)
    
    # Apply cleaning function to numeric columns
    df['temperature'] = clean_and_convert(df['temperature'])
    df['bugs_found'] = clean_and_convert(df['bugs_found'])
    
    # Extract model name from model_temp column
    df['model'] = df['model_temp'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)
    
    return df

def perform_mann_whitney_test(df, model1, model2, temp_value):
    """
    Perform Mann-Whitney U test to compare bug counts between two models at a specific temperature.
    
    Args:
        df: DataFrame containing experiment data
        model1, model2: Names of models to compare
        temp_value: Temperature value to filter on
        
    Returns:
        U-statistic, p-value, and sample sizes for each group
    """
    # Filter data for the specified models and temperature
    group1 = df[(df['model'] == model1) & (df['temperature'] == temp_value)]['bugs_found']
    group2 = df[(df['model'] == model2) & (df['temperature'] == temp_value)]['bugs_found']
    
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    return u_stat, p_value, len(group1), len(group2)

def compare_with_traditional_method(df, traditional_bugs=44):
    """
    Compare each model and temperature combination with the traditional method result.
    Traditional method is represented by a fixed value (44 bugs).
    
    Args:
        df: DataFrame containing experiment data
        traditional_bugs: Number of bugs found by the traditional method
        
    Returns:
        DataFrame with comparison results
    """
    # Get unique models and temperature values
    models = sorted(df['model'].unique())
    temp_values = sorted(df['temperature'].unique())
    
    # Create a constant series to represent traditional method
    # It needs to be the same length as the largest group for comparison
    max_runs = df.groupby(['model', 'temperature']).size().max()
    traditional_method = pd.Series([traditional_bugs] * max_runs)
    
    # Store results
    results = []
    
    # Compare each model and temperature with traditional method
    for model in models:
        for temp in temp_values:
            # Get the bugs found for this model and temperature
            model_bugs = df[(df['model'] == model) & (df['temperature'] == temp)]['bugs_found']
            
            if len(model_bugs) > 0:
                # Match the length to perform the test
                current_traditional = traditional_method[:len(model_bugs)]
                
                # Perform Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(model_bugs, current_traditional, alternative='two-sided')
                
                # Calculate mean and other statistics
                mean_bugs = model_bugs.mean()
                diff_from_traditional = mean_bugs - traditional_bugs
                percent_diff = (diff_from_traditional / traditional_bugs) * 100
                
                # Determine significance
                is_significant = "Yes" if p_value < 0.05 else "No"
                better_than_traditional = "Better" if mean_bugs > traditional_bugs else "Worse"
                
                results.append({
                    'Model': model,
                    'Temperature': temp,
                    'Mean Bugs Found': round(mean_bugs, 2),
                    'Traditional Method Bugs': traditional_bugs,
                    'Difference': round(diff_from_traditional, 2),
                    'Percent Difference': round(percent_diff, 2),
                    'U-statistic': u_stat,
                    'p-value': round(p_value, 6),
                    'Sample Size': len(model_bugs),
                    'Significant (α=0.05)': is_significant,
                    'Performance': better_than_traditional if is_significant == "Yes" else "No significant difference"
                })
    
    return pd.DataFrame(results)

def compare_all_combinations(df):
    """
    Perform Mann-Whitney U tests for all model and temperature combinations.
    Each model is compared with every other model at each temperature.
    """
    # Get unique models and temperature values
    models = sorted(df['model'].unique())
    temp_values = sorted(df['temperature'].unique())
    
    # Store results
    results = []
    
    # Perform tests for all combinations
    for temp in temp_values:
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:  # Compare with models we haven't compared yet
                u_stat, p_value, n1, n2 = perform_mann_whitney_test(df, model1, model2, temp)
                
                # Determine significance
                is_significant = "Yes" if p_value < 0.05 else "No"
                
                # Get mean values for both groups for easier interpretation
                mean1 = df[(df['model'] == model1) & (df['temperature'] == temp)]['bugs_found'].mean()
                mean2 = df[(df['model'] == model2) & (df['temperature'] == temp)]['bugs_found'].mean()
                
                results.append({
                    'Temperature': temp,
                    'Model 1': model1,
                    'Model 2': model2,
                    'Mean Bugs (Model 1)': round(mean1, 2),
                    'Mean Bugs (Model 2)': round(mean2, 2),
                    'U-statistic': u_stat,
                    'p-value': round(p_value, 6),
                    'Sample Size 1': n1,
                    'Sample Size 2': n2,
                    'Significant (α=0.05)': is_significant
                })
    
    return pd.DataFrame(results)

def visualize_comparison_by_temperature(df, traditional_bugs=44):
    """
    Create visualizations comparing all models at each temperature.
    Generates a separate figure for each temperature setting.
    Includes a horizontal line for the traditional method benchmark.
    """
    # Get unique temperature values
    temp_values = sorted(df['temperature'].unique())
    
    # Create a visualization for each temperature
    for temp in temp_values:
        # Filter data for the current temperature
        temp_data = df[df['temperature'] == temp]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create boxplot comparing models at this temperature
        ax = sns.boxplot(x='model', y='bugs_found', data=temp_data)
        
        # Add horizontal line for traditional method
        plt.axhline(y=traditional_bugs, color='r', linestyle='--', label='Traditional Method (44 bugs)')
        
        plt.title(f'Comparison of Bugs Found at Temperature {temp}')
        plt.xlabel('Model')
        plt.ylabel('Number of Bugs Found')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Annotate the traditional method line
        plt.text(len(temp_data['model'].unique()) - 1, traditional_bugs + 1, 'Traditional Method (44 bugs)', 
                 color='r', ha='right', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'temperature_{temp}_comparison.png')
        plt.close()
    
    # Create a heatmap of p-values
    models = sorted(df['model'].unique())
    
    # Create multi-index DataFrame for heatmap
    heatmap_data = []
    for temp in temp_values:
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Only calculate once per pair (upper triangle)
                    u_stat, p_value, n1, n2 = perform_mann_whitney_test(df, model1, model2, temp)
                    heatmap_data.append({
                        'Temperature': temp,
                        'Model 1': model1,
                        'Model 2': model2,
                        'p-value': p_value
                    })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create pivot table for heatmap
    for temp in temp_values:
        temp_heatmap = heatmap_df[heatmap_df['Temperature'] == temp].pivot_table(
            values='p-value', index='Model 1', columns='Model 2')
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(temp_heatmap, annot=True, cmap='coolwarm_r', vmin=0, vmax=0.05, 
                   center=0.025, linewidths=0.5)
        plt.title(f'P-values for Mann-Whitney U Tests at Temperature {temp}')
        plt.tight_layout()
        plt.savefig(f'pvalue_heatmap_temp_{temp}.png')
        plt.close()

def create_comparison_to_traditional_plot(df, traditional_bugs=44):
    """
    Create a unified visualization comparing all models and temperatures to the traditional method.
    """
    # Get unique models and temperatures
    models = sorted(df['model'].unique())
    temp_values = sorted(df['temperature'].unique())
    
    # Prepare data for plotting
    comparison_data = []
    
    for model in models:
        for temp in temp_values:
            model_bugs = df[(df['model'] == model) & (df['temperature'] == temp)]['bugs_found']
            if len(model_bugs) > 0:
                mean_bugs = model_bugs.mean()
                comparison_data.append({
                    'Model': model,
                    'Temperature': temp,
                    'Mean Bugs Found': mean_bugs,
                    'Difference from Traditional': mean_bugs - traditional_bugs
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Bar plot showing difference from traditional method
    g = sns.catplot(
        data=comparison_df,
        kind="bar",
        x="Model", y="Difference from Traditional", hue="Temperature",
        palette="viridis", alpha=.8, height=6, aspect=2
    )
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.title('Difference in Bugs Found Compared to Traditional Method (44 bugs)', fontsize=15)
    plt.ylabel('Difference in Bugs Found', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('comparison_to_traditional_method.png')
    plt.close()
    
    # Create heatmap showing percentage improvement
    pivot_df = comparison_df.copy()
    pivot_df['Percent Improvement'] = (pivot_df['Difference from Traditional'] / traditional_bugs) * 100
    
    # Create pivot table for heatmap
    heatmap_data = pivot_df.pivot_table(
        values='Percent Improvement', 
        index='Model', 
        columns='Temperature'
    )
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, fmt='.1f',
               linewidths=0.5, cbar_kws={'label': 'Percent Improvement Over Traditional Method'})
    plt.title('Percent Improvement Over Traditional Method (44 bugs) by Model and Temperature')
    plt.tight_layout()
    plt.savefig('percent_improvement_heatmap.png')
    plt.close()

def run_analysis(file_path='experiment_data.csv', traditional_bugs=44):
    """
    Main function to run the analysis.
    """
    print("Loading and processing data...")
    df = load_and_clean_data(file_path)
    
    # Display basic data info
    print("\nData Overview:")
    print(f"Total records: {len(df)}")
    print(f"Models found: {df['model'].unique()}")
    print(f"Temperature values: {sorted(df['temperature'].unique())}")
    
    # Compare with traditional method
    print(f"\nComparing with traditional method ({traditional_bugs} bugs)...")
    traditional_comparison = compare_with_traditional_method(df, traditional_bugs)
    
    # Display traditional method comparison results
    print("\nTraditional Method Comparison Results:")
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', 150)      # Wider display
    print(traditional_comparison.to_string(index=False))
    
    # Save traditional method comparison to CSV
    traditional_comparison.to_csv("traditional_method_comparison.csv", index=False)
    print("\nTraditional method comparison saved to 'traditional_method_comparison.csv'")
    
    # Run Mann-Whitney U tests for all model pairs at each temperature
    print("\nPerforming Mann-Whitney U tests for all model and temperature combinations...")
    results = compare_all_combinations(df)
    
    # Display results
    print("\nMann-Whitney U Test Results:")
    print(results.to_string(index=False))
    
    # Save results to CSV
    results.to_csv("mann_whitney_results.csv", index=False)
    print("\nResults saved to 'mann_whitney_results.csv'")
    
    # Generate summary statistics by model and temperature
    print("\nSummary Statistics by Model and Temperature:")
    summary = df.groupby(['model', 'temperature'])['bugs_found'].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index()
    print(summary.to_string(index=False))
    
    # # Save summary to CSV
    # summary.to_csv("summary_statistics.csv", index=False)
    # print("\nSummary statistics saved to 'summary_statistics.csv'")
    
    # # Create visualizations for all models
    # print("\nGenerating visualizations for all models...")
    # plt.figure(figsize=(15, 10))
    
    # # Boxplot comparing all models across temperatures
    # ax = sns.boxplot(x='temperature', y='bugs_found', hue='model', data=df)
    
    # # Add horizontal line for traditional method
    # plt.axhline(y=traditional_bugs, color='r', linestyle='--', label='Traditional Method')
    
    # plt.title('Comparison of Bugs Found Across All Models and Temperatures')
    # plt.xlabel('Temperature')
    # plt.ylabel('Number of Bugs Found')
    
    # # Add the traditional method to the legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels + ['Traditional Method (44 bugs)'], 
    #          title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # plt.tight_layout()
    # plt.savefig('all_models_comparison.png')
    # plt.close()
    # print("Visualizations saved as 'all_models_comparison.png'")
    
    # # Generate per-temperature comparisons
    # visualize_comparison_by_temperature(df, traditional_bugs)
    # print("Per-temperature visualizations and p-value heatmaps generated")
    
    # # Generate comparison to traditional method visualizations
    # create_comparison_to_traditional_plot(df, traditional_bugs)
    # print("Traditional method comparison visualizations generated")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Make a U-test for input file.')
    parser.add_argument('file', help='File to process')
    parser.add_argument('--baseline', '-b', default=44.0, type=float, 
                        help='Baseline value for comparison')
    
    args = parser.parse_args()    
    # Run the analysis with the traditional method baseline of 44 bugs
    print(args)
    run_analysis(args.file, args.baseline)