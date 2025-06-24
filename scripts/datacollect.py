# Standard library imports
import itertools
import os
import sys
import argparse

from common import B1_INST_INFO_FILE

# Third-party library imports
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.gridspec as gridspec

# Headers for CSV files
METRICS_HEADER = 'model,temperature,file,total_files,total_files_with_invalid_json,total_seeds,total_duplicate_seeds,total_seeds_with_invalid_struct,total_args_in_seeds,total_invalid_args_in_seeds,total_functions_in_seeds,total_invalid_function_in_seeds'.split(',')
COVERAGE_HEADER = 'contract,temperature,transaction_index,model,seed_file,totalExecutions,deployFailCount,coveredEdges,coveredInstructions,coveredDefUseChains,bugsFound'.split(',')
B1_TOTAL_COV_HEADER = 'contract,totalInstructions,totalEdges'.split(',')


def get_model_visualization_scheme(models=None):
    
    # Get the deep color palette for better contrast
    #deep_palette = sns.color_palette("deep", 12)  # Get enough colors for all models
    deep_palette = sns.color_palette("tab10", 12)  # Original color palette    
    
    # Convert RGB tuples to hex for easier use
    deep_palette_hex = []
    for rgb in deep_palette:
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        deep_palette_hex.append(hex_color)
    
    # Fixed mapping of models to colors from deep palette and markers
    model_scheme = {
        "Llama3-70B":       {"color": deep_palette_hex[0], "marker": "s"},  # Blue with plus
        "Llama3-8B":        {"color": deep_palette_hex[1], "marker": "o"},  # Orange with circle
        "Llama3.3-70B":     {"color": deep_palette_hex[2], "marker": "D"},  # Green with diamond
        "Gemini-1.5-Flash": {"color": deep_palette_hex[3], "marker": "x"},  # Red with square
        "GPT-4o-Mini":      {"color": deep_palette_hex[4], "marker": "^"},  # Purple with triangle up
        "GPT-4.1-Mini":     {"color": deep_palette_hex[5], "marker": "*"},  # Brown
        "Mixtral-8x7B":     {"color": deep_palette_hex[6], "marker": "v"},  #  with triangle down
    }
    
    # If specific models requested, filter the mapping
    if models is not None:
        filtered_scheme = {}
        for i, model in enumerate(models):
            if model in model_scheme:
                filtered_scheme[model] = model_scheme[model]
            else:
                # For unknown models, assign the next color in the palette
                # and cycle through markers
                markers = ['+', 'o', 'D', 's', '^', 'v', 'x', '*', 'p', 'h', '8', '.']
                color_idx = i % len(deep_palette_hex)
                marker_idx = i % len(markers)
                filtered_scheme[model] = {
                    "color": deep_palette_hex[color_idx], 
                    "marker": markers[marker_idx]
                }
        return filtered_scheme
    
    # Otherwise return complete mapping
    return model_scheme

# Utility function to get colors and markers as separate lists
def get_colors_and_markers(models):
    scheme = get_model_visualization_scheme(models)
    colors = [scheme[model]["color"] for model in models]
    markers = [scheme[model]["marker"] for model in models]
    return colors, markers


def format_model_name(name):
    """Format model names consistently"""
    model_name_map = {
        'llama3.3-70B': 'Llama3.3-70B',
        'llama3-70b': 'Llama3-70B',            
        'gpt4-0mini': 'GPT-4o-Mini',
        'gpt4omini': 'GPT-4o-Mini',         
        'gpt4.1mini': 'GPT-4.1-Mini',                     
        'llama3-8b': 'Llama3-8B',
        'mixtral-8x7b': 'Mixtral-8x7B',
        'gemini-1.5-flash': 'Gemini-1.5-Flash'
    }
    return model_name_map.get(name.lower(), name)

def load_coverage_data(csv):
    """Load coverage data from CSV"""
    df = pd.read_csv(B1_INST_INFO_FILE, header=None, names=B1_TOTAL_COV_HEADER)
    return df

def fill_missing_experiments(df):
    """
    Fill in missing experiment combinations in a DataFrame, keeping only transaction indices 1-10
    and filling remaining values with 0.
    """
    # First filter the original DataFrame to keep only transaction_index 1-10
    df = df[df['transaction_index'].between(1, 10)]
    
    # Get unique values for each dimension
    models = df['model'].unique()
    temperatures = df['temperature'].unique()
    contracts = df['contract'].unique()
    transaction_indices = list(range(1, 11))  # Force transaction indices to be 1-10
    
    # Create all possible combinations
    index = pd.MultiIndex.from_product([contracts, temperatures, transaction_indices, models],
                                    names=['contract', 'temperature', 'transaction_index', 'model'])
    
    # Convert to DataFrame
    complete_df = pd.DataFrame(index=index).reset_index()
    
    # Merge with original data
    result = pd.merge(complete_df, df, 
                    on=['contract', 'temperature', 'transaction_index', 'model'],
                    how='left')
    
    # Sort the result for better readability
    result = result.sort_values(['contract', 'temperature', 'transaction_index', 'model'])
    
    # Reset index
    result = result.reset_index(drop=True)
    
    # Fill all numeric columns with 0
    numeric_columns = ['totalExecutions', 'deployFailCount', 'coveredEdges', 
                    'coveredInstructions', 'coveredDefUseChains', 'bugsFound']
    result[numeric_columns] = result[numeric_columns].fillna(0)
    
    # Fill seed_file with 'missing' if it exists in the DataFrame
    if 'seed_file' in result.columns:
        result['seed_file'] = result['seed_file'].fillna('missing')
        
    return result

def build_coverage_data(csv):
    """Build coverage data from executions CSV"""
    executions_df = pd.read_csv(csv, header=None, names=COVERAGE_HEADER)        
    totals_df = load_coverage_data("B1-ins.csv")
    
    executions_df = fill_missing_experiments(executions_df)

    total_valid_seeds = executions_df.groupby(['model', 'temperature', 'contract']).size().reset_index(name='row_count')
    total_valid_seeds = total_valid_seeds.groupby(['model', 'temperature']).size().reset_index(name='seed_count').sort_values(by=['seed_count'], ascending=False)

    totals = executions_df.groupby(
        ['contract', 'model', 'temperature']
    ).size().reset_index(name='totalSeedsPerModelTemp')

    # Calculate mean values for each contract-model-temperature combination
    grouped_metrics = executions_df.groupby(
        ['contract', 'model', 'temperature']
    ).agg({
        'coveredInstructions': 'mean',
        'coveredEdges': 'mean',
        'bugsFound': 'mean',
        'coveredDefUseChains': 'mean',
    }).reset_index()
    
    grouped_metrics = grouped_metrics.merge(totals, on=['contract', 'model', 'temperature'])
            
    # Generate complete combination matrix
    unique_models = grouped_metrics['model'].unique()
    unique_temperatures = grouped_metrics['temperature'].unique()
    
    # Fill missing combinations with zeros
    new_rows = []
    for _, row in totals_df.iterrows():
        contract = row['contract']
        new_total_instructions = row['totalInstructions']
        new_total_edges = row['totalEdges']
        
        for model, temperature in itertools.product(unique_models, unique_temperatures):
            query_result = grouped_metrics.loc[
                (grouped_metrics['contract'] == contract) &
                (grouped_metrics['model'] == model) &
                (grouped_metrics['temperature'] == temperature)].index
                
            if query_result.empty:
                new_rows.append({
                    'contract': contract,
                    'model': model,
                    'temperature': temperature,
                    'coveredInstructions': 0,
                    'coveredEdges': 0,
                    'bugsFound': 0,
                    'coveredDefUseChains': 0,
                    'totalInstructions': new_total_instructions,
                    'totalEdges': new_total_edges,
                    'totalSeedsPerModelTemp': 0
                })
            else:
                grouped_metrics.loc[query_result, 'totalInstructions'] = new_total_instructions
                grouped_metrics.loc[query_result, 'totalEdges'] = new_total_edges
            
    return pd.concat([grouped_metrics, pd.DataFrame(new_rows)], ignore_index=True), total_valid_seeds

def get_mean_valid_seeds_per_model(csv):
    """Get mean valid seeds per model and temperature"""
    df = pd.read_csv(csv, header=None, names=METRICS_HEADER)
    df['valid_seeds'] = df['total_seeds'] - df['total_duplicate_seeds'] - df['total_seeds_with_invalid_struct']         
    df['valid_seeds'] = df['valid_seeds'].replace(0, pd.NA)
    df['model'] = df['model'].apply(format_model_name)
    
    return (
        df.groupby(['model', 'temperature'])
        .apply(lambda group: pd.Series({
            'mean_valid_seeds': round(group['valid_seeds'].mean())
        }))
        .reset_index()            
    )

def seed_metrics(csv):
    """Generate seed metrics visualization"""
    df = pd.read_csv(csv, header=None, names=METRICS_HEADER)
            
    # Calculate means grouped by model and temperature
    grouped_df = df.groupby(['model', 'temperature']).agg({
        'total_files': 'mean',
        'total_files_with_invalid_json': 'mean',
        'total_seeds': 'mean',
        'total_duplicate_seeds': 'mean',
        'total_seeds_with_invalid_struct': 'mean'
    }).reset_index()

    # Calculate metrics
    grouped_df['valid_files_mean'] = grouped_df['total_files'] - grouped_df['total_files_with_invalid_json']
    grouped_df['valid_files_percentage'] = (grouped_df['valid_files_mean'] / grouped_df['total_files']) * 100
    grouped_df['valid_seeds'] = grouped_df['total_seeds'] - grouped_df['total_duplicate_seeds'] - grouped_df['total_seeds_with_invalid_struct']

    grouped_df['model'] = grouped_df['model'].apply(format_model_name)
    models = grouped_df['model'].unique()
            
    # Get visualization scheme using our function
    model_scheme = get_model_visualization_scheme(models)
        
    # Plot setup
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial']
    })        
    ax = plt.gca()
    
    # Plot for each model
    for model in models:
        model_data = grouped_df[grouped_df['model'] == model]
        temps = model_data['temperature']
        
        # Get color and marker for this model
        color = model_scheme[model]["color"]
        marker = model_scheme[model]["marker"]
        
        # Solid line for valid seeds
        plt.plot(temps, model_data['valid_seeds'], 
                color=color, 
                linestyle='-',
                linewidth=3,
                marker=marker,
                markersize=10,
                label=f'{model}',
                zorder=3)
                
        # Dotted line for total seeds
        plt.plot(temps, model_data['total_seeds'],
                color=color,
                linestyle=':',
                linewidth=2,
                alpha=0.7,
                marker=marker,
                markersize=8,
                zorder=2)
                
        # Fill between total and valid to show invalid
        plt.fill_between(temps, 
                        model_data['total_seeds'],
                        model_data['valid_seeds'],
                        color=color,
                        alpha=0.1,
                        zorder=1)

    # Styling
    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Number of Seeds', fontsize=20)
    plt.grid(True, alpha=0.3)        
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Legend inside the plot
    legend = plt.legend(
        title="Models",
        title_fontsize=16,
        fontsize=15,
        loc='center right',
        bbox_to_anchor=(1, 0.57),                        
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig('plot_seed_metrics.pdf', bbox_inches='tight', dpi=600)
    plt.close()

def combined_plots_percent(csv):
    """Generate combined percentage plots for duplicate and invalid structure seeds"""
    df = pd.read_csv(csv, header=None, names=METRICS_HEADER)
            
    # Calculate means grouped by model and temperature
    grouped_df = df.groupby(['model', 'temperature']).agg({
        'total_files': 'mean',
        'total_files_with_invalid_json': 'mean',
        'total_seeds': 'mean',
        'total_duplicate_seeds': 'mean',
        'total_seeds_with_invalid_struct': 'mean'
    }).reset_index()

    # Calculate metrics
    grouped_df['valid_files_mean'] = grouped_df['total_files'] - grouped_df['total_files_with_invalid_json']
    grouped_df['valid_files_percentage'] = (grouped_df['valid_files_mean'] / grouped_df['total_files']) * 100
    
    # Calculate percentages for the plots
    grouped_df['duplicate_seeds_percentage'] = (grouped_df['total_duplicate_seeds'] / grouped_df['total_seeds']) * 100
    grouped_df['invalid_struct_percentage'] = (grouped_df['total_seeds_with_invalid_struct'] / grouped_df['total_seeds']) * 100

    grouped_df['model'] = grouped_df['model'].apply(format_model_name)
    models = grouped_df['model'].unique()        

    # Set global font parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
    })        
    
    # Get visualization scheme using our function
    model_scheme = get_model_visualization_scheme(models)
            
    # ================================
    # PLOT 1: Duplicate Seeds
    # ================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    for model in models:
        model_data = grouped_df[grouped_df['model'] == model]
        
        # Get color and marker for this model
        color = model_scheme[model]["color"]
        marker = model_scheme[model]["marker"]
        
        ax1.plot(model_data['temperature'], 
                model_data['duplicate_seeds_percentage'], 
                marker=marker,
                markersize=12,
                linewidth=3,
                color=color,
                label=model,
                alpha=0.8)
        
        ax1.plot(model_data['temperature'], 
                model_data['duplicate_seeds_percentage'],
                color='gray',
                linewidth=4,
                alpha=0.2,
                zorder=-1)

    ax1.set_xlabel('Temperature', fontsize=20)
    ax1.set_ylabel('Percentage of Duplicate Seeds (%)', fontsize=20)
    ax1.grid(True, alpha=0.3)                        
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_ylim(-1, 26)        

    # Add legend for plot 1
    legend1 = ax1.legend(
        title="Models",
        title_fontsize=16,
        fontsize=15,
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1
    )
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig('duplicate_seeds_plot.pdf', bbox_inches='tight', dpi=600)
    plt.close()

    # ================================
    # PLOT 2: Invalid Structure Seeds
    # ================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df['model'] == model]
        
        # Get color and marker for this model
        color = model_scheme[model]["color"]
        marker = model_scheme[model]["marker"]
        
        # Create jittered x values
        jitter = (i - len(models)/2) * 0.01  # Small horizontal offset based on model index
        jittered_temps = model_data['temperature'] + jitter
        
        ax2.plot(jittered_temps, 
                model_data['invalid_struct_percentage'], 
                marker=marker,
                markersize=12,
                linewidth=3,
                color=color,
                label=model,
                alpha=0.8)
        
        ax2.plot(jittered_temps, 
                model_data['invalid_struct_percentage'],
                color='gray',
                linewidth=4,
                alpha=0.2,
                zorder=-1)

    ax2.set_xlabel('Temperature', fontsize=20)
    ax2.set_ylabel('Percentage of Seeds with Invalid Structure (%)', fontsize=20)
    ax2.grid(True, alpha=0.3)                                
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_ylim(-1, 26)        

    # Add legend for plot 2
    legend2 = ax2.legend(
        title="Models",
        title_fontsize=16,
        fontsize=15,
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1
    )
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig('invalid_structure_seeds_plot.pdf', bbox_inches='tight', dpi=600)
    plt.close()

def valid_files_percentage(csv):
    """Generate valid files percentage plot"""
    df = pd.read_csv(csv, header=None, names=METRICS_HEADER)
            
    # Calculate means grouped by model and temperature
    grouped_df = df.groupby(['model', 'temperature']).agg({
        'total_files': 'mean',
        'total_files_with_invalid_json': 'mean',
        'total_seeds': 'mean',
        'total_duplicate_seeds': 'mean',
        'total_seeds_with_invalid_struct': 'mean'
    }).reset_index()

    # Calculate metrics
    grouped_df['valid_files_mean'] = grouped_df['total_files'] - grouped_df['total_files_with_invalid_json']
    grouped_df['valid_files_percentage'] = (grouped_df['valid_files_mean'] / grouped_df['total_files']) * 100
    grouped_df['valid_seeds'] = grouped_df['total_seeds'] - grouped_df['total_duplicate_seeds'] - grouped_df['total_seeds_with_invalid_struct']

    grouped_df['model'] = grouped_df['model'].apply(format_model_name)
    models = grouped_df['model'].unique()        

    # Get visualization scheme using our function
    model_scheme = get_model_visualization_scheme(models)
            
    # Plot for valid files percentage
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    for model in models:
        model_data = grouped_df[grouped_df['model'] == model]
        
        # Get color and marker for this model
        color = model_scheme[model]["color"]
        marker = model_scheme[model]["marker"]
        
        plt.plot(model_data['temperature'], 
                model_data['valid_files_percentage'], 
                marker=marker,
                markersize=12,
                linewidth=3,
                color=color,
                label=model,
                alpha=0.8)
        
        plt.plot(model_data['temperature'], 
                model_data['valid_files_percentage'],
                color='gray',
                linewidth=4,
                alpha=0.2,
                zorder=-1)

    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Percentage of Valid Outputs (%)', fontsize=20)
    plt.grid(True, alpha=0.3)                                
    ax.tick_params(axis='both', which='major', labelsize=16)

    legend = plt.legend(
        title="Models",
        title_fontsize=16,
        fontsize=15,
        loc='center right',
        bbox_to_anchor=(1, 0.69),
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig('plot_valid_files_percentage.pdf', bbox_inches='tight', dpi=600)
    plt.close()

def seed_args_and_funcs(csv):
    """Generate plots for seed arguments and functions analysis"""
    df = pd.read_csv(csv, header=None, names=METRICS_HEADER)                    
    # Format model names
    df['model'] = df['model'].apply(format_model_name)
    
    # Get the complete model scheme
    complete_model_scheme = get_model_visualization_scheme()
    
    # Extract the model order from the keys of the scheme
    model_order = list(complete_model_scheme.keys())        

    # Group by model and temperature
    grouped = df.groupby(['model', 'temperature']).agg({
        'total_args_in_seeds': 'sum',
        'total_invalid_args_in_seeds': 'sum',
        'total_functions_in_seeds': 'sum',
        'total_invalid_function_in_seeds': 'sum',
        'file': 'count'  # Count number of runs
    }).reset_index()

    # Ensure the model column follows the desired order
    grouped['model'] = pd.Categorical(grouped['model'], categories=model_order, ordered=True)
    grouped = grouped.sort_values(['model', 'temperature'])

    # Calculate error rates
    grouped['args_error_rate'] = (
        grouped['total_invalid_args_in_seeds'] / grouped['total_args_in_seeds'] * 100
    )

    grouped['functions_error_rate'] = (
        grouped['total_invalid_function_in_seeds'] / grouped['total_functions_in_seeds'] * 100
    )

    grouped['combined_error_rate'] = (
        (grouped['total_invalid_args_in_seeds'] / grouped['total_args_in_seeds'] +
        grouped['total_invalid_function_in_seeds'] / grouped['total_functions_in_seeds']) / 2 * 100
    )

    grouped['invalid_args_per_run'] = grouped['total_invalid_args_in_seeds'] / grouped['file']
    grouped['invalid_functions_per_run'] = grouped['total_invalid_function_in_seeds'] / grouped['file']

    # Get unique models from the data that are actually present
    unique_models = grouped['model'].unique()
    
    # Get the visualization scheme using our function
    model_scheme = get_model_visualization_scheme(unique_models)

    # Print values to stdout
    print("\n===== METRICS VALUES BY MODEL AND TEMPERATURE =====")
    for model in unique_models:
        model_data = grouped[grouped['model'] == model]
        print(f"\nModel: {model}")
        
        for _, row in model_data.iterrows():
            temp = row['temperature']
            print(f"  Temperature: {temp}")
            print(f"    Arguments Error Rate: {row['args_error_rate']:.2f}%")
            print(f"    Functions Error Rate: {row['functions_error_rate']:.2f}%")
            print(f"    Combined Error Rate: {row['combined_error_rate']:.2f}%")
            print(f"    Invalid Arguments Per Run: {row['invalid_args_per_run']:.2f}")
            print(f"    Invalid Functions Per Run: {row['invalid_functions_per_run']:.2f}")
    print("\n=================================================")

    # Create visualizations for each metric
    metrics_to_plot = [
        ('args_error_rate', 'Arguments Error Rate (%)', 'Invalid Functions Arguments (%) By Temperature', 50),
        ('functions_error_rate', 'Functions Error Rate (%)', 'Invalid Functions (%) By Temperature', 50),
    ]
    
    def plot_metric(data, metric, title, ylabel, ylim=0, figsize=(12, 8)):
        plt.figure(figsize=figsize)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial']
        })            
        ax = plt.gca()
        
        for model in unique_models:
            model_data = data[data['model'] == model]
            # Skip if no data for this model
            if model_data.empty:
                continue
                
            # Get the color and marker for this model from our scheme
            color = model_scheme[model]["color"]
            marker = model_scheme[model]["marker"]
        
            # Main line with markers
            plt.plot(model_data['temperature'], 
                    model_data[metric], 
                    marker=marker,
                    markersize=12,
                    linewidth=3,
                    color=color,
                    label=model,
                    alpha=0.8)
            
            # Shadow effect
            plt.plot(model_data['temperature'], 
                    model_data[metric],
                    color='gray',
                    linewidth=4,
                    alpha=0.2,
                    zorder=-1)

        # Styling
        plt.xlabel('Temperature', fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.grid(True, alpha=0.3)            
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Enhanced legend
        legend = plt.legend(
            title="Models",
            title_fontsize=16,
            fontsize=15,
            loc='best',
            frameon=True,
            fancybox=True,
            shadow=True,
            borderpad=1
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        if ylim > 0:
            plt.ylim(-1, ylim+1)

        plt.tight_layout()
        return plt
    
    for metric, ylabel, title, ylim in metrics_to_plot:
        plot = plot_metric(grouped, metric, title, ylabel, ylim)
        plot.savefig(f'{metric}.pdf', bbox_inches='tight', dpi=600)
        plt.close()

def coverage_means(coverage_csv, metrics_csv):
    """Generate coverage analysis plots and statistics"""
    df, total_valid_seeds = build_coverage_data(coverage_csv)
    total_valid_seeds['model'] = total_valid_seeds['model'].apply(format_model_name)
    df['model'] = df['model'].apply(format_model_name)   
         
    df['instruction_coverage'] = (
        df['coveredInstructions'].div(df['totalInstructions'])
        .mul(100)
        .round(2)
    )

    df['edge_coverage'] = (
        df['coveredEdges'].div(df['totalEdges'])
        .mul(100)
        .round(2)
    )

    coverage_by_model_temp = (
        df.groupby(['model', 'temperature'])
        .agg({
            'instruction_coverage': 'mean',
            'edge_coverage': 'mean'
        })
        .round(2)
        .reset_index()
        .rename(columns={
            'instruction_coverage': 'mean_instruction_coverage_percentage',
            'edge_coverage': 'mean_edge_coverage_percentage'
        })
        .sort_values(['mean_instruction_coverage_percentage', 'mean_edge_coverage_percentage'], ascending=False)
    )

    coverage_by_model_temp = coverage_by_model_temp.merge(total_valid_seeds, on=['model', 'temperature'])
    valid_seed = get_mean_valid_seeds_per_model(metrics_csv)
    coverage_by_model_temp = coverage_by_model_temp.merge(valid_seed, on=['model', 'temperature'])
    print(coverage_by_model_temp)

    coverage_by_model= (
        coverage_by_model_temp.groupby(['model'])
        .agg({
            'mean_instruction_coverage_percentage': 'mean',
            'mean_edge_coverage_percentage': 'mean',
            'mean_valid_seeds': 'mean',
            'seed_count': 'mean'
            
        })
        .round(2)
        .reset_index()
        .sort_values(['mean_instruction_coverage_percentage', 'mean_edge_coverage_percentage'], ascending=False)
    )
    print(coverage_by_model)
    
    model_order = coverage_by_model['model'].tolist()
    
    # Plot 1: Coverage Percentage by Model
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(2, 2, 1)
    
    model_graph = coverage_by_model.copy()  # Create a copy to avoid modifying original
    model_graph.rename(columns={'mean_instruction_coverage_percentage': 'instruction_coverage'}, inplace=True)        
    model_graph.rename(columns={'mean_edge_coverage_percentage': 'edge_coverage'}, inplace=True)
    model_graph.drop(columns=['mean_valid_seeds', 'seed_count']).set_index('model').plot(kind='bar', ax=ax1)
    plt.title('Average Coverage by Model')
    plt.xlabel('')
    plt.ylim(0, 100)                
    plt.ylabel('Coverage Percentage')
    plt.xticks(rotation=48)
    plt.legend(title='Coverage Metric')
    
    for p in ax1.patches:
        # Position labels higher above each bar with inclination
        ax1.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center',  # Change to left alignment for inclined text
                    va='bottom',
                    fontsize=7, color='black', 
                    xytext=(0, 2),  # Vertical offset
                    textcoords='offset points',
                    rotation=63)  # Use 45-degree rotation (less extreme than 70)

    # Increase the top margin to accommodate inclined labels
    plt.ylim(0, 110)  # More space at the top
    plt.tight_layout(pad=3.0)  # Increased padding
    plt.savefig('coverage_means.pdf', bbox_inches="tight")
    
    # Plot 2: Temperature Heatmap Instruction
    plt.figure(figsize=(14, 8))
    ax2 = plt.subplot(2, 2, 1)
    
    pivot_data = coverage_by_model_temp.pivot_table(
        values='mean_instruction_coverage_percentage',
        index='model',
        columns='temperature',
        aggfunc='mean'
    )
    pivot_data = pivot_data.reindex(model_order)
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2, vmax=100)
    plt.title('Instruction Coverage Heatmap (Model vs Temperature)')
    plt.xlabel('Temperature')
    plt.ylabel('Model')
    plt.tight_layout()            
    plt.savefig('instruction_heatmap_mode_temp.pdf', bbox_inches="tight")        

    # Plot 3: Temperature Heatmap Edge
    plt.figure(figsize=(14, 8))
    ax2 = plt.subplot(2, 2, 1)
    
    pivot_data = coverage_by_model_temp.pivot_table(
        values='mean_edge_coverage_percentage',
        index='model',
        columns='temperature',
        aggfunc='mean'
    )
    pivot_data = pivot_data.reindex(model_order)
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='Purples', ax=ax2, vmax=100)
    plt.title('Edge Coverage Heatmap (Model vs Temperature)')
    plt.xlabel('Temperature')
    plt.ylabel('Model')
    plt.tight_layout()            
    plt.savefig('edge_heatmap_mode_temp.pdf', bbox_inches="tight")        
    
    # Plot 4: Temperature Heatmap defusechain
    plt.figure(figsize=(14, 8))
    ax2 = plt.subplot(2, 2, 1)        
    defuse_by_model_temp = df.pivot_table(
        values='coveredDefUseChains',
        index='model',
        columns='temperature',
        aggfunc='sum'
    )
    defuse_by_model_temp = defuse_by_model_temp.reindex(model_order)
    
    sns.heatmap(defuse_by_model_temp, annot=True, fmt='.0f', cmap='Greens', ax=ax2)
    plt.title('Total Coverage of DefUse Chain Found (Model vs Temperature)')
    plt.xlabel('Temperature')
    plt.ylabel('Model')
    plt.tight_layout()            
    plt.savefig('defuse_heatmap_mode_temp.pdf', bbox_inches="tight")        
    
    # Plot 5: Temperature Heatmap bugsfound        
    plt.figure(figsize=(14, 8))
    ax2 = plt.subplot(2, 2, 1)        
    bugs_by_model_temp = df.pivot_table(
        values='bugsFound',
        index='model',
        columns='temperature',
        aggfunc='sum'
    )
    bugs_by_model_temp = bugs_by_model_temp.reindex(model_order)
    
    sns.heatmap(bugs_by_model_temp, annot=True, fmt='.0f', cmap="Purples", ax=ax2)
    plt.title('Total Bugs Found (Model vs Temperature)')
    plt.xlabel('Temperature')
    plt.ylabel('Model')
    plt.tight_layout()            
    plt.savefig('bugs_heatmap_mode_temp.pdf', bbox_inches="tight")        
    
    # Plot 6: Coverage Distribution by Model
    plt.figure(figsize=(14, 8))
    ax2 = plt.subplot(2, 2, 1)        
    
    # Create a copy of the dataframe and ensure correct column name
    plot_df = df.copy()
    if 'instruction_coverage' not in plot_df.columns:
        plot_df['instruction_coverage'] = plot_df['mean_instruction_coverage_percentage']
    
    # Use order parameter in violinplot with the processed dataframe
    sns.violinplot(data=plot_df, x='model', y='instruction_coverage', ax=ax2, order=model_order)
    plt.title('Instruction Coverage Distribution by Model')
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Coverage Percentage')
    plt.tight_layout()            
    plt.savefig('violin_inst_cov.pdf', bbox_inches="tight")

def main():
    """Main function to handle CLI arguments and call appropriate functions"""
    parser = argparse.ArgumentParser(description='Data Analysis Script')
    parser.add_argument('function', choices=['seed_metrics', 'combined_plots_percent', 'valid_files_percentage', 'seed_args_and_funcs', 'coverage_means'], 
                       help='Function to execute')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--metrics_csv', help='Path to metrics CSV file (required for coverage_means)')
    
    args = parser.parse_args()
    
    if args.function == 'seed_metrics':
        seed_metrics(args.csv_file)
        print(f"Generated plot_seed_metrics.pdf")
        
    elif args.function == 'combined_plots_percent':
        combined_plots_percent(args.csv_file)
        print(f"Generated duplicate_seeds_plot.pdf and invalid_structure_seeds_plot.pdf")
        
    elif args.function == 'valid_files_percentage':
        valid_files_percentage(args.csv_file)
        print(f"Generated plot_valid_files_percentage.pdf")
        
    elif args.function == 'seed_args_and_funcs':
        seed_args_and_funcs(args.csv_file)
        print(f"Generated args_error_rate.pdf and functions_error_rate.pdf")
        
    elif args.function == 'coverage_means':
        if not args.metrics_csv:
            print("Error: --metrics_csv is required for coverage_means function")
            sys.exit(1)
        coverage_means(args.csv_file, args.metrics_csv)
        print(f"Generated multiple coverage analysis plots")

if __name__ == "__main__":
    main()
