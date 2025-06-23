import matplotlib
matplotlib.use('Agg')  # Force consistent backend for PDF quality
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

# Set PDF-specific settings for high quality output
plt.rcParams['pdf.fonttype'] = 42  # Use Type 1 fonts in PDF
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['xtick.labelsize'] = 14  # Increased x-axis tick font size
plt.rcParams['ytick.labelsize'] = 14  # Increased y-axis tick font size

def read_coverage_file(filename):
    """
    Read coverage data from a file.
    Expected format: 'XXm: YY.YY' (e.g., '01m: 82.47')
    """
    time_points = []
    coverage_values = []
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line and ':' in line:
                    # Parse lines like '01m: 82.47'
                    time_str, coverage_str = line.split(':')
                    
                    # Extract minutes (remove 'm' and convert to int)
                    minutes = int(time_str.strip().replace('m', ''))
                    
                    # Extract coverage value
                    coverage = float(coverage_str.strip())
                    
                    time_points.append(minutes)
                    coverage_values.append(coverage)
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing file '{filename}': {e}")
        sys.exit(1)
    
    return time_points, coverage_values

def plot_coverage_comparison(baseline_file, method1_file, method2_file, output_file=None, legend_text=None, legend_color='black', line_colors=None, baseline_name=None, method1_name=None, method2_name=None):
    """
    Plot coverage comparison between baseline and two methods.
    """
    # Read data from files
    print(f"Reading baseline data from: {baseline_file}")
    time_baseline, coverage_baseline = read_coverage_file(baseline_file)
    
    print(f"Reading method 1 data from: {method1_file}")
    time_method1, coverage_method1 = read_coverage_file(method1_file)
    
    print(f"Reading method 2 data from: {method2_file}")
    time_method2, coverage_method2 = read_coverage_file(method2_file)
    
    # Validate data consistency
    if len(time_baseline) != len(time_method1) or len(time_baseline) != len(time_method2):
        print("Warning: Different number of data points in the files")
    
    # Set method names
    baseline_label = baseline_name if baseline_name else 'Baseline Method'
    method1_label = method1_name if method1_name else 'Method 1'
    method2_label = method2_name if method2_name else 'Method 2'
    
    # Parse line colors if provided
    if line_colors:
        colors = [color.strip() for color in line_colors.split(',')]
        baseline_color = colors[0] if len(colors) > 0 else '#1f77b4'
        method1_color = colors[1] if len(colors) > 1 else '#d62728'
        method2_color = colors[2] if len(colors) > 2 else '#2ca02c'
    else:
        baseline_color = '#1f77b4'  # Blue
        method1_color = '#d62728'   # Red
        method2_color = '#2ca02c'   # Green
    
    # Create the main comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot all three methods with custom colors and names
    plt.plot(time_baseline, coverage_baseline, color=baseline_color, linestyle='-', linewidth=2, label=baseline_label, 
             marker='o', markersize=3, alpha=0.8)
    plt.plot(time_method1, coverage_method1, color=method1_color, linestyle='-', linewidth=2, label=method1_label, 
             marker='s', markersize=3, alpha=0.8)
    plt.plot(time_method2, coverage_method2, color=method2_color, linestyle='-', linewidth=2, label=method2_label, 
             marker='^', markersize=3, alpha=0.8)
    
    # Customize the plot - NO TITLE
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Instruction Coverage (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Explicitly set tick label font sizes (alternative method for more control)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Place legend at the top center like in the image with proper spacing
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=True)
    
    # Set appropriate limits with enhanced y-axis view
    max_time = max(max(time_baseline), max(time_method1), max(time_method2))
    min_coverage = min(min(coverage_baseline), min(coverage_method1), min(coverage_method2))
    max_coverage = max(max(coverage_baseline), max(coverage_method1), max(coverage_method2))
    
    # Skip the 0% start point for better visualization of differences
    all_coverage = coverage_baseline + coverage_method1 + coverage_method2
    min_coverage_nonzero = min([c for c in all_coverage if c > 0])
    y_range = max_coverage - min_coverage_nonzero
    
    plt.xlim(0, max_time)
    # Start y-axis from a higher value to emphasize differences
    plt.ylim(min_coverage_nonzero - y_range * 0.1, max_coverage + y_range * 0.05)
    
    # Add custom legend text if provided
    if legend_text:
        plt.text(0.98, 0.02, legend_text, 
                transform=plt.gca().transAxes, fontsize=18, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='right',
                color=legend_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    # Add extra space at the top for the legend
    plt.subplots_adjust(top=0.8)
    
    # Save or show the plot - PDF only with high quality
    if output_file:
        # Ensure .pdf extension
        if not output_file.lower().endswith('.pdf'):
            output_file = output_file.rsplit('.', 1)[0] + '.pdf'
        
        plt.savefig(output_file, format='pdf', bbox_inches='tight', 
                   dpi=600, facecolor='white', edgecolor='none',
                   pad_inches=0.1)
        print(f"Full comparison plot saved as: {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("COVERAGE ANALYSIS SUMMARY")
    print("="*50)
    
    if len(coverage_baseline) > 0 and len(coverage_method1) > 0 and len(coverage_method2) > 0:
        print(f"{baseline_label}:")
        print(f"  - Coverage at 1min: {coverage_baseline[1]:.2f}%")
        print(f"  - Final coverage: {coverage_baseline[-1]:.2f}%")
        print(f"  - Total improvement: {coverage_baseline[-1] - coverage_baseline[0]:.2f}%")
        
        print(f"\n{method1_label}:")
        print(f"  - Coverage at 1min: {coverage_method1[1]:.2f}%")
        print(f"  - Final coverage: {coverage_method1[-1]:.2f}%")
        print(f"  - Total improvement: {coverage_method1[-1] - coverage_method1[0]:.2f}%")
        
        print(f"\n{method2_label}:")
        print(f"  - Coverage at 1min: {coverage_method2[1]:.2f}%")
        print(f"  - Final coverage: {coverage_method2[-1]:.2f}%")
        print(f"  - Total improvement: {coverage_method2[-1] - coverage_method2[0]:.2f}%")
        
        print(f"\nComparison:")
        print(f"  - {method1_label} advantage at 1min: {coverage_method1[1] - coverage_baseline[1]:.2f}%")
        print(f"  - {method2_label} advantage at 1min: {coverage_method2[1] - coverage_baseline[1]:.2f}%")
        print(f"  - {method1_label} final advantage: {coverage_method1[-1] - coverage_baseline[-1]:.2f}%")
        print(f"  - {method2_label} final advantage: {coverage_method2[-1] - coverage_baseline[-1]:.2f}%")
        
        # Calculate improvement rate in first 5 minutes
        if len(coverage_baseline) >= 6 and len(coverage_method1) >= 6 and len(coverage_method2) >= 6:
            baseline_5min_rate = (coverage_baseline[5] - coverage_baseline[0]) / 5
            method1_5min_rate = (coverage_method1[5] - coverage_method1[0]) / 5
            method2_5min_rate = (coverage_method2[5] - coverage_method2[0]) / 5
            print(f"  - Avg. improvement rate (0-5min):")
            print(f"    {baseline_label}: {baseline_5min_rate:.2f}%/min")
            print(f"    {method1_label}: {method1_5min_rate:.2f}%/min")
            print(f"    {method2_label}: {method2_5min_rate:.2f}%/min")
    

def main():
    parser = argparse.ArgumentParser(description='Compare fuzzing coverage between baseline and two methods')
    parser.add_argument('baseline_file', help='File containing baseline method coverage data')
    parser.add_argument('method1_file', help='File containing method 1 coverage data')
    parser.add_argument('method2_file', help='File containing method 2 coverage data')
    parser.add_argument('-o', '--output', help='Output file for saving plots (optional)')
    parser.add_argument('-l', '--legend', help='Custom legend text to display on the plot')
    parser.add_argument('--baseline-name', help='Custom name for the baseline method (default: "Baseline Method")')
    parser.add_argument('--method1-name', help='Custom name for method 1 (default: "Method 1")')
    parser.add_argument('--method2-name', help='Custom name for method 2 (default: "Method 2")')
    parser.add_argument('--colors', help='Line colors for baseline, method1, and method2 (comma-separated hex colors, e.g., "#1f77b4,#d62728,#2ca02c")')
    parser.add_argument('-c', '--color', default='black', help='Color for the legend text (default: black)')
    
    args = parser.parse_args()
    
    # Generate the comparison plots
    plot_coverage_comparison(args.baseline_file, args.method1_file, args.method2_file, args.output, args.legend, args.color, args.colors, args.baseline_name, args.method1_name, args.method2_name)

if __name__ == "__main__":
    main()

# Example usage (if running without command line arguments):
# Uncomment the lines below and modify the filenames as needed
# plot_coverage_comparison('baseline_coverage.txt', 'method1_coverage.txt', 'method2_coverage.txt', 
#                         legend_text='Fuzzing Comparison', legend_color='blue', 
#                         line_colors='#1f77b4,#d62728,#2ca02c',
#                         baseline_name='Baseline', method1_name='Method A', method2_name='Method B')
