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
plt.rcParams['font.size'] = 14  # Increase base font size
plt.rcParams['xtick.labelsize'] = 14  # Larger tick labels
plt.rcParams['ytick.labelsize'] = 14  # Larger tick labels
plt.rcParams['axes.labelsize'] = 16  # Larger axis labels
plt.rcParams['legend.fontsize'] = 14  # Larger legend

# Default colors for up to 10 methods (can be extended)
DEFAULT_COLORS = [
    '#1f77b4',  # Blue
    '#d62728',  # Red
    '#2ca02c',  # Green
    '#ff7f0e',  # Orange
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]

# Default line markers for distinguishing methods
DEFAULT_MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

def read_coverage_file(filename):
    """
    Read coverage data from a file, **only from the section after the last '===================================' line**.
    Expected format for data lines: 'XXm: YY.YY' (e.g., '01m: 82.47')
    """
    time_points = []
    coverage_values = []

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Find the last occurrence of the separator
        sep = "==================================="
        last_sep_index = -1
        for idx, line in enumerate(lines):
            if sep in line:
                last_sep_index = idx
        
        # Only process lines after the last separator
        for line in lines[last_sep_index+1:]:
            line = line.strip()
            if line and ':' in line:
                try:
                    # Parse lines like '01m: 82.47'
                    time_str, coverage_str = line.split(':')
                    # Extract minutes (remove 'm' and convert to int)
                    minutes = int(time_str.strip().replace('m', ''))
                    # Extract coverage value
                    coverage = float(coverage_str.strip())
                    time_points.append(minutes)
                    coverage_values.append(coverage)
                except ValueError as e:
                    print(f"Warning: Could not parse line '{line}': {e}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

    return time_points, coverage_values

def plot_coverage_comparison(baseline_file, method_files, output_file=None, legend_text=None, 
                           legend_color='black', line_colors=None, baseline_name=None, 
                           method_names=None, ylabel='Instruction Coverage (%)', ylim_from_zero=False, 
                           ylim_custom=None, xtick_interval=None, ytick_interval=None, legend_loc='lower right',
                           subplot_layout=False, left_ylabel='Total # of CVEs found', left_data_files=None,
                           left_ylim_from_zero=False, left_ylim_custom=None):
    """
    Plot coverage comparison between baseline and multiple methods.
    
    Args:
        baseline_file: Path to baseline coverage file
        method_files: List of paths to method coverage files
        output_file: Output file path for saving the plot
        legend_text: Custom legend text to display
        legend_color: Color for the legend text
        line_colors: Comma-separated string of colors for all lines
        baseline_name: Custom name for baseline method
        method_names: List of custom names for methods
        ylabel: Y-axis label text
        ylim_from_zero: If True, y-axis starts from 0; if False, use optimized view
        ylim_custom: Tuple of (min, max) for custom y-axis limits, overrides other ylim options
        xtick_interval: Interval for x-axis ticks (e.g., 5 for every 5 minutes)
        ytick_interval: Interval for y-axis ticks (e.g., 10 for every 10%)
        legend_loc: Location for the legend (matplotlib location string or tuple)
        subplot_layout: If True, create side-by-side subplots
        left_ylabel: Y-axis label for left subplot
        left_data_files: List of data files for left subplot [baseline_file, method_files]
        left_ylim_from_zero: If True, left subplot y-axis starts from 0
        left_ylim_custom: Tuple of (min, max) for custom left subplot y-axis limits
    """
    # Read baseline data
    print(f"Reading baseline data from: {baseline_file}")
    time_baseline, coverage_baseline = read_coverage_file(baseline_file)
    
    # Read method data
    method_data = []
    for i, method_file in enumerate(method_files):
        print(f"Reading method {i+1} data from: {method_file}")
        time_method, coverage_method = read_coverage_file(method_file)
        method_data.append((time_method, coverage_method))
    
    # Validate data consistency
    baseline_length = len(time_baseline)
    for i, (time_method, coverage_method) in enumerate(method_data):
        if len(time_method) != baseline_length:
            print(f"Warning: Different number of data points in method {i+1} file")
    
    # Set method names
    baseline_label = baseline_name if baseline_name else 'Baseline Method'
    method_labels = []
    if method_names and len(method_names) >= len(method_files):
        method_labels = method_names[:len(method_files)]
    else:
        method_labels = [f'Method {i+1}' for i in range(len(method_files))]
    
    # Parse line colors if provided
    if line_colors:
        colors = [color.strip() for color in line_colors.split(',')]
        # Ensure we have enough colors for baseline + all methods
        total_needed = 1 + len(method_files)
        while len(colors) < total_needed:
            colors.extend(DEFAULT_COLORS[:total_needed - len(colors)])
    else:
        colors = DEFAULT_COLORS[:1 + len(method_files)]
    
    # Calculate common variables for both single and subplot layouts
    all_time_points = [time_baseline] + [time_method for time_method, _ in method_data]
    all_coverage_values = [coverage_baseline] + [coverage_method for _, coverage_method in method_data]
    
    max_time = max(max(time_points) for time_points in all_time_points)
    min_coverage = min(min(coverage_values) for coverage_values in all_coverage_values)
    max_coverage = max(max(coverage_values) for coverage_values in all_coverage_values)
    
    # Create the main comparison plot
    if subplot_layout:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Wider and shorter to match reference
        
        # Left subplot - additional data (e.g., CVEs found)
        if left_data_files:
            left_baseline_file = left_data_files[0] if len(left_data_files) > 0 else baseline_file
            left_method_files = left_data_files[1:] if len(left_data_files) > 1 else method_files
            
            # Read left subplot data
            time_left_baseline, coverage_left_baseline = read_coverage_file(left_baseline_file)
            left_method_data = []
            for method_file in left_method_files:
                time_left_method, coverage_left_method = read_coverage_file(method_file)
                left_method_data.append((time_left_method, coverage_left_method))
            
            # Plot left subplot with larger markers and thicker lines
            ax1.plot(time_left_baseline, coverage_left_baseline, color=colors[0], linestyle='-', linewidth=3, 
                    label=baseline_label, marker=DEFAULT_MARKERS[0], markersize=4, alpha=0.9)
            
            for i, ((time_left_method, coverage_left_method), method_label) in enumerate(zip(left_method_data, method_labels)):
                color_idx = (i + 1) % len(colors)
                marker_idx = (i + 1) % len(DEFAULT_MARKERS)
                ax1.plot(time_left_method, coverage_left_method, color=colors[color_idx], linestyle='-', 
                        linewidth=3, label=method_label, marker=DEFAULT_MARKERS[marker_idx], 
                        markersize=4, alpha=0.9)
        
        # Right subplot - coverage data with larger markers and thicker lines
        ax2.plot(time_baseline, coverage_baseline, color=colors[0], linestyle='-', linewidth=3, 
                label=baseline_label, marker=DEFAULT_MARKERS[0], markersize=4, alpha=0.9)
        
        for i, ((time_method, coverage_method), method_label) in enumerate(zip(method_data, method_labels)):
            color_idx = (i + 1) % len(colors)
            marker_idx = (i + 1) % len(DEFAULT_MARKERS)
            ax2.plot(time_method, coverage_method, color=colors[color_idx], linestyle='-', 
                    linewidth=3, label=method_label, marker=DEFAULT_MARKERS[marker_idx], 
                    markersize=4, alpha=0.9)
        
        # Customize left subplot with larger fonts
        ax1.set_xlabel('Time (min.)', fontsize=18)
        ax1.set_ylabel(left_ylabel, fontsize=18)
        ax1.grid(True, alpha=0.4, linewidth=0.8)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # Customize right subplot with larger fonts
        ax2.set_xlabel('Time (min.)', fontsize=18)
        ax2.set_ylabel(ylabel, fontsize=18)
        ax2.grid(True, alpha=0.4, linewidth=0.8)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        
        # Add legend to the right subplot only with larger font
        ax2.legend(fontsize=16, loc=legend_loc, ncol=1, frameon=True, 
                  fancybox=True, shadow=False, framealpha=0.95, borderpad=0.8)
        
        # Set limits for both subplots
        if left_data_files:
            all_left_time = [time_left_baseline] + [time_left_method for time_left_method, _ in left_method_data]
            all_left_coverage = [coverage_left_baseline] + [coverage_left_method for _, coverage_left_method in left_method_data]
            max_left_time = max(max(time_points) for time_points in all_left_time)
            min_left_coverage = min(min(coverage_values) for coverage_values in all_left_coverage)
            max_left_coverage = max(max(coverage_values) for coverage_values in all_left_coverage)
            
            ax1.set_xlim(0, max_left_time)
            
            # Set left subplot y-axis limits
            if left_ylim_custom is not None:
                # Use custom y-axis limits (highest priority)
                ax1.set_ylim(left_ylim_custom[0], left_ylim_custom[1])
            elif left_ylim_from_zero:
                # Start y-axis from 0
                ax1.set_ylim(0, max_left_coverage + (max_left_coverage - min_left_coverage) * 0.05)
            else:
                # Use optimized view
                y_range = max_left_coverage - min_left_coverage
                ax1.set_ylim(min_left_coverage - y_range * 0.1, max_left_coverage + y_range * 0.05)
        
        # Set right subplot limits
        ax2.set_xlim(0, max_time)
        if ylim_custom is not None:
            ax2.set_ylim(ylim_custom[0], ylim_custom[1])
        elif ylim_from_zero:
            ax2.set_ylim(0, max_coverage + (max_coverage - min_coverage) * 0.05)
        else:
            all_coverage_flat = [c for coverage_values in all_coverage_values for c in coverage_values]
            min_coverage_nonzero = min([c for c in all_coverage_flat if c > 0])
            y_range = max_coverage - min_coverage_nonzero
            ax2.set_ylim(min_coverage_nonzero - y_range * 0.1, max_coverage + y_range * 0.05)
        
    else:
        # Single plot layout (original behavior)
        plt.figure(figsize=(12, 8))
        
        # Plot baseline with thicker lines and larger markers
        plt.plot(time_baseline, coverage_baseline, color=colors[0], linestyle='-', linewidth=3, 
                 label=baseline_label, marker=DEFAULT_MARKERS[0], markersize=4, alpha=0.9)
        
        # Plot all methods with thicker lines and larger markers
        for i, ((time_method, coverage_method), method_label) in enumerate(zip(method_data, method_labels)):
            color_idx = (i + 1) % len(colors)
            marker_idx = (i + 1) % len(DEFAULT_MARKERS)
            plt.plot(time_method, coverage_method, color=colors[color_idx], linestyle='-', 
                    linewidth=3, label=method_label, marker=DEFAULT_MARKERS[marker_idx], 
                    markersize=4, alpha=0.9)
        
        # Customize the plot with larger fonts
        plt.xlabel('Time (minutes)', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.grid(True, alpha=0.4, linewidth=0.8)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # Place legend with configurable location and larger font
        plt.legend(fontsize=16, loc=legend_loc, ncol=1, frameon=True, 
                  fancybox=True, shadow=False, framealpha=0.95, borderpad=0.8)
        
        # Set appropriate limits
        plt.xlim(0, max_time)
        
        if ylim_custom is not None:
            plt.ylim(ylim_custom[0], ylim_custom[1])
        elif ylim_from_zero:
            plt.ylim(0, max_coverage + (max_coverage - min_coverage) * 0.05)
        else:
            all_coverage_flat = [c for coverage_values in all_coverage_values for c in coverage_values]
            min_coverage_nonzero = min([c for c in all_coverage_flat if c > 0])
            y_range = max_coverage - min_coverage_nonzero
            plt.ylim(min_coverage_nonzero - y_range * 0.1, max_coverage + y_range * 0.05)
    
    # Set custom tick intervals
    if xtick_interval is not None or ytick_interval is not None:
        if subplot_layout:
            # Apply to both subplots
            for ax in [ax1, ax2] if left_data_files else [ax2]:
                if xtick_interval is not None:
                    xmin, xmax = ax.get_xlim()
                    xticks = np.arange(0, xmax + xtick_interval, xtick_interval)
                    ax.set_xticks(xticks)
                
                if ytick_interval is not None:
                    ymin, ymax = ax.get_ylim()
                    ymin_rounded = np.floor(ymin / ytick_interval) * ytick_interval
                    ymax_rounded = np.ceil(ymax / ytick_interval) * ytick_interval
                    yticks = np.arange(ymin_rounded, ymax_rounded + ytick_interval, ytick_interval)
                    ax.set_yticks(yticks)
        else:
            # Single plot
            if xtick_interval is not None:
                xmin, xmax = plt.xlim()
                xticks = np.arange(0, xmax + xtick_interval, xtick_interval)
                plt.xticks(xticks)
            
            if ytick_interval is not None:
                ymin, ymax = plt.ylim()
                ymin_rounded = np.floor(ymin / ytick_interval) * ytick_interval
                ymax_rounded = np.ceil(ymax / ytick_interval) * ytick_interval
                yticks = np.arange(ymin_rounded, ymax_rounded + ytick_interval, ytick_interval)
                plt.yticks(yticks)
        
    # Add custom legend text if provided
    if legend_text:
        if subplot_layout:
            # Add text to the figure (spans both subplots)
            fig.text(0.98, 0.02, legend_text, 
                    fontsize=20, fontweight='bold',
                    verticalalignment='bottom', horizontalalignment='right',
                    color=legend_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            plt.text(0.98, 0.02, legend_text, 
                    transform=plt.gca().transAxes, fontsize=20, fontweight='bold',
                    verticalalignment='bottom', horizontalalignment='right',
                    color=legend_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
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
    
    if len(coverage_baseline) > 0 and all(len(coverage_method) > 0 for _, coverage_method in method_data):
        # Baseline statistics
        print(f"{baseline_label}:")
        print(f"  - Coverage at 1min: {coverage_baseline[1]:.2f}%")
        print(f"  - Final coverage: {coverage_baseline[-1]:.2f}%")
        print(f"  - Total improvement: {coverage_baseline[-1] - coverage_baseline[0]:.2f}%")
        
        # Method statistics
        for i, ((_, coverage_method), method_label) in enumerate(zip(method_data, method_labels)):
            print(f"\n{method_label}:")
            print(f"  - Coverage at 1min: {coverage_method[1]:.2f}%")
            print(f"  - Final coverage: {coverage_method[-1]:.2f}%")
            print(f"  - Total improvement: {coverage_method[-1] - coverage_method[0]:.2f}%")
        
        # Comparison statistics
        print(f"\nComparison vs {baseline_label}:")
        for i, ((_, coverage_method), method_label) in enumerate(zip(method_data, method_labels)):
            print(f"  - {method_label} advantage at 1min: {coverage_method[1] - coverage_baseline[1]:.2f}%")
            print(f"  - {method_label} final advantage: {coverage_method[-1] - coverage_baseline[-1]:.2f}%")
        
        # Calculate improvement rate in first 5 minutes
        if len(coverage_baseline) >= 6 and all(len(coverage_method) >= 6 for _, coverage_method in method_data):
            baseline_5min_rate = (coverage_baseline[5] - coverage_baseline[0]) / 5
            print(f"\n  - Avg. improvement rate (0-5min):")
            print(f"    {baseline_label}: {baseline_5min_rate:.2f}%/min")
            
            for i, ((_, coverage_method), method_label) in enumerate(zip(method_data, method_labels)):
                method_5min_rate = (coverage_method[5] - coverage_method[0]) / 5
                print(f"    {method_label}: {method_5min_rate:.2f}%/min")

def main():
    parser = argparse.ArgumentParser(description='Compare fuzzing coverage between baseline and multiple methods')
    parser.add_argument('baseline_file', help='File containing baseline method coverage data')
    parser.add_argument('method_files', nargs='+', help='Files containing method coverage data (can specify multiple files)')
    parser.add_argument('-o', '--output', help='Output file for saving plots (optional)')
    parser.add_argument('-l', '--legend', help='Custom legend text to display on the plot')
    parser.add_argument('--baseline-name', help='Custom name for the baseline method (default: "Baseline Method")')
    parser.add_argument('--method-names', nargs='+', help='Custom names for methods (space-separated, e.g., --method-names "Method A" "Method B" "Method C")')
    parser.add_argument('--colors', help='Line colors for baseline and all methods (comma-separated hex colors, e.g., "#1f77b4,#d62728,#2ca02c")')
    parser.add_argument('-c', '--color', default='black', help='Color for the legend text (default: black)')
    parser.add_argument('--ylabel', default='Instruction Coverage (%)', help='Y-axis label text (default: "Instruction Coverage (%)")')
    parser.add_argument('--ylim-from-zero', action='store_true', help='Start y-axis from 0 instead of optimized view (default: False)')
    parser.add_argument('--ylim', nargs=2, type=float, metavar=('MIN', 'MAX'), help='Set custom y-axis limits (e.g., --ylim 10 90)')
    parser.add_argument('--xtick-interval', type=float, help='Interval for x-axis ticks (e.g., 5 for every 5 minutes)')
    parser.add_argument('--ytick-interval', type=float, help='Interval for y-axis ticks (e.g., 10 for every 10 percent)')
    parser.add_argument('--legend-loc', default='lower right', help='Legend location (e.g., "upper left", "lower right", "center", "best", or coordinates like "0.1,0.9")')
    parser.add_argument('--subplot-layout', action='store_true', help='Create side-by-side subplots instead of single plot')
    parser.add_argument('--left-ylabel', default='Total # of CVEs found', help='Y-axis label for left subplot (only used with --subplot-layout)')
    parser.add_argument('--left-data-files', nargs='+', help='Data files for left subplot: baseline_file followed by method_files (only used with --subplot-layout)')
    parser.add_argument('--left-ylim-from-zero', action='store_true', help='Start left subplot y-axis from 0 instead of optimized view (only used with --subplot-layout)')
    parser.add_argument('--left-ylim', nargs=2, type=float, metavar=('MIN', 'MAX'), help='Set custom y-axis limits for left subplot (e.g., --left-ylim 0 30, only used with --subplot-layout)')
    
    args = parser.parse_args()
    
    # Convert ylim to tuple if provided
    ylim_custom = tuple(args.ylim) if args.ylim else None
    left_ylim_custom = tuple(args.left_ylim) if args.left_ylim else None
    
    # Parse legend location - handle both string locations and coordinate tuples
    legend_loc = args.legend_loc
    if ',' in legend_loc:
        # Handle coordinate format like "0.1,0.9"
        try:
            coords = [float(x.strip()) for x in legend_loc.split(',')]
            if len(coords) == 2:
                legend_loc = tuple(coords)
        except ValueError:
            print(f"Warning: Invalid legend location coordinates '{args.legend_loc}', using 'lower right'")
            legend_loc = 'lower right'
    
    # Generate the comparison plots
    plot_coverage_comparison(args.baseline_file, args.method_files, args.output, args.legend, 
                           args.color, args.colors, args.baseline_name, args.method_names, args.ylabel, 
                           args.ylim_from_zero, ylim_custom, args.xtick_interval, args.ytick_interval, legend_loc,
                           args.subplot_layout, args.left_ylabel, args.left_data_files,
                           args.left_ylim_from_zero, left_ylim_custom)

if __name__ == "__main__":
    main()

# Example usage (if running without command line arguments):
# Uncomment the lines below and modify the filenames as needed

# Single plot (original behavior):
# plot_coverage_comparison('baseline_coverage.txt', 
#                         ['method1_coverage.txt', 'method2_coverage.txt', 'method3_coverage.txt', 'method4_coverage.txt'], 
#                         output_file='comparison.pdf',
#                         legend_text='Fuzzing Comparison', legend_color='blue', 
#                         line_colors='#1f77b4,#d62728,#2ca02c,#ff7f0e,#9467bd',
#                         baseline_name='Baseline', 
#                         method_names=['Method A', 'Method B', 'Method C', 'Method D'],
#                         ylabel='Branch Coverage (%)',
#                         ylim_from_zero=True,
#                         ylim_custom=(10, 90),
#                         xtick_interval=5,
#                         ytick_interval=10,
#                         legend_loc='upper left')

# Subplot layout (like the reference image):
# plot_coverage_comparison('baseline_coverage.txt', 
#                         ['method1_coverage.txt', 'method2_coverage.txt', 'method3_coverage.txt'], 
#                         output_file='comparison_subplots.pdf',
#                         baseline_name='sFuzz', 
#                         method_names=['Smartian', 'Manticore', 'Mythril'],
#                         ylabel='Instruction Coverage',
#                         subplot_layout=True,
#                         left_ylabel='Total # of CVEs found',
#                         left_data_files=['baseline_cves.txt', 'method1_cves.txt', 'method2_cves.txt', 'method3_cves.txt'],
#                         legend_loc='upper left')
