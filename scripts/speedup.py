import re
import os
import numpy as np
from typing import Dict, List, Tuple, Set

def extract_times(line: str) -> List[int]:
    """Extract the execution times from a line of output."""
    match = re.search(r'\[(.*?)\]', line)
    if match:
        # Extract numbers from inside the brackets
        times_str = match.group(1)
        # Handle both comma-separated and space-separated times
        if ',' in times_str:
            times = [int(t.strip()) for t in times_str.split(',') if t.strip().isdigit()]
        else:
            times = [int(t.strip()) for t in times_str.split() if t.strip().isdigit()]
        return times
    return []

def parse_vulnerability_file(file_path: str) -> Dict[str, Dict[str, List[int]]]:
    """Parse a vulnerability detection output file and extract execution times."""
    results = {}
    current_type = None
    global_metrics = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if this line indicates a new vulnerability type or global metrics
            if line.startswith("==================================="):
                # Check if the next lines contain global metrics
                if i < len(lines) and re.match(r'\d+m:', lines[i].strip()):
                    # Parse global metrics
                    while i < len(lines) and not lines[i].strip().startswith("==================================="):
                        metric_line = lines[i].strip()
                        i += 1
                        
                        if re.match(r'\d+m:', metric_line):
                            parts = metric_line.split(':')
                            if len(parts) == 2:
                                time_min = parts[0].replace('m', '')
                                value = parts[1].strip()
                                global_metrics[f"{time_min}m"] = float(value)
                        elif "bugs were found" in metric_line:
                            global_metrics[metric_line] = True
                            
                current_type = None
                continue
            
            # Check for vulnerability type from line format
            match = re.match(r'(Fully|Partly|Never) found (\w+) from', line)
            if match:
                # If this is the first instance of this vulnerability type
                vuln_type = match.group(2)
                if vuln_type not in results:
                    results[vuln_type] = {}
                current_type = vuln_type
            
            # If we have a current type, extract contract info and times
            if current_type:
                # Match traditional hex address or contract name format
                match1 = re.match(r'(Fully|Partly|Never) found \w+ from (0x[a-fA-F0-9]+|[a-zA-Z_]+)(:? \[(.*?)\] sec)?', line)
                # Match CVE-style ID format
                match2 = re.match(r'(Fully|Partly|Never) found \w+ from (\d{4}-\d+)(:? \[(.*?)\] sec)?', line)
                
                match = match1 or match2
                if match:
                    status = match.group(1)
                    contract = match.group(2)
                    
                    if status == "Never":
                        # No times for failures
                        times = []
                    else:
                        times = extract_times(line)
                    
                    results[current_type][contract] = times
    
    # Store global metrics in a special key
    if global_metrics:
        results["__global_metrics__"] = global_metrics
    
    return results

def calculate_average_times(results: Dict[str, Dict[str, List[int]]]) -> Dict[str, Dict[str, float]]:
    """Calculate average execution times for each contract and vulnerability type."""
    avg_times = {}
    
    for vuln_type, contracts in results.items():
        # Skip global metrics
        if vuln_type == "__global_metrics__":
            avg_times[vuln_type] = contracts
            continue
            
        avg_times[vuln_type] = {}
        
        for contract, times in contracts.items():
            if times:  # If we have times (not a "Never found" case)
                avg_times[vuln_type][contract] = sum(times) / len(times)
            # else:
            #     avg_times[vuln_type][contract] = float('inf')  # Use infinity for never found
    
    return avg_times

def calculate_speedups(baseline_avgs: Dict[str, Dict[str, float]], 
                      method_avgs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Calculate speedup ratios (baseline / method) for each contract and vulnerability type.
    Values > 1.0 mean method is faster than baseline.
    Values < 1.0 mean method is slower than baseline."""
    speedups = {}
    
    # Initialize all vulnerability types from both methods
    all_vuln_types = set(baseline_avgs.keys()) & set(method_avgs.keys())
    
    # For global metrics section
    global_speedups = {}
    
    for vuln_type in all_vuln_types:
        # Special handling for global metrics
        if vuln_type == "__global_metrics__":
            baseline_metrics = baseline_avgs.get(vuln_type, {})
            method_metrics = method_avgs.get(vuln_type, {})
            
            # Calculate speedups for time-based metrics
            for time_point in set(baseline_metrics.keys()) & set(method_metrics.keys()):
                if time_point.endswith('m') and time_point in baseline_metrics and time_point in method_metrics:
                    baseline_value = baseline_metrics[time_point]
                    method_value = method_metrics[time_point]
                    
                    # Calculate detection rate speedup (if method finds more bugs in same time)
                    if baseline_value > 0 and method_value > 0:
                        global_speedups[f"{time_point}_speedup"] = method_value / baseline_value
            
            continue
            
        speedups[vuln_type] = {}
        
        # Get all contracts from both methods for this vulnerability type
        baseline_contracts = baseline_avgs.get(vuln_type, {})
        method_contracts = method_avgs.get(vuln_type, {})
        all_contracts = set(baseline_contracts.keys()) & set(method_contracts.keys())
        
        for contract in all_contracts:
            baseline_time = baseline_contracts.get(contract, float('inf'))
            method_time = method_contracts.get(contract, float('inf'))
            
            # Ensure we don't have zeros that would cause division errors
            if baseline_time == 0:
                baseline_time = 0.1  # Avoid division by zero
            
            if method_time == 0:
                method_time = 0.1  # Avoid division by zero
                
            if baseline_time == float('inf') and method_time == float('inf'):
                # Both methods failed
                speedups[vuln_type][contract] = 1.0
            elif baseline_time == float('inf'):
                # Baseline failed, method succeeded
                speedups[vuln_type][contract] = float('inf')
            elif method_time == float('inf'):
                # Method failed, baseline succeeded
                speedups[vuln_type][contract] = 0.0
            else:
                # Both methods succeeded - Method / Baseline ratio
                # For speedup: > 1.0 means method is faster (baseline/method), < 1.0 means baseline is faster
                speedups[vuln_type][contract] = baseline_time / method_time
    
    # Add global speedups to the result
    if global_speedups:
        speedups["__global_speedups__"] = global_speedups
    
    return speedups

def calculate_time_to_baseline_max(baseline_avgs, method_avgs):
    """Calculate the speedup in time to reach baseline's maximum bug detection."""
    global_metrics_baseline = baseline_avgs.get("__global_metrics__", {})
    global_metrics_method = method_avgs.get("__global_metrics__", {})
    
    # Find maximum detection rate in baseline
    max_baseline_rate = 0
    max_baseline_time = None
    
    for time_point, value in global_metrics_baseline.items():
        if time_point.endswith('m') and isinstance(value, (int, float)) and value > max_baseline_rate:
            max_baseline_rate = value
            max_baseline_time = time_point
    
    if not max_baseline_time:
        return None
    
    # Find when method reaches this detection rate
    baseline_minutes = int(max_baseline_time.replace('m', ''))
    method_time_to_reach = None
    
    # Sort method time points
    sorted_method_times = sorted(
        [(int(k.replace('m', '')), v) for k, v in global_metrics_method.items() if k.endswith('m')],
        key=lambda x: x[0]
    )
    
    for minutes, rate in sorted_method_times:
        if rate >= max_baseline_rate:
            method_time_to_reach = minutes
            break
    
    if method_time_to_reach is None:
        return None
    
    # Calculate speedup
    speedup = baseline_minutes / method_time_to_reach if method_time_to_reach > 0 else float('inf')
    return {
        'baseline_max_rate': max_baseline_rate,
        'baseline_time': baseline_minutes,
        'method_time': method_time_to_reach,
        'speedup': speedup
    }

def print_global_metrics(baseline_avgs, method_avgs, speedups):
    """Print the global performance metrics for both methods."""
    global_metrics_baseline = baseline_avgs.get("__global_metrics__", {})
    global_metrics_method = method_avgs.get("__global_metrics__", {})
    global_speedups = speedups.get("__global_speedups__", {})
    
    if global_metrics_baseline or global_metrics_method:
        print("\n## Global Performance Metrics")
        print("------------------------------------------------")
        
        # Print time-based metrics
        time_points = sorted(set([k for k in global_metrics_baseline.keys() if 'm' in k] + 
                                 [k for k in global_metrics_method.keys() if 'm' in k]))
        
        if time_points:
            print("\n### Detection Rate Over Time")
            print(f"{'Time':<10} {'Baseline':<15} {'Method':<15} {'Improvement':<15} {'Speedup':<10}")
            print("-" * 65)
            
            for time_point in time_points:
                value_baseline = global_metrics_baseline.get(time_point, "-")
                value_method = global_metrics_method.get(time_point, "-")
                
                improvement = "-"
                speedup_value = "-"
                
                if isinstance(value_baseline, (int, float)) and isinstance(value_method, (int, float)) and value_baseline > 0:
                    ratio = value_method / value_baseline
                    if ratio > 1:
                        improvement = f"+{(ratio-1)*100:.2f}%"
                    else:
                        improvement = f"{(ratio-1)*100:.2f}%"
                    
                    # Add speedup value from global_speedups
                    speedup_key = f"{time_point}_speedup"
                    if speedup_key in global_speedups:
                        speedup_value = f"{global_speedups[speedup_key]:.2f}x"
                
                print(f"{time_point:<10} {value_baseline:<15} {value_method:<15} {improvement:<15} {speedup_value:<10}")
        
        # Print time to reach baseline max detection rate
        time_to_max = calculate_time_to_baseline_max(baseline_avgs, method_avgs)
        if time_to_max:
            print("\n### Time to Reach Baseline Maximum Detection Rate")
            print(f"Baseline maximum detection rate: {time_to_max['baseline_max_rate']:.2f} (at {time_to_max['baseline_time']}m)")
            print(f"Method reaches this rate at: {time_to_max['method_time']}m")
            print(f"Speedup to reach baseline maximum: {time_to_max['speedup']:.2f}x")
        
        # Print bug statistics
        bug_stats = [k for k in global_metrics_baseline.keys() if "bugs were found" in k]
        bug_stats += [k for k in global_metrics_method.keys() if "bugs were found" in k and k not in bug_stats]
        
        if bug_stats:
            print("\n### Bug Detection Statistics")
            
            print("\nBaseline:")
            for stat in sorted(global_metrics_baseline.keys()):
                if "bugs were found" in stat:
                    print(f"  {stat}")
                    
            print("\nMethod:")
            for stat in sorted(global_metrics_method.keys()):
                if "bugs were found" in stat:
                    print(f"  {stat}")

def calculate_geometric_mean(values):
    """Calculate the geometric mean of a list of values."""
    if not values:
        return None
    
    # For geometric mean, we need to handle only positive values
    positive_values = [v for v in values if v > 0]
    
    if not positive_values:
        return None
    
    # Calculate using numpy's geometric mean function
    return np.exp(np.mean(np.log(positive_values)))

def print_speedup_summary(speedups: Dict[str, Dict[str, float]], 
                         baseline_avgs: Dict[str, Dict[str, float]], 
                         method_avgs: Dict[str, Dict[str, float]]):
    """Print a summary of speedup statistics by vulnerability type."""
    print("## Speedup Summary (New Method vs Baseline)")
    print("------------------------------------------------")
    # print("Note: Values > 1.0 mean the new method is faster than baseline")
    # print("      Values < 1.0 mean the new method is slower than baseline")
    print()
    
    # Calculate overall statistics
    all_valid_speedups = []
    for vuln_type, contracts in speedups.items():
        # Skip special keys
        if vuln_type in ["__global_metrics__", "__global_speedups__"]:
            continue
            
        for contract, speedup in contracts.items():
            if 0 < speedup < float('inf'):
                all_valid_speedups.append(speedup)
    
    if all_valid_speedups:
        overall_avg = np.mean(all_valid_speedups)
        overall_median = np.median(all_valid_speedups)
        overall_geo_mean = calculate_geometric_mean(all_valid_speedups)
        better_count = sum(1 for s in all_valid_speedups if s > 1.0)
        worse_count = sum(1 for s in all_valid_speedups if s < 1.0)
        same_count = sum(1 for s in all_valid_speedups if s == 1.0)
        
        print(f"Overall Average Speedup: {overall_avg:.2f}x")
        print(f"Overall Median Speedup: {overall_median:.2f}x")
        print(f"Overall Geometric Mean Speedup: {overall_geo_mean:.2f}x")
        print(f"New method faster: {better_count} vulnerabilities  ({better_count/len(all_valid_speedups)*100:.1f}%)")
        print(f"Baseline faster: {worse_count} vulnerabilities  ({worse_count/len(all_valid_speedups)*100:.1f}%)")
        if same_count:
            print(f"Equal performance: {same_count} vulnerabilities  ({same_count/len(all_valid_speedups)*100:.1f}%)")
        print()
    
    # Calculate and print statistics by vulnerability type
    print("## Speedup by Vulnerability Type")
    print("------------------------------------------------")
    
    for vuln_type in sorted(speedups.keys()):
        # Skip special keys
        if vuln_type in ["__global_metrics__", "__global_speedups__"]:
            continue
            
        contracts = speedups[vuln_type]
        
        # Skip empty vulnerability types
        if not contracts:
            continue
            
        type_speedups = [s for s in contracts.values() if 0 < s < float('inf')]
        
        print(f"\n### {vuln_type}")
        
        if type_speedups:
            avg_speedup = np.mean(type_speedups)
            median_speedup = np.median(type_speedups)
            geo_mean_speedup = calculate_geometric_mean(type_speedups)
            max_speedup = max(type_speedups)
            min_speedup = min(type_speedups)
            
            print(f"Average Speedup: {avg_speedup:.2f}x")
            print(f"Median Speedup: {median_speedup:.2f}x")
            print(f"Geometric Mean Speedup: {geo_mean_speedup:.2f}x")
            print(f"Range: {min_speedup:.2f}x - {max_speedup:.2f}x")
            
            # Count cases where each method is better
            method_better = sum(1 for s in contracts.values() if s > 1.0)
            baseline_better = sum(1 for s in contracts.values() if 0 < s < 1.0)
            method_only = sum(1 for s in contracts.values() if s == float('inf'))
            baseline_only = sum(1 for s in contracts.values() if s == 0.0)
            
            print(f"New method faster: {method_better} contracts")
            print(f"Baseline faster: {baseline_better} contracts")
            print(f"Only new method succeeded: {method_only} contracts")
            print(f"Only baseline succeeded: {baseline_only} contracts")
            
            # Print details for individual contracts (limit to top/bottom 10 if more than 20 contracts)
            print("\nContract Details (sorted by speedup):")
            sorted_contracts = sorted(contracts.items(), key=lambda x: (0 if x[1] == 0 else float('inf') if x[1] == float('inf') else x[1]), reverse=True)
            
            # Print all contracts
            for contract, speedup in sorted_contracts:
                baseline_time = baseline_avgs.get(vuln_type, {}).get(contract, float('inf'))
                method_time = method_avgs.get(vuln_type, {}).get(contract, float('inf'))
                
                if speedup == float('inf'):
                    print(f"  {contract}: Only new method succeeded ({method_time:.1f} sec)")
                elif speedup == 0.0:
                    print(f"  {contract}: Only baseline succeeded ({baseline_time:.1f} sec)")
                else:
                    print(f"  {contract}: {speedup:.2f}x ({baseline_time:.1f} sec → {method_time:.1f} sec)")
        else:
            print("No valid speedup data available for this vulnerability type")

def calculate_total_time_speedups(baseline_avgs, method_avgs):
    """Calculate speedup based on the sum of all execution times."""
    
    # Calculate total execution time across all contracts and vulnerability types
    baseline_total = 0
    method_total = 0
    
    # Track totals by vulnerability type
    vuln_totals = {}
    
    for vuln_type in baseline_avgs.keys():
        # Skip special keys
        if vuln_type in ["__global_metrics__", "__global_speedups__"]:
            continue
            
        baseline_vuln_total = 0
        method_vuln_total = 0
        
        for contract, time in baseline_avgs.get(vuln_type, {}).items():
            if time != float('inf'):
                baseline_total += time
                baseline_vuln_total += time
        
        for contract, time in method_avgs.get(vuln_type, {}).items():
            if time != float('inf'):
                method_total += time
                method_vuln_total += time
        
        # Store totals for this vulnerability type
        if baseline_vuln_total > 0 or method_vuln_total > 0:
            if method_vuln_total == 0:
                method_vuln_total = 0.001  # Avoid division by zero, use 1ms
                
            if baseline_vuln_total == 0:
                baseline_vuln_total = 0.001  # Avoid division by zero, use 1ms
                
            vuln_totals[vuln_type] = {
                'baseline_total': baseline_vuln_total,
                'method_total': method_vuln_total,
                'speedup': baseline_vuln_total / method_vuln_total
            }
    
    # Avoid division by zero
    if method_total == 0:
        method_total = 0.001  # Use 1ms as minimum
        
    if baseline_total == 0:
        baseline_total = 0.001  # Use 1ms as minimum
    
    total_speedup = baseline_total / method_total
    
    return {
        'overall': {
            'baseline_total': baseline_total,
            'method_total': method_total,
            'speedup': total_speedup
        },
        'by_vulnerability': vuln_totals
    }

def print_total_time_speedups(total_speedups):
    """Print the speedup statistics based on total execution times."""
    
    print("\n## Total Execution Time Speedups")
    print("------------------------------------------------")
    
    overall = total_speedups['overall']
    print(f"Overall Total Time Speedup: {overall['speedup']:.2f}x")
    print(f"  Baseline Total Time: {overall['baseline_total']:.1f} seconds")
    print(f"  New Method Total Time: {overall['method_total']:.1f} seconds")
    print(f"  Time Saved: {overall['baseline_total'] - overall['method_total']:.1f} seconds")
    
    print("\n### Total Time by Vulnerability Type")
    print(f"{'Vulnerability Type':<25} {'Baseline Time (s)':<20} {'Method Time (s)':<20} {'Speedup':<10} {'Time Saved (s)':<15}")
    print("-" * 90)
    
    # Sort vulnerability types by speedup (highest first)
    sorted_vuln_types = sorted(
        total_speedups['by_vulnerability'].items(),
        key=lambda x: x[1]['speedup'],
        reverse=True
    )
    
    # Create a list of all valid speedups for geometric mean calculation
    all_speedups = [data['speedup'] for _, data in sorted_vuln_types if 0 < data['speedup'] < float('inf')]
    if all_speedups:
        geo_mean = calculate_geometric_mean(all_speedups)
        print(f"Geometric Mean Speedup across vulnerability types: {geo_mean:.2f}x\n")
    
    for vuln_type, data in sorted_vuln_types:
        baseline_time = data['baseline_total']
        method_time = data['method_total']
        speedup = data['speedup']
        time_saved = baseline_time - method_time
        
        print(f"{vuln_type:<25} {baseline_time:<20.1f} {method_time:<20.1f} {speedup:<10.2f}x {time_saved:<15.1f}")

# Modify the process_files function to include the new total time speedup calculation
def process_files(baseline_file: str, method_file: str):
    """Process two output files and calculate speedups."""
    print(f"Processing Baseline file: {baseline_file}")
    baseline_results = parse_vulnerability_file(baseline_file)
    
    print(f"Processing New Method file: {method_file}")
    method_results = parse_vulnerability_file(method_file)
    
    # Calculate average times
    baseline_avgs = calculate_average_times(baseline_results)
    method_avgs = calculate_average_times(method_results)
    
    # Calculate speedups (baseline / method)
    speedups = calculate_speedups(baseline_avgs, method_avgs)
    
    # Calculate total time speedups
    total_speedups = calculate_total_time_speedups(baseline_avgs, method_avgs)
    
    # Print summary
    print_speedup_summary(speedups, baseline_avgs, method_avgs)
    
    # # Print total time speedups
    # print_total_time_speedups(total_speedups)
    
    # Print global metrics
    print_global_metrics(baseline_avgs, method_avgs, speedups)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate vulnerability detection speedups of a method against a baseline')
    parser.add_argument('--baseline', required=True, help='File containing baseline method output')
    parser.add_argument('--method', required=True, help='File containing new method output')
    
    args = parser.parse_args()
    process_files(args.baseline, args.method)

if __name__ == "__main__":
    main()