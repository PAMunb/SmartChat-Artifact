import re
import os
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import scipy.stats as stats
from scipy import stats as scipy_stats
import warnings

def extract_times(line: str) -> List[float]:
    """Extract the execution times from a line of output."""
    # Look for patterns like [1, 2, 3] or [1, 2, 3,4,5] sec]
    match = re.search(r'\[(.*?)\]\s*sec\]?', line)
    if match:
        times_str = match.group(1)
        # Handle various separators (comma, space, comma+space)
        times_str = re.sub(r'[,\s]+', ',', times_str.strip())
        try:
            times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
            return times
        except ValueError as e:
            print(f"Warning: Could not parse times from '{times_str}': {e}")
            return []
    
    # Fallback: look for just brackets with numbers
    match = re.search(r'\[(.*?)\]', line)
    if match:
        times_str = match.group(1)
        times_str = re.sub(r'[,\s]+', ',', times_str.strip())
        try:
            times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
            return times
        except ValueError:
            return []
    
    return []

def parse_vulnerability_file(file_path: str) -> Dict[str, Dict[str, List[float]]]:
    """Parse a vulnerability detection output file and extract execution times."""
    results = {}
    current_type = None
    global_metrics = {}
    
    print(f"Parsing file: {file_path}")
    
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
                                try:
                                    value = float(parts[1].strip())
                                    global_metrics[f"{time_min}m"] = value
                                except ValueError:
                                    pass
                        elif "bugs were found" in metric_line:
                            global_metrics[metric_line] = True
                            
                current_type = None
                continue
            
            # Check for vulnerability type from line format
            match = re.match(r'(Fully|Partly|Never) found (\w+) from', line)
            if match:
                vuln_type = match.group(2)
                if vuln_type not in results:
                    results[vuln_type] = {}
                current_type = vuln_type
            
            # If we have a current type, extract contract info and times
            if current_type:
                # Match various contract identifier formats
                patterns = [
                    r'(Fully|Partly|Never) found \w+ from (0x[a-fA-F0-9]+)',  # Hex addresses
                    r'(Fully|Partly|Never) found \w+ from ([a-zA-Z_][a-zA-Z0-9_]*)',  # Contract names
                    r'(Fully|Partly|Never) found \w+ from (\d{4}-\d+)',  # CVE-style IDs
                ]
                
                match = None
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        break
                
                if match:
                    status = match.group(1)
                    contract = match.group(2)
                    
                    if status == "Never":
                        # No times for failures - store empty list
                        times = []
                    else:
                        times = extract_times(line)
                    
                    results[current_type][contract] = times
    
    # Store global metrics in a special key
    if global_metrics:
        results["__global_metrics__"] = global_metrics
    
    print(f"Parsed {len(results)} vulnerability types")
    for vuln_type, contracts in results.items():
        if vuln_type != "__global_metrics__":
            print(f"  {vuln_type}: {len(contracts)} contracts")
    
    return results

def validate_data(baseline_results: Dict, method_results: Dict) -> Dict[str, Set[str]]:
    """Validate that both datasets have overlapping contracts."""
    baseline_contracts = set()
    method_contracts = set()
    
    for vuln_type, contracts in baseline_results.items():
        if vuln_type != "__global_metrics__":
            baseline_contracts.update(contracts.keys())
    
    for vuln_type, contracts in method_results.items():
        if vuln_type != "__global_metrics__":
            method_contracts.update(contracts.keys())
    
    overlap = baseline_contracts & method_contracts
    
    print(f"\nData Validation:")
    print(f"Contracts in baseline: {len(baseline_contracts)}")
    print(f"Contracts in method: {len(method_contracts)}")
    print(f"Overlapping contracts: {len(overlap)}")
    
    if len(overlap) < 0.5 * min(len(baseline_contracts), len(method_contracts)):
        print("⚠️  Warning: Low overlap between datasets")
    
    return {
        'baseline': baseline_contracts,
        'method': method_contracts,
        'overlap': overlap
    }

def calculate_speedups_robust(baseline_results: Dict, method_results: Dict) -> Dict:
    """Calculate speedups with proper handling of edge cases."""
    speedup_data = {}
    all_speedups = []
    detailed_results = []
    
    for vuln_type in baseline_results.keys():
        if vuln_type == "__global_metrics__":
            continue
            
        speedup_data[vuln_type] = {}
        baseline_contracts = baseline_results.get(vuln_type, {})
        method_contracts = method_results.get(vuln_type, {})
        
        # Get contracts that exist in both methods
        common_contracts = set(baseline_contracts.keys()) & set(method_contracts.keys())
        
        for contract in common_contracts:
            baseline_times = baseline_contracts[contract]
            method_times = method_contracts[contract]
            
            # Handle different scenarios
            if not baseline_times and not method_times:
                # Both methods failed
                speedup = 1.0
                category = "both_failed"
            elif not baseline_times:
                # Only baseline failed, method succeeded
                if method_times:
                    speedup = float('inf')
                    category = "baseline_failed"
                else:
                    continue
            elif not method_times:
                # Only method failed, baseline succeeded
                speedup = 0.0
                category = "method_failed"
            else:
                # Both methods succeeded - calculate speedup
                baseline_avg = np.mean(baseline_times)
                method_avg = np.mean(method_times)
                
                # Handle zero times appropriately
                if baseline_avg == 0 and method_avg == 0:
                    speedup = 1.0  # Both instant
                    category = "both_instant"
                elif baseline_avg == 0:
                    speedup = 0.0  # Baseline was instant, method was not
                    category = "baseline_instant"
                elif method_avg == 0:
                    speedup = float('inf')  # Method was instant, baseline was not
                    category = "method_instant"
                else:
                    speedup = baseline_avg / method_avg
                    category = "normal"
            
            speedup_data[vuln_type][contract] = speedup
            
            # Store detailed information for analysis
            detailed_results.append({
                'vuln_type': vuln_type,
                'contract': contract,
                'speedup': speedup,
                'category': category,
                'baseline_times': baseline_times,
                'method_times': method_times,
                'baseline_avg': np.mean(baseline_times) if baseline_times else None,
                'method_avg': np.mean(method_times) if method_times else None
            })
            
            # Add to overall speedup list (excluding infinite and zero speedups for CI calculation)
            if 0 < speedup < float('inf'):
                all_speedups.append(speedup)
    
    return {
        'speedups': speedup_data,
        'all_speedups': all_speedups,
        'detailed_results': detailed_results
    }

def bootstrap_confidence_interval(data: List[float], confidence_level: float = 0.95, 
                                 n_bootstrap: int = 10000) -> Optional[Dict]:
    """Calculate confidence interval using bootstrapping - more robust for skewed data."""
    if len(data) < 2:
        return None
    
    # Remove any infinite or zero values for CI calculation
    clean_data = [x for x in data if 0 < x < float('inf')]
    if len(clean_data) < 2:
        return None
    
    # Bootstrap resampling
    bootstrap_means = []
    bootstrap_medians = []
    bootstrap_geomeans = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(clean_data, size=len(clean_data), replace=True)
        bootstrap_means.append(np.mean(sample))
        bootstrap_medians.append(np.median(sample))
        # Geometric mean for positive values
        bootstrap_geomeans.append(scipy_stats.gmean(sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    return {
        'mean': np.mean(clean_data),
        'mean_lower': np.percentile(bootstrap_means, lower_percentile),
        'mean_upper': np.percentile(bootstrap_means, upper_percentile),
        'median': np.median(clean_data),
        'median_lower': np.percentile(bootstrap_medians, lower_percentile),
        'median_upper': np.percentile(bootstrap_medians, upper_percentile),
        'geomean': scipy_stats.gmean(clean_data),
        'geomean_lower': np.percentile(bootstrap_geomeans, lower_percentile),
        'geomean_upper': np.percentile(bootstrap_geomeans, upper_percentile),
        'n': len(clean_data),
        'n_total': len(data)
    }

def detect_outliers(data: List[float], method: str = 'iqr') -> Dict:
    """Detect and categorize outliers in speedup data."""
    if len(data) < 4:
        return {'outliers': [], 'clean_data': data, 'bounds': None}
    
    data_array = np.array(data)
    
    if method == 'iqr':
        q1, q3 = np.percentile(data_array, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    elif method == 'zscore':
        mean = np.mean(data_array)
        std = np.std(data_array)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
    else:
        return {'outliers': [], 'clean_data': data, 'bounds': None}
    
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    clean_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return {
        'outliers': outliers,
        'clean_data': clean_data,
        'bounds': (lower_bound, upper_bound),
        'method': method
    }

def calculate_time_to_baseline_max(baseline_results: Dict, method_results: Dict) -> Optional[Dict]:
    """Calculate the speedup in time to reach baseline's maximum bug detection."""
    global_metrics_baseline = baseline_results.get("__global_metrics__", {})
    global_metrics_method = method_results.get("__global_metrics__", {})
    
    if not global_metrics_baseline or not global_metrics_method:
        return None
    
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
        [(int(k.replace('m', '')), v) for k, v in global_metrics_method.items() 
         if k.endswith('m') and isinstance(v, (int, float))],
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

def print_comprehensive_summary(speedup_result: Dict, confidence_level: float = 0.95):
    """Print a comprehensive summary with robust statistics."""
    speedups = speedup_result['speedups']
    all_speedups = speedup_result['all_speedups']
    detailed_results = speedup_result['detailed_results']
    
    print("## Speedup Summary (New Method vs Baseline)")
    print("------------------------------------------------")
    print()
    
    # Overall statistics
    if all_speedups:
        # Calculate bootstrap confidence intervals
        ci_result = bootstrap_confidence_interval(all_speedups, confidence_level)
        
        # Detect outliers
        outlier_analysis = detect_outliers(all_speedups)
        
        # Basic statistics
        overall_mean = np.mean(all_speedups)
        overall_median = np.median(all_speedups)
        overall_geomean = scipy_stats.gmean([x for x in all_speedups if x > 0])
        
        # Categorize results
        better_count = sum(1 for s in all_speedups if s > 1.0)
        worse_count = sum(1 for s in all_speedups if s < 1.0)
        same_count = sum(1 for s in all_speedups if s == 1.0)
        
        # Count special cases from detailed results
        infinite_speedups = sum(1 for r in detailed_results if r['speedup'] == float('inf'))
        zero_speedups = sum(1 for r in detailed_results if r['speedup'] == 0.0)
        
        print(f"Overall Average Speedup: {overall_mean:.2f}x")
        print(f"Overall Median Speedup: {overall_median:.2f}x")
        print(f"Overall Geometric Mean Speedup: {overall_geomean:.2f}x")
        print(f"New method faster: {better_count} vulnerabilities  ({better_count/len(all_speedups)*100:.1f}%)")
        print(f"Baseline faster: {worse_count} vulnerabilities  ({worse_count/len(all_speedups)*100:.1f}%)")
        if same_count:
            print(f"Equal performance: {same_count} vulnerabilities  ({same_count/len(all_speedups)*100:.1f}%)")
        if infinite_speedups:
            print(f"Only method succeeded: {infinite_speedups} vulnerabilities")
        if zero_speedups:
            print(f"Only baseline succeeded: {zero_speedups} vulnerabilities")
        print()
        
        # Outlier analysis
        if outlier_analysis['outliers']:
            print(f"Outliers detected ({outlier_analysis['method']} method): {len(outlier_analysis['outliers'])} values")
            print(f"Range of outliers: {min(outlier_analysis['outliers']):.2f}x - {max(outlier_analysis['outliers']):.2f}x")
            print()
        
        # Confidence intervals
        if ci_result:
            print(f"## Bootstrap Confidence Intervals ({confidence_level*100:.0f}% Confidence)")
            print("=" * 60)
            print()
            print(f"Mean Speedup: {ci_result['mean']:.2f}x")
            print(f"95% Confidence Interval: [{ci_result['mean_lower']:.2f}x, {ci_result['mean_upper']:.2f}x]")
            print(f"Number of measurements: {ci_result['n']}")
            
            # Statistical significance test
            if ci_result['mean_lower'] > 1.0:
                print("✅ **Statistically significant speedup** (lower bound > 1.0x)")
            elif ci_result['mean_upper'] < 1.0:
                print("❌ **Statistically significant slowdown** (upper bound < 1.0x)")
            else:
                print("⚠️  **No statistically significant difference** (interval includes 1.0x)")
            print()
            
            print(f"Additional Statistics:")
            print(f"Geometric Mean: {ci_result['geomean']:.2f}x")
            print(f"Median: {ci_result['median']:.2f}x")
            print(f"Contracts with speedup: {better_count}/{ci_result['n_total']} ({better_count/ci_result['n_total']*100:.1f}%)")
            print()
    
    # Print details by vulnerability type
    print("## Speedup by Vulnerability Type")
    print("------------------------------------------------")
    
    for vuln_type in sorted(speedups.keys()):
        if vuln_type in ["__global_metrics__", "__global_speedups__"]:
            continue
            
        contracts = speedups[vuln_type]
        if not contracts:
            continue
            
        # Get all speedups for this type (including infinite/zero)
        all_type_speedups = list(contracts.values())
        finite_speedups = [s for s in all_type_speedups if 0 < s < float('inf')]
        
        print(f"\n### {vuln_type}")
        
        if finite_speedups:
            avg_speedup = np.mean(finite_speedups)
            median_speedup = np.median(finite_speedups)
            geo_mean_speedup = scipy_stats.gmean(finite_speedups)
            max_speedup = max(finite_speedups)
            min_speedup = min(finite_speedups)
            
            print(f"Average Speedup: {avg_speedup:.2f}x")
            print(f"Median Speedup: {median_speedup:.2f}x")
            print(f"Geometric Mean Speedup: {geo_mean_speedup:.2f}x")
            print(f"Range: {min_speedup:.2f}x - {max_speedup:.2f}x")
        
        # Count different outcomes
        method_better = sum(1 for s in all_type_speedups if s > 1.0)
        baseline_better = sum(1 for s in all_type_speedups if 0 < s < 1.0)
        method_only = sum(1 for s in all_type_speedups if s == float('inf'))
        baseline_only = sum(1 for s in all_type_speedups if s == 0.0)
        
        print(f"New method faster: {method_better} contracts")
        print(f"Baseline faster: {baseline_better} contracts")
        print(f"Only new method succeeded: {method_only} contracts")
        print(f"Only baseline succeeded: {baseline_only} contracts")
        
        # Print individual contract details
        print("\nContract Details (sorted by speedup):")
        # Sort with special handling for inf and 0
        sorted_contracts = sorted(contracts.items(), 
                                key=lambda x: (x[1] if 0 < x[1] < float('inf') else 
                                              (float('inf') if x[1] == float('inf') else -1)), 
                                reverse=True)
        
        for contract, speedup in sorted_contracts:
            # Find detailed info for this contract
            detail = next((r for r in detailed_results 
                          if r['contract'] == contract and r['vuln_type'] == vuln_type), None)
            
            if detail:
                if speedup == float('inf'):
                    if detail['method_avg'] is not None:
                        print(f"  {contract}: ∞x (never found → {detail['method_avg']:.1f} sec)")
                    else:
                        print(f"  {contract}: ∞x (never found → found)")
                elif speedup == 0.0:
                    if detail['baseline_avg'] is not None:
                        print(f"  {contract}: 0.00x ({detail['baseline_avg']:.1f} sec → never found)")
                    else:
                        print(f"  {contract}: 0.00x (found → never found)")
                else:
                    baseline_time = detail['baseline_avg'] or 0
                    method_time = detail['method_avg'] or 0
                    print(f"  {contract}: {speedup:.2f}x ({baseline_time:.1f} sec → {method_time:.1f} sec)")

def print_global_metrics(baseline_results: Dict, method_results: Dict):
    """Print global performance metrics comparison."""
    global_metrics_baseline = baseline_results.get("__global_metrics__", {})
    global_metrics_method = method_results.get("__global_metrics__", {})
    
    if not global_metrics_baseline and not global_metrics_method:
        return
    
    print("\n## Global Performance Metrics")
    print("------------------------------------------------")
    
    # Time-based metrics
    time_points = sorted(set([k for k in global_metrics_baseline.keys() if k.endswith('m')] + 
                            [k for k in global_metrics_method.keys() if k.endswith('m')]),
                        key=lambda x: int(x.replace('m', '')))
    
    if time_points:
        print("\n### Detection Rate Over Time")
        print(f"{'Time':<10} {'Baseline':<15} {'Method':<15} {'Improvement':<15} {'Speedup':<10}")
        print("-" * 65)
        
        for time_point in time_points:
            value_baseline = global_metrics_baseline.get(time_point, 0)
            value_method = global_metrics_method.get(time_point, 0)
            
            improvement = "-"
            speedup_value = "-"
            
            if isinstance(value_baseline, (int, float)) and isinstance(value_method, (int, float)):
                if value_baseline > 0:
                    ratio = value_method / value_baseline
                    improvement = f"{(ratio-1)*100:+.2f}%"
                    speedup_value = f"{ratio:.2f}x"
                elif value_method > 0:
                    improvement = "+∞%"
                    speedup_value = "∞x"
            
            print(f"{time_point:<10} {value_baseline:<15} {value_method:<15} {improvement:<15} {speedup_value:<10}")
    
    # Time to reach baseline maximum
    time_to_max = calculate_time_to_baseline_max(baseline_results, method_results)
    if time_to_max:
        print("\n### Time to Reach Baseline Maximum Detection Rate")
        print(f"Baseline maximum detection rate: {time_to_max['baseline_max_rate']:.2f} (at {time_to_max['baseline_time']}m)")
        print(f"Method reaches this rate at: {time_to_max['method_time']}m")
        print(f"Speedup to reach baseline maximum: {time_to_max['speedup']:.2f}x")

def process_files(baseline_file: str, method_file: str, confidence_level: float = 0.95):
    """Process two output files and calculate comprehensive speedup analysis."""
    print(f"Processing Baseline file: {baseline_file}")
    baseline_results = parse_vulnerability_file(baseline_file)
    
    print(f"Processing New Method file: {method_file}")
    method_results = parse_vulnerability_file(method_file)
    
    # Validate data overlap
    validation = validate_data(baseline_results, method_results)
    
    # Calculate speedups with robust handling
    speedup_result = calculate_speedups_robust(baseline_results, method_results)
    
    # Print comprehensive summary
    print_comprehensive_summary(speedup_result, confidence_level)
    
    # Print global metrics
    print_global_metrics(baseline_results, method_results)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate vulnerability detection speedups with robust statistical analysis')
    parser.add_argument('--baseline', required=True, help='File containing baseline method output')
    parser.add_argument('--method', required=True, help='File containing new method output')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level for intervals (default: 0.95)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline file '{args.baseline}' not found")
        return
    
    if not os.path.exists(args.method):
        print(f"Error: Method file '{args.method}' not found")
        return
    
    process_files(args.baseline, args.method, args.confidence)

if __name__ == "__main__":
    main()
