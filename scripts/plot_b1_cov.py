import sys, os
from common import TOTAL_TIME, PLOT_INTERVAL, COV_FILE_NAME, B1_INST_INFO_FILE

ALLOWED_WARNING_1 = "Unexpected deployed address"
ALLOWED_WARNING_2 = "Deployed address is same to sender"

### Functions for parsing coverage files.

def init_b1_inst_info():
    cve_info = { }
    cve_csv_file = open(B1_INST_INFO_FILE, "r")
    for buf in cve_csv_file:
        tokens = buf.strip().split(",")
        if len(tokens) != 3:
            print("Invalid entry in CSV file: %s" % buf)
            exit(1)
        file = tokens[0]
        total_inst = int(tokens[1])
        total_edges = int(tokens[2])
        cve_info[file] = (total_inst, total_edges)
    cve_csv_file.close()
    return cve_info


def parse_cov_line(line):
    tokens = line.strip().split()
    edges = int(tokens[1])
    instrs = int(tokens[3])
    sec = int(tokens[0].split('m:')[0]) * 60
    return sec, edges, instrs

def record_cov_over_time(time_map, targ, cov_lines):
    for line in cov_lines:
        if ALLOWED_WARNING_1 in line or ALLOWED_WARNING_2 in line:
            continue
        if "No test case generated" in line:
            print("[Warning] No test case generated for %s" % targ)
            continue
        if "Edges" not in line:
            print("Unexpected line format: %s" % line)
            exit(1)
        sec, edges, instrs = parse_cov_line(line)
        time_map[(targ, sec)] = instrs

def analyze_dir(inst_info, result_dir, targ_list):
    cov_time_map = { }
    total_inst = 0
    for targ in targ_list:
        targ_dir = os.path.join(result_dir, targ)
        cov_path = os.path.join(targ_dir, COV_FILE_NAME)
        # When we don't have cov_path, just ignore it.
        if not os.path.exists(cov_path):
            print('No cov.txt: %s' % cov_path)
            continue
        cov_f = open(cov_path, "r")
        cov_lines = cov_f.readlines()
        record_cov_over_time(cov_time_map, targ, cov_lines)
        total_inst += inst_info[targ][0]
        cov_f.close()
    return cov_time_map, total_inst

### Functions to print analyzed results.

def get_cov_before(time_map, sec):
    targ_cov_map = { }
    for key in time_map:
        targ, time = key
        if time <= sec:
            if targ not in targ_cov_map:
                targ_cov_map[targ] = 0
            # Collect coverage per target by taking maximum value
            # (coverage is accumulated)
            targ_cov_map[targ] = max(targ_cov_map[targ], time_map[key])
    return sum(targ_cov_map.values())

def plot_cov_over_time(time_map_list, total_inst):
    for minute in range(0, TOTAL_TIME + PLOT_INTERVAL, PLOT_INTERVAL):
        sec = 60 * minute
        cov_list = []
        for time_map in time_map_list:
            cov_list.append(get_cov_before(time_map, sec))
        cov_avg = float(sum(cov_list)) / float(len(cov_list))
        print("%02dm: %.2f" % (minute, (cov_avg / total_inst) * 100 ))

def main():
    if len(sys.argv) < 2:
        print("Usage: %s [result dirs]" % sys.argv[0])
        exit(1)

    result_dirs = sys.argv[1:]

    targ_list = os.listdir(result_dirs[0])
    targ_list.sort()

    cov_time_map_list = []
    inst_info = init_b1_inst_info()
    for result_dir in result_dirs:
        covs, total_inst = analyze_dir(inst_info, result_dir, targ_list)
        cov_time_map_list.append(covs)

    plot_cov_over_time(cov_time_map_list, total_inst)

if __name__ == "__main__":
    main()
