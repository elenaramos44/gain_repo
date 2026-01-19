import json

# Load JSON
json_file = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files/run2306_part_pulse_widths.json"
with open(json_file, "r") as f:
    data = json.load(f)

# Parameters
chunk_size = 250  # same as SLURM script
tasks_file = "/scratch/elena/WCTE_recovery/scripts/tasks_run2306_790ps.txt"

tasks = []

for part_str, info in data.items():
    if info["pulse_width"] == "790ps":
        part_number = int(part_str[1:])  # remove "P" prefix
        # split into chunks of chunk_size events
        # assume 1 chunk per part if unknown, or you can adjust
        tasks.append(f"{part_number} 0")  # chunk_id 0; extend if needed

# Write tasks file
with open(tasks_file, "w") as f:
    f.write("\n".join(tasks))

print(f"[INFO] {len(tasks)} tasks written to {tasks_file}")
