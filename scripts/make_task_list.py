#!/usr/bin/env python3
import json

# Load the number of chunks per part file
with open("chunks_per_part.json") as f:
    parts = json.load(f)

# Generate the actual list of tasks
tasks = []
for part in sorted(parts.keys(), key=int):
    n_chunks = parts[part]
    for chunk in range(n_chunks):
        tasks.append((int(part), chunk))

# Save to a plain text file
with open("tasks.txt", "w") as f:
    for part, chunk in tasks:
        f.write(f"{part} {chunk}\n")

print(f"{len(tasks)} tasks written to tasks.txt")
