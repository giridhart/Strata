#!/usr/bin/env python3
import sys
import re
from collections import defaultdict

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else "build_sycl/build_all_clean.log"
    warning_pattern = re.compile(r"^(/[^:\s]+):(\d+):(?:\d+:)?\s+warning:\s+(.+)$")
    
    warnings_by_file = defaultdict(list)
    total_warnings = 0
    
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = warning_pattern.match(line.strip())
            if match:
                file_path, line_no, msg = match.groups()
                # filter duplicates from multi-target compilation
                entry = (line_no, msg)
                if entry not in warnings_by_file[file_path]:
                    warnings_by_file[file_path].append(entry)
                    total_warnings += 1

    print(f"=== WARNING REPORT for {log_file} ===")
    print(f"Total Unique Warnings: {total_warnings}")
    print(f"Files Affected: {len(warnings_by_file)}\n")
    
    for file_path, entries in sorted(warnings_by_file.items()):
        print(f"File: {file_path} ({len(entries)} warning(s))")
        for line_no, msg in entries:
            print(f"  L{line_no}: {msg}")
        print()

if __name__ == "__main__":
    main()
