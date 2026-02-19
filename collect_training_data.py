#!/usr/bin/env python3
"""
Integrated Training Data Collection (FIXED)
Runs ALL collectors simultaneously
"""

import subprocess
import time
import os
import signal
import argparse
from datetime import datetime
import json


class TrainingDataCollector:

    def __init__(self, output_dir="data/raw", session_id=None):
        self.output_dir = output_dir
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collectors = []

        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print(f"Training Data Session: {self.session_id}")
        print("=" * 70)

    # ---------------------------------------------------
    # START COLLECTORS
    # ---------------------------------------------------

    def start_collectors(self, duration):

        print("\n[*] Starting collectors...")

        # 1️⃣ STATE SNAPSHOT COLLECTOR (MAIN DATASET)
        state_output = f"{self.output_dir}/state_{self.session_id}.jsonl"
        state_proc = subprocess.Popen([
            "sudo", "python3",
            "src/bpf/state_collector_simple.py",
            "-d", str(duration),
            "-o", state_output
        ])
        self.collectors.append(("StateSnapshot", state_proc, state_output))
        print(f"    ✓ State snapshot collector started")

        # 2️⃣ SCHEDULER COLLECTOR
        sched_output = f"{self.output_dir}/scheduler_{self.session_id}.jsonl"
        sched_proc = subprocess.Popen([
            "sudo", "python3",
            "src/bpf/scheduler_collector.py",
            "-d", str(duration),
            "-o", sched_output
        ])
        self.collectors.append(("Scheduler", sched_proc, sched_output))
        print(f"    ✓ Scheduler collector started")

        # 3️⃣ PMC / RUNTIME COLLECTOR
        pmc_output = f"{self.output_dir}/pmc_{self.session_id}.jsonl"
        pmc_proc = subprocess.Popen([
            "sudo", "python3",
            "src/bpf/pmc_collector_simple.py",
            "-d", str(duration),
            "-o", pmc_output
        ])
        self.collectors.append(("PMC", pmc_proc, pmc_output))
        print(f"    ✓ PMC collector started")

        # 4️⃣ RAPL ENERGY COLLECTOR
        try:
            rapl_output = f"{self.output_dir}/rapl_{self.session_id}.jsonl"
            rapl_proc = subprocess.Popen([
                "sudo", "python3",
                "src/bpf/rapl_collector.py",
                "-d", str(duration),
                "-o", rapl_output
            ])
            self.collectors.append(("RAPL", rapl_proc, rapl_output))
            print(f"    ✓ RAPL collector started")
        except Exception:
            print("    ⚠ RAPL unavailable")

        time.sleep(3)
        print("[*] All collectors initialized.")

    # ---------------------------------------------------

    def stop_collectors(self):

        print("\n[*] Stopping collectors...")

        for name, proc, output in self.collectors:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5)

                if os.path.exists(output):
                    lines = sum(1 for _ in open(output))
                    print(f"    ✓ {name}: {lines} samples")
                else:
                    print(f"    ✗ {name}: no data")

            except Exception as e:
                print(f"    ⚠ {name} error: {e}")

    # ---------------------------------------------------

    def run_workload(self, workload, duration):

        workloads = {
            "cpu_light": ["stress-ng", "--cpu", "4", "--timeout", f"{duration}s"],
            "cpu_medium": ["stress-ng", "--cpu", "8", "--timeout", f"{duration}s"],
            "cpu_heavy": ["stress-ng", "--cpu", "12", "--timeout", f"{duration}s"],
            "cpu_vm_mixed": [
                "stress-ng", "--cpu", "8",
                "--vm", "2", "--vm-bytes", "1G",
                "--timeout", f"{duration}s"
            ]
        }

        cmd = workloads[workload]

        print(f"\n[Workload] {workload}")
        subprocess.run(cmd)

    # ---------------------------------------------------

    def collect_session(self, workloads, duration):

        self.start_collectors(duration)

        for workload in workloads:
            self.run_workload(workload, duration)

        self.stop_collectors()

        import subprocess, os

        print("[*] Fixing file ownership...")
        subprocess.run([
            "sudo",
            "chown",
            "-R",
            f"{os.getenv('SUDO_USER')}:{os.getenv('SUDO_USER')}",
            self.output_dir
        ])

        print("\n✓ Data collection complete.")

# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--workloads",
        nargs="+",
        default=["cpu_light", "cpu_medium", "cpu_heavy", "cpu_vm_mixed"]
    )
    parser.add_argument("-d", "--duration", type=int, default=60)
    parser.add_argument("-o", "--output-dir", default="data/raw")

    args = parser.parse_args()

    collector = TrainingDataCollector(args.output_dir)
    collector.collect_session(args.workloads, args.duration)


if __name__ == "__main__":
    main()
