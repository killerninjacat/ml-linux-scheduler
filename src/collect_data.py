#!/usr/bin/env python3
"""
Integrated Data Collection
Runs PMC and RAPL collectors simultaneously with workloads
"""

import subprocess
import time
import os
import signal
import argparse
from datetime import datetime

class DataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        self.collectors = []
        os.makedirs(output_dir, exist_ok=True)
    
    def start_collectors(self, session_id):
        print("[*] Starting data collectors...")
        
        # Start PMC collector
        pmc_output = f"{self.output_dir}/pmc_{session_id}.jsonl"
        pmc_proc = subprocess.Popen([
            'sudo', 'python3', 'src/bpf/pmc_collector.py',
            '-o', pmc_output
        ])
        self.collectors.append(('PMC', pmc_proc))
        print(f"    ✓ PMC collector (PID: {pmc_proc.pid})")
        
        # Start RAPL collector
        rapl_output = f"{self.output_dir}/rapl_{session_id}.jsonl"
        rapl_proc = subprocess.Popen([
            'sudo', 'python3', 'src/bpf/rapl_collector.py',
            '-o', rapl_output
        ])
        self.collectors.append(('RAPL', rapl_proc))
        print(f"    ✓ RAPL collector (PID: {rapl_proc.pid})")
        
        time.sleep(2)  # Let collectors initialize
    
    def stop_collectors(self):
        print("\n[*] Stopping collectors...")
        for name, proc in self.collectors:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            print(f"    ✓ Stopped {name} collector")
        self.collectors.clear()
    
    def run_workload(self, workload_type, duration=60):
        print(f"\n[*] Running workload: {workload_type} (duration: {duration}s)")
        
        if workload_type == "cpu_low":
            cmd = ['stress-ng', '--cpu', '4', '--timeout', f'{duration}s']
        elif workload_type == "cpu_medium":
            cmd = ['stress-ng', '--cpu', '8', '--timeout', f'{duration}s']
        elif workload_type == "cpu_high":
            cmd = ['stress-ng', '--cpu', '16', '--timeout', f'{duration}s']
        elif workload_type == "cpu_vm_mixed":
            cmd = ['stress-ng', '--cpu', '8', '--vm', '2', '--vm-bytes', '1G', '--timeout', f'{duration}s']
        elif workload_type == "io_intensive":
            cmd = ['stress-ng', '--io', '4', '--hdd', '2', '--timeout', f'{duration}s']
        else:
            print(f"[!] Unknown workload: {workload_type}")
            return
        
        print(f"    Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"    ✓ Workload completed")
        except subprocess.CalledProcessError as e:
            print(f"    [!] Workload failed: {e}")
    
    def collect_session(self, workloads, duration_per_workload=60):
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("=" * 60)
        print(f"Data Collection Session: {session_id}")
        print("=" * 60)
        
        # Start collectors
        self.start_collectors(session_id)
        
        # Run workloads
        for i, workload in enumerate(workloads, 1):
            print(f"\n[{i}/{len(workloads)}] Workload: {workload}")
            self.run_workload(workload, duration=duration_per_workload)
            
            # Brief pause between workloads
            if i < len(workloads):
                print("[*] Cooling down for 10 seconds...")
                time.sleep(10)
        
        # Stop collectors
        self.stop_collectors()
        
        print("\n" + "=" * 60)
        print(f"Session {session_id} completed!")
        print(f"Data saved in: {self.output_dir}/")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Integrated data collection')
    parser.add_argument('-w', '--workloads', nargs='+', 
                       default=['cpu_low', 'cpu_medium', 'cpu_high', 'cpu_vm_mixed'],
                       help='Workload types to run')
    parser.add_argument('-d', '--duration', type=int, default=60,
                       help='Duration per workload (seconds)')
    parser.add_argument('-o', '--output-dir', default='data/raw',
                       help='Output directory')
    args = parser.parse_args()
    
    collector = DataCollector(output_dir=args.output_dir)
    collector.collect_session(args.workloads, duration_per_workload=args.duration)

if __name__ == '__main__':
    main()
