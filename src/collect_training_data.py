#!/usr/bin/env python3
"""
Integrated Training Data Collection
Runs multiple collectors simultaneously with workloads
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
        print(f"Training Data Collection Session: {self.session_id}")
        print("=" * 70)
    
    def start_collectors(self):
        """Start all data collectors"""
        print("\n[*] Starting data collectors...")
        
        # 1. Simple collector (task scheduling)
        simple_output = f"{self.output_dir}/simple_{self.session_id}.jsonl"
        simple_proc = subprocess.Popen([
            'sudo', 'python3', 'src/bpf/pmc_collector_simple.py',
            '-o', simple_output
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.collectors.append(('Simple', simple_proc, simple_output))
        print(f"    ✓ Simple collector started (PID: {simple_proc.pid})")
        
        # 2. Scheduler collector (migration events)
        sched_output = f"{self.output_dir}/scheduler_{self.session_id}.jsonl"
        sched_proc = subprocess.Popen([
            'sudo', 'python3', 'src/bpf/scheduler_collector.py',
            '-o', sched_output
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.collectors.append(('Scheduler', sched_proc, sched_output))
        print(f"    ✓ Scheduler collector started (PID: {sched_proc.pid})")
        
        # 3. RAPL collector (energy) - optional
        try:
            rapl_output = f"{self.output_dir}/rapl_{self.session_id}.jsonl"
            rapl_proc = subprocess.Popen([
                'sudo', 'python3', 'src/bpf/rapl_collector.py',
                '-o', rapl_output
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.collectors.append(('RAPL', rapl_proc, rapl_output))
            print(f"    ✓ RAPL collector started (PID: {rapl_proc.pid})")
        except:
            print(f"    ⚠️  RAPL collector not available (skipping)")
        
        # Let collectors initialize
        print("[*] Waiting for collectors to initialize...")
        time.sleep(3)
        print("[*] All collectors ready!")
    
    def stop_collectors(self):
        """Stop all running collectors"""
        print("\n[*] Stopping collectors...")
        for name, proc, output in self.collectors:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5)
                
                # Check if data was collected
                if os.path.exists(output):
                    lines = sum(1 for _ in open(output))
                    print(f"    ✓ {name}: {lines} samples → {output}")
                else:
                    print(f"    ✗ {name}: No data collected")
            except Exception as e:
                print(f"    ⚠️  {name}: Error stopping - {e}")
        
        self.collectors.clear()
    
    def run_workload(self, workload_type, duration=60):
        """Run a specific workload"""
        workloads = {
            'cpu_light': ['stress-ng', '--cpu', '4', '--timeout', f'{duration}s'],
            'cpu_medium': ['stress-ng', '--cpu', '8', '--timeout', f'{duration}s'],
            'cpu_heavy': ['stress-ng', '--cpu', '12', '--timeout', f'{duration}s'],
            'cpu_vm_mixed': ['stress-ng', '--cpu', '8', '--vm', '2', '--vm-bytes', '1G', '--timeout', f'{duration}s'],
            'io_intensive': ['stress-ng', '--io', '4', '--hdd', '2', '--timeout', f'{duration}s'],
            'mixed': ['stress-ng', '--cpu', '6', '--vm', '2', '--io', '2', '--timeout', f'{duration}s']
        }
        
        if workload_type not in workloads:
            print(f"[!] Unknown workload: {workload_type}")
            return False
        
        cmd = workloads[workload_type]
        print(f"\n[Workload] {workload_type}")
        print(f"           Command: {' '.join(cmd)}")
        print(f"           Duration: {duration}s")
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"           ✓ Completed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"           ✗ Failed: {e}")
            return False
    
    def collect_session(self, workloads, duration_per_workload=60):
        """Run complete data collection session"""
        
        # Start all collectors
        self.start_collectors()
        
        # Run each workload
        total_workloads = len(workloads)
        for i, workload in enumerate(workloads, 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{total_workloads}] Running: {workload}")
            print(f"{'='*70}")
            
            self.run_workload(workload, duration=duration_per_workload)
            
            # Brief pause between workloads
            if i < total_workloads:
                print("\n[*] Cooling down for 10 seconds...")
                time.sleep(10)
        
        # Stop all collectors
        self.stop_collectors()
        
        # Generate summary
        self.generate_summary()
        
        print("\n" + "=" * 70)
        print(f"✓ Session {self.session_id} completed!")
        print(f"  Data location: {self.output_dir}/")
        print("=" * 70)
    
    def generate_summary(self):
        """Generate collection summary"""
        summary_file = f"{self.output_dir}/summary_{self.session_id}.json"
        
        summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'collectors': []
        }
        
        for name, _, output in self.collectors:
            if os.path.exists(output):
                lines = sum(1 for _ in open(output))
                size = os.path.getsize(output)
                summary['collectors'].append({
                    'name': name,
                    'output': output,
                    'samples': lines,
                    'size_bytes': size
                })
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[*] Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect training data')
    parser.add_argument('-w', '--workloads', nargs='+',
                       default=['cpu_light', 'cpu_medium', 'cpu_heavy', 'cpu_vm_mixed'],
                       help='Workload types to run')
    parser.add_argument('-d', '--duration', type=int, default=60,
                       help='Duration per workload (seconds)')
    parser.add_argument('-o', '--output-dir', default='data/raw',
                       help='Output directory')
    parser.add_argument('-s', '--session-id', help='Session ID (default: timestamp)')
    args = parser.parse_args()
    
    collector = TrainingDataCollector(
        output_dir=args.output_dir,
        session_id=args.session_id
    )
    
    collector.collect_session(
        workloads=args.workloads,
        duration_per_workload=args.duration
    )

if __name__ == '__main__':
    main()