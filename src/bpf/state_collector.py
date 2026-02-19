#!/usr/bin/env python3
"""
Hybrid State Collector
Combines kernel-accurate data with userspace migration tracking
"""

from bcc import BPF
import time
import json
import os
import argparse
from datetime import datetime
from collections import defaultdict
import psutil

BPF_PROGRAM = r"""
#include <linux/sched.h>

struct cpu_state_t {
    u64 timestamp;
    u32 cpu;
    u32 nr_running;
    u32 numa_node;
};

BPF_PERF_OUTPUT(events);

// Store latest state per CPU
BPF_HASH(cpu_states, u32, struct cpu_state_t);

int collect_snapshot(void *ctx) {
    u32 cpu = bpf_get_smp_processor_id();
    
    struct cpu_state_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.cpu = cpu;
    
    // Get runqueue info
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct rq *rq = task->se.cfs_rq->rq;
    data.nr_running = rq->nr_running;
    
    // Get NUMA node
    data.numa_node = cpu_to_node(cpu);
    
    // Update hash map
    cpu_states.update(&cpu, &data);
    
    return 0;
}
"""

class HybridStateCollector:
    def __init__(self, output_file="data/raw/hybrid_snapshots.jsonl", interval_ms=100):
        self.output_file = output_file
        self.interval_sec = interval_ms / 1000.0
        
        self.data_buffer = []
        self.buffer_size = 50
        self.stats = defaultdict(int)
        
        # Track task locations for migration detection
        self.task_locations = {}  # {pid: cpu}
        
        self.num_cpus = os.cpu_count()
        print(f"[*] Detected {self.num_cpus} CPUs")
        
        # Initialize BPF
        print("[*] Loading BPF program...")
        self.bpf = BPF(text=BPF_PROGRAM)
        self.bpf.attach_kprobe(event="finish_task_switch", fn_name="collect_snapshot")
        print("    ✓ BPF program loaded")
        
        # Initialize CPU stats
        self.prev_cpu_stats = self.get_cpu_stats()
        
        print("[*] Hybrid State Collector ready!")
    
    def get_cpu_stats(self):
        """Get CPU statistics from /proc/stat"""
        cpu_stats = {}
        
        try:
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('cpu') and len(line) > 3 and line[3].isdigit():
                        parts = line.split()
                        cpu_id = int(parts[0][3:])
                        
                        user = int(parts[1])
                        nice = int(parts[2])
                        system = int(parts[3])
                        idle = int(parts[4])
                        iowait = int(parts[5])
                        irq = int(parts[6])
                        softirq = int(parts[7])
                        
                        total = user + nice + system + idle + iowait + irq + softirq
                        
                        cpu_stats[cpu_id] = {
                            'total': total,
                            'idle': idle
                        }
        except Exception as e:
            print(f"[!] Error reading /proc/stat: {e}")
        
        return cpu_stats
    
    def calculate_cpu_load(self, cpu_id):
        """Calculate CPU load percentage"""
        current_stats = self.get_cpu_stats()
        
        if cpu_id not in current_stats or cpu_id not in self.prev_cpu_stats:
            return 0.0
        
        prev = self.prev_cpu_stats[cpu_id]
        curr = current_stats[cpu_id]
        
        total_delta = curr['total'] - prev['total']
        idle_delta = curr['idle'] - prev['idle']
        
        if total_delta == 0:
            return 0.0
        
        load = 100.0 * (1.0 - idle_delta / total_delta)
        return max(0.0, min(100.0, load))
    
    def get_kernel_cpu_state(self, cpu_id):
        """Get accurate CPU state from kernel (via BPF map)"""
        try:
            cpu_states = self.bpf["cpu_states"]
            state = cpu_states[cpu_id]
            return {
                'timestamp': state.timestamp,
                'nr_running': state.nr_running,
                'numa_node': state.numa_node
            }
        except:
            return {
                'timestamp': 0,
                'nr_running': 0,
                'numa_node': cpu_id // 8
            }
    
    def take_snapshot(self):
        """Take snapshot and create training records"""
        
        # Update CPU stats
        current_stats = self.get_cpu_stats()
        self.prev_cpu_stats = current_stats
        
        current_tasks = {}
        
        # Scan all running tasks
        for proc in psutil.process_iter(['pid', 'status', 'cpu_num', 'name']):
            try:
                info = proc.info
                pid = info['pid']
                cpu_num = info['cpu_num']
                
                if info['status'] in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]:
                    if cpu_num is not None and 0 <= cpu_num < self.num_cpus:
                        current_tasks[pid] = cpu_num
                        
                        # Check for migration
                        if pid in self.task_locations:
                            prev_cpu = self.task_locations[pid]
                            
                            if prev_cpu != cpu_num:
                                # MIGRATION DETECTED!
                                record = self.create_record(
                                    pid, prev_cpu, cpu_num, migrated=True
                                )
                                self.data_buffer.append(record)
                                self.stats['positive'] += 1
                            else:
                                # NO MIGRATION - create negative example (sample 20%)
                                if self.stats['total'] % 5 == 0:
                                    # Pick a random other CPU as "candidate"
                                    candidate_cpu = (cpu_num + 1) % self.num_cpus
                                    record = self.create_record(
                                        pid, cpu_num, candidate_cpu, migrated=False
                                    )
                                    self.data_buffer.append(record)
                                    self.stats['negative'] += 1
                        
                        self.stats['total'] += 1
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Update task locations
        self.task_locations = current_tasks
        
        # Flush buffer
        if len(self.data_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def create_record(self, pid, src_cpu, dst_cpu, migrated):
        """Create training record with both kernel and userspace data"""
        
        # Get ACCURATE kernel state
        src_kernel = self.get_kernel_cpu_state(src_cpu)
        dst_kernel = self.get_kernel_cpu_state(dst_cpu)
        
        # Get userspace calculated loads
        src_load = self.calculate_cpu_load(src_cpu)
        dst_load = self.calculate_cpu_load(dst_cpu)
        
        record = {
            'timestamp': time.time_ns(),
            'pid': pid,
            
            # Source CPU
            'src_cpu': src_cpu,
            'src_load': round(src_load, 2),
            'src_runqueue_len': src_kernel['nr_running'],  # ACCURATE!
            'src_numa_node': src_kernel['numa_node'],
            'src_cpu_idle': 1 if src_load < 5.0 else 0,
            
            # Destination CPU
            'dst_cpu': dst_cpu,
            'dst_load': round(dst_load, 2),
            'dst_runqueue_len': dst_kernel['nr_running'],  # ACCURATE!
            'dst_numa_node': dst_kernel['numa_node'],
            'dst_cpu_idle': 1 if dst_load < 5.0 else 0,
            
            # Derived features
            'cross_node': 1 if src_kernel['numa_node'] != dst_kernel['numa_node'] else 0,
            'load_diff': abs(src_load - dst_load),
            
            # LABEL
            'decision': 1 if migrated else 0
        }
        
        return record
    
    def flush_buffer(self):
        """Write data to file"""
        if not self.data_buffer:
            return
        
        with open(self.output_file, 'a') as f:
            for record in self.data_buffer:
                f.write(json.dumps(record) + '\n')
        
        pos = self.stats['positive']
        neg = self.stats['negative']
        total = pos + neg
        
        if total > 0:
            print(f"[*] Total: {total:>6,} | "
                  f"Positive: {pos:>6,} ({pos/total*100:.1f}%) | "
                  f"Negative: {neg:>6,} ({neg/total*100:.1f}%)", end='\r')
        
        self.data_buffer.clear()
    
    def collect(self, duration=None):
        """Main collection loop"""
        print(f"[*] Starting hybrid collection...")
        print(f"[*] Sampling every {self.interval_sec*1000:.0f}ms")
        print(f"[*] Press Ctrl+C to stop")
        
        start_time = time.time()
        
        try:
            while True:
                self.take_snapshot()
                
                if duration and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(self.interval_sec)
                
        except KeyboardInterrupt:
            print("\n[*] Stopping...")
        finally:
            self.flush_buffer()
            
            pos = self.stats['positive']
            neg = self.stats['negative']
            total = pos + neg
            
            print(f"\n[*] Final statistics:")
            print(f"    Total samples: {total:,}")
            if total > 0:
                print(f"    Positive (migrated): {pos:,} ({pos/total*100:.1f}%)")
                print(f"    Negative (stayed): {neg:,} ({neg/total*100:.1f}%)")
            print(f"[*] Data saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description='Hybrid state collector')
    parser.add_argument('-d', '--duration', type=int, help='Duration in seconds')
    parser.add_argument('-i', '--interval', type=int, default=100, 
                       help='Sampling interval in ms (default: 100)')
    parser.add_argument('-o', '--output', default='data/raw/hybrid_snapshots.jsonl')
    args = parser.parse_args()
    
    collector = HybridStateCollector(
        output_file=args.output,
        interval_ms=args.interval
    )
    collector.collect(duration=args.duration)

if __name__ == '__main__':
    main()