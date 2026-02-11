#!/usr/bin/env python3
"""
Scheduler State Collector - Fixed function name
"""

from bcc import BPF
import time
import json
import argparse
import os
from collections import defaultdict

class SchedulerCollector:
    def __init__(self, output_file="data/raw/scheduler_data.jsonl"):
        self.output_file = output_file
        self.data_buffer = []
        self.buffer_size = 50
        self.stats = defaultdict(int)
        
        self.num_cpus = os.cpu_count()
        print(f"[*] Detected {self.num_cpus} CPUs")
        
        bpf_text = self.generate_bpf_code()
        
        print("[*] Initializing Scheduler Collector...")
        self.bpf = BPF(text=bpf_text)
        
        # print("[*] Attaching to sched:sched_migrate_task tracepoint...")
        # # FIXED: Use the auto-generated function name
        # self.bpf.attach_tracepoint(tp="sched:sched_migrate_task", 
        #                            fn_name="tracepoint__sched__sched_migrate_task")
        
        print("[*] Scheduler Collector ready!")
    
    def generate_bpf_code(self):
        return f"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct migration_event {{
    u64 timestamp_ns;
    u32 pid;
    u32 src_cpu;
    u32 dst_cpu;
    
    // NUMA info
    u8 src_numa_node;
    u8 dst_numa_node;
    u8 cross_node;
    
    // Always 1 for this tracepoint (successful migration)
    u8 migrated;
    
    char comm[16];
}};

BPF_PERF_OUTPUT(migration_events);

// Helper to get NUMA node from CPU - using bitshift instead of division
static inline u8 cpu_to_node(u32 cpu) {{
    // For 16 CPUs with 8 CPUs per node:
    // Node 0 = CPUs 0-7, Node 1 = CPUs 8-15
    // Using bitshift: cpu >> 3 is same as cpu / 8 (unsigned)
    return (u8)(cpu >> 3);
}}

// Trace actual task migrations
TRACEPOINT_PROBE(sched, sched_migrate_task) {{
    struct migration_event data = {{}};
    
    data.timestamp_ns = bpf_ktime_get_ns();
    data.pid = args->pid;
    data.src_cpu = args->orig_cpu;
    data.dst_cpu = args->dest_cpu;
    data.migrated = 1;
    
    // Get process name from current task
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    bpf_probe_read_kernel_str(&data.comm, sizeof(data.comm), task->comm);
    
    // Calculate NUMA topology
    data.src_numa_node = cpu_to_node(data.src_cpu);
    data.dst_numa_node = cpu_to_node(data.dst_cpu);
    data.cross_node = (data.src_numa_node != data.dst_numa_node) ? 1 : 0;
    
    migration_events.perf_submit(args, &data, sizeof(data));
    
    return 0;
}}
"""
    
    def process_event(self, cpu, data, size):
        event = self.bpf["migration_events"].event(data)
        
        record = {
            'timestamp': event.timestamp_ns,
            'pid': event.pid,
            'comm': event.comm.decode('utf-8', 'ignore'),
            'src_cpu': event.src_cpu,
            'dst_cpu': event.dst_cpu,
            'src_numa_node': event.src_numa_node,
            'dst_numa_node': event.dst_numa_node,
            'cross_node': event.cross_node,
            'migrated': event.migrated
        }
        
        self.data_buffer.append(record)
        self.stats['total'] += 1
        self.stats['migrated'] += 1
        
        if event.cross_node:
            self.stats['cross_node'] += 1
        else:
            self.stats['same_node'] += 1
        
        if len(self.data_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        if not self.data_buffer:
            return
        
        with open(self.output_file, 'a') as f:
            for record in self.data_buffer:
                f.write(json.dumps(record) + '\n')
        
        print(f"[*] Total: {self.stats['total']} | "
              f"Same-node: {self.stats['same_node']} | "
              f"Cross-node: {self.stats['cross_node']}", end='\r')
        
        self.data_buffer.clear()
    
    def collect(self, duration=None):
        print(f"[*] Collecting migration events... (Ctrl+C to stop)")
        print(f"[*] Note: Only capturing SUCCESSFUL migrations")
        
        self.bpf["migration_events"].open_perf_buffer(self.process_event)
        
        start_time = time.time()
        
        try:
            while True:
                self.bpf.perf_buffer_poll(timeout=1000)
                
                if duration and (time.time() - start_time) >= duration:
                    break
                    
        except KeyboardInterrupt:
            print("\n[*] Stopping...")
        finally:
            self.flush_buffer()
            print(f"\n[*] Final stats:")
            print(f"    Total migrations: {self.stats['total']}")
            print(f"    Same-node: {self.stats['same_node']}")
            print(f"    Cross-node: {self.stats['cross_node']}")
            print(f"[*] Data saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect scheduler migration events')
    parser.add_argument('-d', '--duration', type=int, help='Duration in seconds')
    parser.add_argument('-o', '--output', default='data/raw/scheduler_data.jsonl')
    args = parser.parse_args()
    
    collector = SchedulerCollector(output_file=args.output)
    collector.collect(duration=args.duration)

if __name__ == '__main__':
    main()