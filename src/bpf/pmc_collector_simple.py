#!/usr/bin/env python3
"""
Simple PMC Collector (without hardware counters for testing)
"""

from bcc import BPF
import time
import json
import argparse
import os
from collections import defaultdict

class SimplePMCCollector:
    def __init__(self, output_file="data/raw/pmc_data.jsonl"):
        self.output_file = output_file
        self.data_buffer = []
        self.buffer_size = 50
        self.stats = defaultdict(int)
        
        self.num_cpus = os.cpu_count()
        print(f"[*] Detected {self.num_cpus} CPUs")
        
        bpf_text = f"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct task_event {{
    u64 timestamp_ns;
    u32 pid;
    u32 cpu;
    u64 runtime_ns;
    char comm[16];
}};

BPF_PERF_OUTPUT(task_events);
BPF_HASH(task_start, u32, u64);

int trace_sched_switch(struct pt_regs *ctx) {{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 cpu = bpf_get_smp_processor_id();
    u64 ts = bpf_ktime_get_ns();
    
    if (pid == 0) return 0;
    
    // Calculate runtime for this task
    u64 *start_ptr = task_start.lookup(&pid);
    
    struct task_event data = {{}};
    data.timestamp_ns = ts;
    data.pid = pid;
    data.cpu = cpu;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    if (start_ptr) {{
        data.runtime_ns = ts - *start_ptr;
    }} else {{
        data.runtime_ns = 0;
    }}
    
    task_start.update(&pid, &ts);
    
    if (data.runtime_ns > 0) {{
        task_events.perf_submit(ctx, &data, sizeof(data));
    }}
    
    return 0;
}}
"""
        
        print("[*] Initializing Simple Collector...")
        self.bpf = BPF(text=bpf_text)
        
        print("[*] Attaching to sched:sched_switch tracepoint...")
        self.bpf.attach_tracepoint(tp="sched:sched_switch", fn_name="trace_sched_switch")
        
        print("[*] Simple Collector ready!")
    
    def process_event(self, cpu, data, size):
        event = self.bpf["task_events"].event(data)
        
        record = {
            'timestamp': event.timestamp_ns,
            'pid': event.pid,
            'cpu': event.cpu,
            'comm': event.comm.decode('utf-8', 'ignore'),
            'runtime_ns': event.runtime_ns,
            'runtime_ms': round(event.runtime_ns / 1_000_000, 3)
        }
        
        self.data_buffer.append(record)
        self.stats['total'] += 1
        
        if len(self.data_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        if not self.data_buffer:
            return
        
        with open(self.output_file, 'a') as f:
            for record in self.data_buffer:
                f.write(json.dumps(record) + '\n')
        
        print(f"[*] Collected {self.stats['total']} samples", end='\r')
        self.data_buffer.clear()
    
    def collect(self, duration=None):
        print(f"[*] Collecting data... (Ctrl+C to stop)")
        
        self.bpf["task_events"].open_perf_buffer(self.process_event)
        
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
            print(f"\n[*] Total samples: {self.stats['total']}")
            print(f"[*] Data saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration', type=int)
    parser.add_argument('-o', '--output', default='data/raw/pmc_data.jsonl')
    args = parser.parse_args()
    
    collector = SimplePMCCollector(output_file=args.output)
    collector.collect(duration=args.duration)

if __name__ == '__main__':
    main()
