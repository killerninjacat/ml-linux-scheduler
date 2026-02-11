#!/usr/bin/env python3
"""
PMC Collector - Uses tracepoints for better compatibility
"""

from bcc import BPF, PerfType, PerfHWConfig
import time
import json
import argparse
import os
from collections import defaultdict

class PMCCollector:
    def __init__(self, output_file="data/raw/pmc_data.jsonl"):
        self.output_file = output_file
        self.data_buffer = []
        self.buffer_size = 100
        self.stats = defaultdict(int)
        
        self.num_cpus = os.cpu_count()
        print(f"[*] Detected {self.num_cpus} CPUs")
        
        bpf_text = self.generate_bpf_code()
        
        print("[*] Initializing PMC Collector...")
        self.bpf = BPF(text=bpf_text)
        
        # Use tracepoint instead of kprobe
        print("[*] Attaching to sched:sched_switch tracepoint...")
        self.bpf.attach_tracepoint(tp="sched:sched_switch", fn_name="trace_sched_switch")
        
        print("[*] Setting up performance counters...")
        self.setup_perf_counters()
        
        print("[*] PMC Collector ready!")
    
    def generate_bpf_code(self):
        return f"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct pmc_event {{
    u64 timestamp_ns;
    u32 pid;
    u32 cpu;
    u64 llc_misses;
    u64 instructions;
    u64 cycles;
    char comm[16];
}};

BPF_PERF_OUTPUT(pmc_events);
BPF_PERF_ARRAY(llc_misses, {self.num_cpus});
BPF_PERF_ARRAY(instructions, {self.num_cpus});
BPF_PERF_ARRAY(cpu_cycles, {self.num_cpus});
BPF_HASH(prev_llc, u32, u64);
BPF_HASH(prev_inst, u32, u64);
BPF_HASH(prev_cycles, u32, u64);

// Use tracepoint instead of kprobe
TRACEPOINT_PROBE(sched, sched_switch) {{
    u32 pid = args->next_pid;
    u32 cpu = bpf_get_smp_processor_id();
    
    if (pid == 0) return 0;
    
    struct pmc_event data = {{}};
    data.timestamp_ns = bpf_ktime_get_ns();
    data.pid = pid;
    data.cpu = cpu;
    bpf_probe_read_kernel_str(&data.comm, sizeof(data.comm), args->next_comm);
    
    // Read hardware counters
    u64 llc = llc_misses.perf_read(CUR_CPU_IDENTIFIER);
    u64 inst = instructions.perf_read(CUR_CPU_IDENTIFIER);
    u64 cyc = cpu_cycles.perf_read(CUR_CPU_IDENTIFIER);
    
    // Check for errors
    if ((s64)llc < 0 || (s64)inst < 0 || (s64)cyc < 0) {{
        return 0;
    }}
    
    // Calculate deltas
    u64 *prev_llc_ptr = prev_llc.lookup(&pid);
    u64 *prev_inst_ptr = prev_inst.lookup(&pid);
    u64 *prev_cycles_ptr = prev_cycles.lookup(&pid);
    
    if (prev_llc_ptr && prev_inst_ptr && prev_cycles_ptr) {{
        data.llc_misses = llc > *prev_llc_ptr ? llc - *prev_llc_ptr : 0;
        data.instructions = inst > *prev_inst_ptr ? inst - *prev_inst_ptr : 0;
        data.cycles = cyc > *prev_cycles_ptr ? cyc - *prev_cycles_ptr : 0;
    }} else {{
        data.llc_misses = 0;
        data.instructions = 0;
        data.cycles = 0;
    }}
    
    // Update tracking
    prev_llc.update(&pid, &llc);
    prev_inst.update(&pid, &inst);
    prev_cycles.update(&pid, &cyc);
    
    // Submit if there was activity
    if (data.instructions > 100) {{  // Threshold to reduce noise
        pmc_events.perf_submit(args, &data, sizeof(data));
    }}
    
    return 0;
}}
"""
    
    def setup_perf_counters(self):
        try:
            self.bpf["llc_misses"].open_perf_event(
                PerfType.HARDWARE, PerfHWConfig.CACHE_MISSES
            )
            print("    ✓ LLC misses counter")
            
            self.bpf["instructions"].open_perf_event(
                PerfType.HARDWARE, PerfHWConfig.INSTRUCTIONS
            )
            print("    ✓ Instructions counter")
            
            self.bpf["cpu_cycles"].open_perf_event(
                PerfType.HARDWARE, PerfHWConfig.CPU_CYCLES
            )
            print("    ✓ CPU cycles counter")
            
        except Exception as e:
            print(f"[!] Warning: Could not setup hardware counters: {e}")
            print("[!] Hardware PMC data will not be available")
            print("[!] Continuing anyway...")
    
    def process_event(self, cpu, data, size):
        event = self.bpf["pmc_events"].event(data)
        
        ipc = event.instructions / event.cycles if event.cycles > 0 else 0.0
        llc_miss_rate = event.llc_misses / event.instructions if event.instructions > 0 else 0.0
        
        record = {
            'timestamp': event.timestamp_ns,
            'pid': event.pid,
            'cpu': event.cpu,
            'comm': event.comm.decode('utf-8', 'ignore'),
            'llc_misses': event.llc_misses,
            'instructions': event.instructions,
            'cycles': event.cycles,
            'ipc': round(ipc, 4),
            'llc_miss_rate': round(llc_miss_rate, 6)
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
        
        print(f"[*] Collected {self.stats['total']} PMC samples", end='\r')
        self.data_buffer.clear()
    
    def collect(self, duration=None):
        print(f"[*] Collecting PMC data... (Ctrl+C to stop)")
        
        self.bpf["pmc_events"].open_perf_buffer(self.process_event)
        
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
    parser = argparse.ArgumentParser(description='Collect PMC data with hardware counters')
    parser.add_argument('-d', '--duration', type=int, help='Duration in seconds')
    parser.add_argument('-o', '--output', default='data/raw/pmc_data.jsonl')
    args = parser.parse_args()
    
    collector = PMCCollector(output_file=args.output)
    collector.collect(duration=args.duration)

if __name__ == '__main__':
    main()
