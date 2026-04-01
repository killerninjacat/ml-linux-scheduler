#!/usr/bin/env python3
"""
PMC Collector with hardware counters (instructions, cycles, IPC)
"""

from bcc import BPF, PerfType, PerfHWConfig
from bcc.libbcc import lib
import time
import json
import argparse
import os
import struct
from collections import defaultdict

class SimplePMCCollector:
    def __init__(self, output_file="data/raw/pmc_data.jsonl"):
        self.output_file = output_file
        self.data_buffer = []
        self.buffer_size = 50
        self.stats = defaultdict(int)
        
        self.num_cpus = os.cpu_count()
        print(f"[*] Detected {self.num_cpus} CPUs")

        self.perf_fds = {
            'instructions': {},
            'cycles': {}
        }
        self.prev_counters = {
            'instructions': {},
            'cycles': {}
        }
        self.ipc_supported = False
        
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

        self.setup_perf_counters()
        
        print("[*] Attaching to sched:sched_switch tracepoint...")
        self.bpf.attach_tracepoint(tp="sched:sched_switch", fn_name="trace_sched_switch")
        
        print("[*] Simple Collector ready!")

    def setup_perf_counters(self):
        """
        Open per-CPU perf event FDs for instructions and cycles.
        Values are read in userspace to derive per-event deltas and IPC.
        """
        print("[*] Enabling hardware counters (instructions/cycles)...")

        for cpu in range(self.num_cpus):
            instr_fd = lib.bpf_open_perf_event(
                PerfType.HARDWARE,
                PerfHWConfig.INSTRUCTIONS,
                -1,
                cpu
            )
            cycles_fd = lib.bpf_open_perf_event(
                PerfType.HARDWARE,
                PerfHWConfig.CPU_CYCLES,
                -1,
                cpu
            )

            if instr_fd < 0 or cycles_fd < 0:
                if instr_fd >= 0:
                    os.close(instr_fd)
                if cycles_fd >= 0:
                    os.close(cycles_fd)
                self.close_perf_counters()
                print("[!] Hardware counters unavailable; IPC fields will default to 0")
                return

            self.perf_fds['instructions'][cpu] = instr_fd
            self.perf_fds['cycles'][cpu] = cycles_fd

        self.ipc_supported = True
        print("[✓] Hardware counters enabled")

    def close_perf_counters(self):
        for counter_type in self.perf_fds.values():
            for fd in counter_type.values():
                try:
                    os.close(fd)
                except OSError:
                    pass

        self.perf_fds = {'instructions': {}, 'cycles': {}}
        self.prev_counters = {'instructions': {}, 'cycles': {}}
        self.ipc_supported = False

    def read_counter(self, fd):
        """Read current perf counter value from FD."""
        if fd is None:
            return None

        try:
            raw = os.read(fd, 8)
            if len(raw) != 8:
                return None
            return struct.unpack("Q", raw)[0]
        except OSError:
            return None
    
    def process_event(self, cpu, data, size):
        event = self.bpf["task_events"].event(data)

        instructions_delta = 0
        cycles_delta = 0
        ipc_value = 0.0

        if self.ipc_supported:
            cpu_id = event.cpu
            instr_now = self.read_counter(self.perf_fds['instructions'].get(cpu_id))
            cycles_now = self.read_counter(self.perf_fds['cycles'].get(cpu_id))

            if instr_now is not None and cycles_now is not None:
                prev_instr = self.prev_counters['instructions'].get(cpu_id)
                prev_cycles = self.prev_counters['cycles'].get(cpu_id)

                if prev_instr is not None and instr_now >= prev_instr:
                    instructions_delta = instr_now - prev_instr

                if prev_cycles is not None and cycles_now >= prev_cycles:
                    cycles_delta = cycles_now - prev_cycles

                self.prev_counters['instructions'][cpu_id] = instr_now
                self.prev_counters['cycles'][cpu_id] = cycles_now

                if cycles_delta > 0:
                    ipc_value = instructions_delta / cycles_delta
                    self.stats['ipc_samples'] += 1
                else:
                    self.stats['zero_cycle_windows'] += 1
            else:
                self.stats['counter_read_errors'] += 1
        
        record = {
            'timestamp': event.timestamp_ns,
            'pid': event.pid,
            'cpu': event.cpu,
            'comm': event.comm.decode('utf-8', 'ignore'),
            'runtime_ns': event.runtime_ns,
            'runtime_ms': round(event.runtime_ns / 1_000_000, 3),
            'instructions_retired': int(instructions_delta),
            'cycles': int(cycles_delta),
            'ipc': round(ipc_value, 4),
            'ipc_available': 1 if self.ipc_supported else 0
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
            ipc_mode_enabled = self.ipc_supported
            self.flush_buffer()
            print(f"\n[*] Total samples: {self.stats['total']}")
            if ipc_mode_enabled:
                print(f"[*] IPC samples with valid cycles: {self.stats['ipc_samples']}")
            else:
                print("[*] IPC collection mode: fallback (hardware counters unavailable)")
            self.close_perf_counters()
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
