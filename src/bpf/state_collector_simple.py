#!/usr/bin/env python3
"""
Simplified State Collector (FIXED VERSION)
Pure userspace approach - balanced dataset with positive/negative examples
"""

import time
import json
import os
import argparse
from collections import defaultdict
import psutil


class SimpleStateCollector:
    def __init__(self, output_file="data/raw/state_snapshots.jsonl", interval_ms=100):
        self.output_file = output_file
        self.interval_sec = interval_ms / 1000.0

        self.data_buffer = []
        self.buffer_size = 50
        self.stats = defaultdict(int)

        self.task_locations = {}
        self.task_info = {}

        self.num_cpus = os.cpu_count()
        print(f"[*] Detected {self.num_cpus} CPUs")
        print(f"[*] Sampling interval: {interval_ms}ms")

        self.prev_cpu_times = psutil.cpu_times(percpu=True)

        print("[*] Simple State Collector ready!")

    # ---------------- CPU LOAD ----------------
    def get_cpu_load(self, cpu_id, current_times):
        if cpu_id >= len(current_times) or cpu_id >= len(self.prev_cpu_times):
            return 0.0

        prev = self.prev_cpu_times[cpu_id]
        curr = current_times[cpu_id]

        prev_total = sum([
            prev.user, prev.nice, prev.system,
            prev.idle, prev.iowait, prev.irq, prev.softirq
        ])

        curr_total = sum([
            curr.user, curr.nice, curr.system,
            curr.idle, curr.iowait, curr.irq, curr.softirq
        ])

        total_delta = curr_total - prev_total
        idle_delta = curr.idle - prev.idle

        if total_delta <= 0:
            return 0.0

        load = 100.0 * (1.0 - idle_delta / total_delta)
        return max(0.0, min(100.0, load))

    # ---------------- NUMA ----------------
    def cpu_to_numa(self, cpu_id):
        try:
            with open(f'/sys/devices/system/cpu/cpu{cpu_id}/node_id', 'r') as f:
                return int(f.read().strip())
        except:
            return cpu_id // 8

    # ---------------- RECORD CREATION ----------------
    def create_record(self, pid, src_cpu, dst_cpu, migrated,
                      current_cpu_times, cpu_runnable_counts):

        try:
            src_load = self.get_cpu_load(src_cpu, current_cpu_times)
            dst_load = self.get_cpu_load(dst_cpu, current_cpu_times)

            src_runqueue = cpu_runnable_counts.get(src_cpu, 0)
            dst_runqueue = cpu_runnable_counts.get(dst_cpu, 0)

            src_numa = self.cpu_to_numa(src_cpu)
            dst_numa = self.cpu_to_numa(dst_cpu)

            task_name = self.task_info.get(pid, {}).get('name', 'unknown')

            record = {
                'timestamp': time.time_ns(),
                'pid': pid,
                'comm': task_name,

                'src_cpu': src_cpu,
                'src_load': round(src_load, 2),
                'src_runqueue_len': src_runqueue,
                'src_numa_node': src_numa,
                'src_cpu_idle': 1 if src_load < 5.0 else 0,

                'dst_cpu': dst_cpu,
                'dst_load': round(dst_load, 2),
                'dst_runqueue_len': dst_runqueue,
                'dst_numa_node': dst_numa,
                'dst_cpu_idle': 1 if dst_load < 5.0 else 0,

                'cross_node': 1 if src_numa != dst_numa else 0,
                'load_diff': round(abs(src_load - dst_load), 2),
                'load_imbalance': round(src_load - dst_load, 2),

                'decision': 1 if migrated else 0
            }

            return record

        except Exception:
            return None

    # ---------------- SNAPSHOT ----------------
    def take_snapshot(self):
        current_cpu_times = psutil.cpu_times(percpu=True)

        cpu_runnable_counts = defaultdict(int)
        current_tasks = {}

        processes = list(psutil.process_iter(['pid', 'name', 'cpu_num', 'status']))

        for proc in processes:
            try:
                info = proc.info
                pid = info['pid']
                cpu_num = info['cpu_num']

                if cpu_num is None or cpu_num < 0 or cpu_num >= self.num_cpus:
                    continue

                if info['status'] not in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]:
                    continue

                current_tasks[pid] = cpu_num
                cpu_runnable_counts[cpu_num] += 1

                if pid not in self.task_info:
                    self.task_info[pid] = {
                        'name': info['name'],
                        'first_seen': time.time()
                    }

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        for pid, cpu_num in current_tasks.items():
            if pid in self.task_locations:
                prev_cpu = self.task_locations[pid]

                if prev_cpu != cpu_num:
                    record = self.create_record(
                        pid, prev_cpu, cpu_num, True,
                        current_cpu_times, cpu_runnable_counts
                    )
                    if record:
                        self.data_buffer.append(record)
                        self.stats['positive'] += 1
                else:
                    if self.stats['total'] % 5 == 0:
                        candidate_cpu = (cpu_num + 1) % self.num_cpus
                        record = self.create_record(
                            pid, cpu_num, candidate_cpu, False,
                            current_cpu_times, cpu_runnable_counts
                        )
                        if record:
                            self.data_buffer.append(record)
                            self.stats['negative'] += 1

            self.stats['total'] += 1

        self.task_locations = current_tasks

        for pid in list(self.task_info.keys()):
            if pid not in current_tasks:
                del self.task_info[pid]

        # IMPORTANT: update CPU times AFTER snapshot
        self.prev_cpu_times = current_cpu_times

        if len(self.data_buffer) >= self.buffer_size:
            self.flush_buffer()

    # ---------------- BUFFER ----------------
    def flush_buffer(self):
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

    # ---------------- LOOP ----------------
    def collect(self, duration=None):
        print("[*] Starting state snapshot collection...")
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
            print(f"\n[*] Data saved to {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Simple state snapshot collector')
    parser.add_argument('-d', '--duration', type=int)
    parser.add_argument('-i', '--interval', type=int, default=100)
    parser.add_argument('-o', '--output', default='data/raw/state_snapshots.jsonl')
    args = parser.parse_args()

    collector = SimpleStateCollector(
        output_file=args.output,
        interval_ms=args.interval
    )

    collector.collect(duration=args.duration)


if __name__ == '__main__':
    main()