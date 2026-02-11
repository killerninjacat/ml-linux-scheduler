#!/usr/bin/env python3
"""
RAPL Energy Collector
Monitors CPU package power consumption
"""

import time
import json
import os
import argparse

class RAPLCollector:
    def __init__(self, output_file="data/raw/rapl_data.jsonl"):
        self.output_file = output_file
        self.rapl_base = "/sys/class/powercap/intel-rapl"
        self.packages = []
        
        print("[*] Initializing RAPL collector...")
        self.discover_rapl_domains()
    
    def discover_rapl_domains(self):
        if not os.path.exists(self.rapl_base):
            print("[!] RAPL not available (non-Intel CPU or RAPL disabled)")
            print("[!] Will continue without energy data")
            return
        
        for entry in os.listdir(self.rapl_base):
            if entry.startswith("intel-rapl:"):
                domain_path = os.path.join(self.rapl_base, entry)
                name_file = os.path.join(domain_path, "name")
                
                if os.path.exists(name_file):
                    with open(name_file, 'r') as f:
                        name = f.read().strip()
                    
                    if name.startswith("package-"):
                        energy_file = os.path.join(domain_path, "energy_uj")
                        max_energy_file = os.path.join(domain_path, "max_energy_range_uj")
                        
                        self.packages.append({
                            'name': name,
                            'energy_file': energy_file,
                            'max_energy_file': max_energy_file,
                            'prev_energy': 0,
                            'total_energy': 0
                        })
                        
                        print(f"    ✓ Found {name}")
    
    def read_energy(self, package):
        try:
            with open(package['energy_file'], 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    
    def collect_sample(self):
        timestamp = time.time_ns()
        records = []
        
        for pkg in self.packages:
            current_energy = self.read_energy(pkg)
            
            # Handle wraparound
            if current_energy < pkg['prev_energy']:
                with open(pkg['max_energy_file'], 'r') as f:
                    max_energy = int(f.read().strip())
                delta = (max_energy - pkg['prev_energy']) + current_energy
            else:
                delta = current_energy - pkg['prev_energy']
            
            pkg['total_energy'] += delta
            pkg['prev_energy'] = current_energy
            
            record = {
                'timestamp': timestamp,
                'package': pkg['name'],
                'energy_uj': current_energy,
                'delta_uj': delta,
                'total_uj': pkg['total_energy']
            }
            
            records.append(record)
        
        return records
    
    def collect(self, duration=None, interval=0.1):
        if not self.packages:
            print("[!] No RAPL packages available, skipping energy collection")
            return
        
        print(f"[*] Collecting RAPL data (interval: {interval}s)...")
        
        for pkg in self.packages:
            pkg['prev_energy'] = self.read_energy(pkg)
        
        start_time = time.time()
        sample_count = 0
        
        try:
            with open(self.output_file, 'w') as f:
                while True:
                    records = self.collect_sample()
                    
                    for record in records:
                        f.write(json.dumps(record) + '\n')
                    
                    sample_count += 1
                    if sample_count % 10 == 0:
                        print(f"[*] Collected {sample_count} energy samples")
                    
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            print("\n[*] Stopping...")
        finally:
            print(f"[*] Data saved to {self.output_file}")
            self.print_summary()
    
    def print_summary(self):
        print("\n[*] Energy Summary:")
        for pkg in self.packages:
            energy_j = pkg['total_energy'] / 1_000_000
            print(f"    {pkg['name']}: {energy_j:.2f} J")

def main():
    parser = argparse.ArgumentParser(description='Collect RAPL energy data')
    parser.add_argument('-d', '--duration', type=int)
    parser.add_argument('-i', '--interval', type=float, default=0.1)
    parser.add_argument('-o', '--output', default='data/raw/rapl_data.jsonl')
    args = parser.parse_args()
    
    collector = RAPLCollector(output_file=args.output)
    collector.collect(duration=args.duration, interval=args.interval)

if __name__ == '__main__':
    main()
