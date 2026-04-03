import torch
import GPUtil
import json

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []

    def snapshot(self, label=""):
        gpus = GPUtil.getGPUs()
        self.snapshots.append({
            'label': label,
            'gpus': [{'id': g.id, 'memory_used_mb': g.memoryUsed} for g in gpus]
        })

    def get_peak_memory(self):
        peak = {}
        for snap in self.snapshots:
            for gpu in snap['gpus']:
                peak[gpu['id']] = max(peak.get(gpu['id'], 0), gpu['memory_used_mb'])
        return peak

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({'snapshots': self.snapshots, 'peak': self.get_peak_memory()}, f, indent=2)
