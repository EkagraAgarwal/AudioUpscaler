#!/bin/bash

# Monitor GPU utilization during training
# Run this in a separate terminal while training

echo "Monitoring GPU utilization... (Press Ctrl+C to stop)"
echo ""

watch -n 1 '
echo "=========================================="
echo "GPU Utilization Monitor - RX 7700S"
echo "=========================================="
rocm-smi
echo ""
echo "=========================================="
echo "Detailed Memory Usage"
echo "=========================================="
source venv/bin/activate
python -c "
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f\"Memory Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)\")
    print(f\"Memory Reserved:  {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)\")
"
'
