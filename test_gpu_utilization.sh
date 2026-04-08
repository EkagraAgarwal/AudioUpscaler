#!/bin/bash

echo "=========================================="
echo "GPU Utilization Test for RX 7700S"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Test 1: Basic GPU detection
echo "Test 1: GPU Detection"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
echo ""

# Test 2: Memory bandwidth test
echo "Test 2: Memory Bandwidth"
python -c "
import torch
import time

device = torch.device('cuda')
size = 100_000_000

x = torch.randn(size, device=device)
torch.cuda.synchronize()

start = time.time()
for _ in range(10):
    y = x * 2
    torch.cuda.synchronize()
elapsed = time.time() - start

bandwidth = (size * 4 * 2 * 10) / elapsed / 1e9
print(f'Memory bandwidth: {bandwidth:.2f} GB/s')
"
echo ""

# Test 3: Compute throughput test
echo "Test 3: Compute Throughput"
python -c "
import torch
import time

device = torch.device('cuda')
size = 8192

a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
torch.cuda.synchronize()

start = time.time()
for _ in range(10):
    c = torch.mm(a, b)
    torch.cuda.synchronize()
elapsed = time.time() - start

flops = 2 * size**3 * 10
tflops = flops / elapsed / 1e12
print(f'Compute throughput: {tflops:.2f} TFLOPS')
"
echo ""

echo "=========================================="
echo "GPU is ready for training!"
echo "=========================================="
