# Codebase Cleanup Report

## Files to Remove

### 1. Legacy Training Scripts (Redundant - replaced by train_optimized.sh)
- `run_training.sh` - Old training script (80 lines, complex)
- `run_full_training.sh` - Superseded by `train_optimized.sh`
- `run_quick_training.sh` - Superseded by `test_training.sh`

### 2. Test Output Files (Should be gitignored)
- `test_output/` - Testing artifacts
- `test_output2/` - Testing artifacts
- `train_test.log` - Log file
- `training_15epochs.log` - Log file
- `training_30epochs.log` - Log file

### 3. Potentially Unused Scripts (Need verification)
- `benchmark_optimizations.py` - Benchmarking script (489 lines)
  - NOTE: Keep if you plan to run benchmarks in future
  - Uses deprecated `torch.cuda.amp` (line 23)
  - Can be useful for performance testing

- `compress.py` - Audio compression utility (348 lines)
  - NOTE: Keep if you need to create compressed audio pairs
  - Not used in current training pipeline (uses on-the-fly compression)
  - Could be useful for pre-compressing audio

### 4. Documentation Consolidation
- Multiple overlapping documentation files:
  - `AMD_GPU_SETUP.md` (4.3 KB)
  - `DATA.md` (5.5 KB)
  - `GPU_OPTIMIZATION_GUIDE.md` (4.3 KB)
  - `INSTALL.md` (4.8 KB)
  - `MEMMAP_TRAINING.md` (7.0 KB)
  - `OPTIMIZATION_RESULTS.md` (5.8 KB)
  - `OPTIMIZATION_SUMMARY.md` (2.4 KB)
  - `QUICKSTART.md` (2.9 KB)
  - `TRAINING_FIXES.md` (2.9 KB)

## Recommended Actions

### Immediate Cleanup (Safe to remove):
1. Legacy training scripts
2. Test output directories
3. Log files

### Documentation Consolidation:
Consider consolidating into:
- `README.md` - Main documentation
- `SETUP.md` - Installation and GPU setup
- `TRAINING.md` - Training guide with optimizations
- `API.md` - Script usage documentation

### Files to Keep:
- `train_optimized.sh` - Current training script
- `test_training.sh` - Quick test script
- `monitor_gpu.sh` - GPU monitoring
- `test_gpu_utilization.sh` - GPU testing
- All source files in `src/`
- `download_data.py` - Data download utility
- `convert_to_wav.py` - WAV conversion utility
- `config.yaml` - Configuration file
- `requirements.txt` - Dependencies

## Proposed Changes

Will remove:
1. Legacy training scripts (run_training.sh, run_full_training.sh, run_quick_training.sh)
2. Test outputs and logs
3. Update .gitignore to ignore test outputs
