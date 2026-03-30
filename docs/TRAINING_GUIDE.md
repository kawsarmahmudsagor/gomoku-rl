# Training Guide - Complete Instructions

## 📋 Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows (WSL2)
- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB+ recommended)
- **GPU** (optional): NVIDIA GPU with CUDA for faster training

### Software Setup

```bash
# 1. Clone/download project
git clone <repo> gomoku-rl
cd gomoku-rl

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r python/requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### GPU Setup (Optional)

For faster training, install CUDA support:

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch (if not already)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Training Scenarios

### Scenario 1: Default Training (Recommended)

For learning, use default settings:

```bash
cd gomoku-rl/python
python train.py
```

**Configuration:**
- Episodes: 200,000
- Checkpoint interval: 10,000
- Evaluation interval: 1,000
- Self-play ratio: 20%
- Learning rate: 1e-4

**Expected Results:**
- Takes ~30-60 minutes on CPU, 10-20 minutes on GPU
- ~50-60% win rate against random opponent by end
- Final model: `models/gomoku_agent_final.pt`
- ONNX export: `models/gomoku_agent.onnx`

**Expected Output:**
```
==============================================================
🎮 GOMOKU RL TRAINING
==============================================================
Episodes: 200,000
Learning Rate: 0.0001
Gamma: 0.99
Self-play Ratio: 0.2
Device: cuda

Training: 100%|████████| 200000/200000 [00:45:32<00:00, 73.3s/ep]
Episode 1000: Win Rate = 12.00%, Avg Reward = -0.150, Epsilon = 0.8934
Episode 2000: Win Rate = 18.00%, Avg Reward = -0.085, Epsilon = 0.7892
...
Episode 200000: Win Rate = 58.00%, Avg Reward = 0.420, Epsilon = 0.0500

✓ Evaluation complete: 58/100 wins (58.0% win rate)

Exporting model to ONNX format: models/gomoku_agent.onnx
✓ Model exported to models/gomoku_agent.onnx
  File size: 2.45 MB

==============================================================
✅ TRAINING COMPLETE
==============================================================
Final Win Rate: 58.0%
Models saved to: python/models
==============================================================
```

### Scenario 2: Extended Training (Better Performance)

For a stronger agent:

```bash
cd gomoku-rl/python
python train.py --episodes 500000 \
                 --self-play-ratio 0.4 \
                 --learning-rate 5e-5
```

**Configuration:**
- Episodes: 500,000
- Self-play ratio: 40% (more challenging training)
- Learning rate: 5e-5 (slower, more stable)
- Everything else: default

**Expected Results:**
- Takes ~2-3 hours on CPU, 30-60 minutes on GPU
- ~75-85% win rate against random opponent
- Significantly stronger gameplay

### Scenario 3: Quick Testing

For rapid iteration:

```bash
cd gomoku-rl/python
python train.py --episodes 20000 \
                 --checkpoint-interval 2000 \
                 --eval-interval 200
```

**Use case:**
- Testing code changes
- Verifying setup works
- Takes ~5 minutes on CPU
- Produces weak but playable agent

### Scenario 4: Continue Training

Resume from saved checkpoint:

```bash
cd gomoku-rl/python

# Train more episodes starting from checkpoint
# Creates new checkpoint from there
python train.py --episodes 300000 \
                 --model-path models/agent_ep100000.pt
```

**Note:** Checkpoint paths are from checkpoint files saved during training

### Scenario 5: Evaluation Only

Test existing model without training:

```bash
cd gomoku-rl/python

python train.py --eval-only \
                 --model-path models/gomoku_agent_final.pt
```

**Output:**
```
Evaluating agent (100 games)...
  Progress: 20/100 games
  Progress: 40/100 games
  Progress: 60/100 games
  Progress: 80/100 games
  Progress: 100/100 games
✓ Evaluation complete: 58/100 wins (58.0% win rate)

Final win rate: 58.0%
```

## 🔧 Advanced Configuration

### Command-line Arguments

```bash
python train.py --help

optional arguments:
  --episodes EPISODES           Total episodes to train [default: 200000]
  --learning-rate LR           Learning rate [default: 1e-4]
  --gamma GAMMA                Discount factor [default: 0.99]
  --buffer-size SIZE           Replay buffer size [default: 100000]
  --checkpoint-interval N      Save every N episodes [default: 10000]
  --eval-interval N            Evaluate every N episodes [default: 1000]
  --self-play-ratio RATIO      Fraction against self-play [default: 0.2]
  --no-export                  Skip ONNX export
  --eval-only                  Skip training, only evaluate
  --model-path PATH            Path to existing model
```

### Hyperparameter Tuning

#### Learning Rate Optimization
```bash
# Slower, more stable learning
python train.py --learning-rate 5e-5 --episodes 500000

# Faster, more aggressive learning
python train.py --learning-rate 2e-4 --episodes 100000
```

**Rule of thumb:**
- High LR (>1e-3): Unstable, may diverge
- Medium LR (1e-4): Good balance (default)
- Low LR (<1e-5): Slow convergence

#### Replay Buffer Size
```bash
# Larger buffer (more memory, better training)
python train.py --buffer-size 200000

# Smaller buffer (faster but less stable)
python train.py --buffer-size 50000
```

#### Self-play Ratio
```bash
# Early self-play (challenging early learning)
python train.py --self-play-ratio 0.0 --episodes 100000

# Balanced training
python train.py --self-play-ratio 0.2  # default

# Heavy self-play (agent learns from itself)
python train.py --self-play-ratio 0.8 --episodes 500000
```

## 📊 Monitoring Training

### Real-time Metrics

Training displays every 1000 episodes:

```
Episode 10000: Win Rate = 45.00%, Avg Reward = 0.150, Epsilon = 0.7234
```

**What these mean:**
- **Win Rate**: % of evaluation games won vs random opponent
- **Avg Reward**: Average episode reward (higher = better)
- **Epsilon**: Current exploration rate (decreases over time)

### Training Files

```
python/models/
├── agent_ep10000.pt         # Checkpoint at 10k episodes
├── agent_ep20000.pt         # Checkpoint at 20k episodes
├── agent_ep100000.pt        # Checkpoint at 100k episodes
├── ...
├── gomoku_agent_final.pt    # Final PyTorch model
├── gomoku_agent.onnx        # Final ONNX model (for web)
├── gomoku_agent_weights.json # Weight dump (reference)
└── logs/
    └── training_metrics.json
```

### Analyzing Training Logs

```python
import json

# Load metrics
with open('models/logs/training_metrics.json') as f:
    metrics = json.load(f)

# Extract data
episode_rewards = metrics['episode_rewards']
win_rates = [h['win_rate'] for h in metrics['win_rate_history']]

# Analyze
import statistics
print(f"Average reward: {statistics.mean(episode_rewards)}")
print(f"Final win rate: {win_rates[-1]}")
print(f"Episodes with high reward: {sum(1 for r in episode_rewards if r > 0.3)}")
```

## 🚀 Deploying Trained Model

### Step 1: Wait for Training to Complete

```
✅ TRAINING COMPLETE
Final Win Rate: XX.X%
Models saved to: python/models
```

### Step 2: Copy ONNX Model to Web

```bash
# From gomoku-rl root directory
cp python/models/gomoku_agent.onnx web/models/

# Verify file exists
ls -lh web/models/gomoku_agent.onnx
```

### Step 3: Launch Web Server

```bash
cd web

# Option 1: Python (built-in)
python -m http.server 8000

# Option 2: Node.js (if installed)
npx http-server -p 8000

# Option 3: Node.js with cache control
npx http-server -p 8000 -c-1

# Option 4: PHP (if installed)
php -S localhost:8000
```

### Step 4: Test in Browser

1. Open `http://localhost:8000`
2. Check console for model loading status (F12)
3. Model should say "✓ Model ready"
4. Play a test game

**Troubleshooting:**
- If model doesn't load, check file exists: `web/models/gomoku_agent.onnx`
- Check console for errors: F12 → Console tab
- Ensure web server is running on correct port

## 💾 Backing Up and Managing Models

### Save Current Best Model

```bash
# After training, copy the best model
cp python/models/gomoku_agent_final.pt \
   backups/gomoku_agent_excellent.pt

# Create backup of ONNX too
cp python/models/gomoku_agent.onnx \
   backups/gomoku_agent_excellent.onnx
```

### Version Control

```bash
# Add to git (if using version control)
git add python/models/gomoku_agent_final.pt web/models/gomoku_agent.onnx

# Or ignore large files in .gitignore
echo "python/models/*.pt" >> .gitignore
echo "web/models/*.onnx" >> .gitignore
```

## 🔍 Debugging Common Issues

### Issue: "CUDA Out of Memory"
```bash
# Reduce batch size or buffer size
python train.py --buffer-size 50000

# Or use CPU
python train.py --episodes 50000  # smaller training
```

### Issue: Training Very Slow

1. Check if GPU is being used:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

2. If False, install CUDA support
3. If True but still slow, check GPU utilization:
   ```bash
   # On Linux/Windows with NVIDIA GPU
   nvidia-smi  # Should show GPU usage > 50%
   ```

### Issue: Model Not Loading in Browser

```javascript
// In browser console (F12 → Console)
console.log(ONNXAgent.modelPath);              // Check path
console.log(ONNXAgent.isReady());              // Should be true
console.log(ONNXAgent.session);                // Should not be null
```

### Issue: Weak Agent (Win Rate < 30%)

1. **Too few episodes**: Train for at least 50k
2. **Learning rate too high**: Reduce to 5e-5
3. **Self-play too early**: Start with 0% self-play

## 📈 Performance Benchmarks

### Training Speed (per 1000 episodes)

| Hardware | Time | Notes |
|----------|------|-------|
| CPU (4 cores) | ~15 min | Baseline |
| CPU (8 cores) | ~8 min | Parallelization helps |
| GPU (RTX 2070) | ~2 min | Order of magnitude faster |
| GPU (RTX 3090) | ~30 sec | High-end GPU |

### Inference Speed (per move)

| Platform | Time | Notes |
|----------|------|-------|
| Python (CPU) | 10-20ms | PyTorch on CPU |
| Python (GPU) | 2-5ms | PyTorch on CUDA |
| Browser (WASM) | 10-50ms | ONNX Runtime JS |
| Browser (cached) | 5-10ms | After warmup |

### Model Quality vs Training

| Episodes | Expected Win Rate | Play Strength |
|----------|------------------|---------------|
| 10,000 | 10-20% | Very weak |
| 50,000 | 30-40% | Weak |
| 100,000 | 50-60% | Average |
| 200,000 | 60-70% | Good |
| 500,000 | 75-85% | Strong |

## 💡 Tips and Best Practices

### Training Tips
1. **Monitor early**: Check metrics after 10k episodes to spot issues
2. **Save gradually**: Checkpoints let you recover from crashes
3. **Use self-play late**: After agent reaches ~40% win rate
4. **Stop if diverging**: If loss increases, reduce learning rate

### Deployment Tips
1. **Test locally first**: Verify model works before deploying
2. **Cache busting**: Use query strings if updating model
3. **Monitor stats**: Check localStorage for player stats
4. **Backup models**: Keep multiple versions

### Performance Tips
1. **Profile your code**: Use cProfile to find bottlenecks
2. **Batch operations**: Train on larger batches when possible
3. **Use GPU**: Biggest single improvement
4. **Async game logic**: Non-blocking JS improves UI responsiveness

---

**Questions?** See README.md or ARCHITECTURE.md for more details.
