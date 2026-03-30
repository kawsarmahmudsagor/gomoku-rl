# 🎮 Gomoku - Hybrid RL Agent (PyTorch + ONNX Runtime JS)

A complete implementation of a 9×9 Gomoku agent trained with Double DQN in Python and deployed to the browser using ONNX Runtime JS.

## 🌐 System Architecture

```
┌─────────────────────────────────────────┐
│     Python Training Backend             │
├─────────────────────────────────────────┤
│ • Gomoku Environment (Gym-compatible)   │
│ • Double DQN Agent (PyTorch)            │
│ • Experience Replay Buffer              │
│ • Training Loop (200k-500k episodes)    │
│ • Model Export (→ ONNX)                 │
└────────── Python/train.py ──────────────┘
                 ↓↓↓
           gomoku_agent.onnx
                 ↓↓↓
┌─────────────────────────────────────────┐
│    Web Deployment (Browser)             │
├─────────────────────────────────────────┤
│ • ONNX Runtime JS (inference)           │
│ • Canvas Board (9×9 Gomoku)             │
│ • Real-time Gameplay                    │
│ • Game Statistics                       │
└────────── Web/index.html ────────────────┘
```

## 📋 Project Structure

```
gomoku-rl/
├── python/                      # Training backend
│   ├── requirements.txt         # Python dependencies
│   ├── train.py               # Main entry point
│   ├── environment/
│   │   └── gomoku_env.py      # 9×9 Gomoku game rules
│   ├── agent/
│   │   ├── network.py         # DQN neural network
│   │   ├── dqn.py            # Double DQN algorithm
│   │   └── experience_replay.py
│   ├── opponents/
│   │   ├── random_agent.py    # Random baseline
│   │   └── self_play_agent.py # Self-play training
│   ├── training/
│   │   └── trainer.py         # Training loop
│   └── models/                # Saved models & exports
│
├── web/                        # Web frontend
│   ├── index.html             # Main interface
│   ├── styles.css             # Styling
│   ├── js/
│   │   ├── main.js           # App entry point
│   │   ├── game.js           # Game logic
│   │   ├── agent.js          # ONNX model inference
│   │   ├── board-renderer.js # Canvas rendering
│   │   └── utils.js          # Helper functions
│   └── models/               # ONNX model files
│
└── docs/
    ├── ARCHITECTURE.md
    └── TRAINING_GUIDE.md
```

## 🚀 Quick Start

### 1. Train the Agent (Python)

```bash
cd gomoku-rl/python

# Install dependencies
pip install -r requirements.txt

# Train agent (default: 200k episodes)
python train.py

# With custom parameters
python train.py --episodes 500000 --learning-rate 1e-4 --self-play-ratio 0.3
```

**Training will:**
- Train against random opponents and self-play
- Save checkpoints every 10k episodes
- Export model to ONNX format automatically
- Display win rate metrics every 1k episodes

**Output files:**
- `python/models/gomoku_agent_final.pt` - Final PyTorch model
- `python/models/gomoku_agent.onnx` - ONNX model for web
- `python/models/gomoku_agent_weights.json` - Weights reference

### 2. Deploy Web App

```bash
cd web

# Option 1: Use Python's built-in server
python -m http.server 8000

# Option 2: Use Node.js http-server
npx http-server -p 8000

# Option 3: Use any web server
# Just serve the web/ directory and navigate to index.html
```

**Access:**
- Open `http://localhost:8000` in your browser
- ONNX model will auto-load from `web/models/gomoku_agent.onnx`

### 3. Copy Trained Model to Web

After training, copy the ONNX model to the web directory:

```bash
cp python/models/gomoku_agent.onnx web/models/
```

## 🎮 How to Play

1. **Board**: 9×9 Gomoku grid
2. **Your Stones**: Dark (●) - you go first
3. **AI Stones**: Light (○)
4. **Win Condition**: Get 5 consecutive stones (row, column, or diagonal)
5. **Controls**:
   - Click on empty cells to place your stone
   - "New Game" - Start fresh game
   - "Reset Board" - Clear board
   - "Undo Move" - Take back your last move

## 🧠 AI Agent Specs

| Component | Details |
|-----------|---------|
| **Board Size** | 9 × 9 |
| **Algorithm** | Double DQN |
| **Framework** | PyTorch 2.0+ |
| **Network** | Conv2D(2×) → Dense(2×) |
| **Output** | 81 Q-values (one per action) |
| **Replay Buffer** | 100k transitions |
| **Discount Factor (γ)** | 0.99 |
| **Epsilon Decay** | 1.0 → 0.05 over 500k steps |
| **Exploration** | ε-greedy |
| **Training Opponents** | Random (80%), Self-play (20%) |

### Reward Structure
- **Win**: +1.0
- **Loss**: -1.0
- **Draw**: 0.0
- **Invalid Move**: -0.5
- **Per Move**: -0.01

## 📊 Training Metrics

Training produces:
- **Win Rate**: % of games won vs random opponent
- **Average Reward**: Mean episode reward
- **Loss Curve**: DQN training loss over time
- **Epsilon Schedule**: Exploration decay

Example training output:
```
Episode 10000: Win Rate = 45.00%, Avg Reward = 0.150, Epsilon = 0.7234
Episode 20000: Win Rate = 62.00%, Avg Reward = 0.280, Epsilon = 0.5821
Episode 50000: Win Rate = 78.00%, Avg Reward = 0.450, Epsilon = 0.3451
Episode 100000: Win Rate = 85.00%, Avg Reward = 0.520, Epsilon = 0.2145
```

## 🔧 Advanced Training Options

```bash
# Longer training with higher self-play ratio
python train.py --episodes 500000 --self-play-ratio 0.5

# Faster training for testing
python train.py --episodes 50000 --checkpoint-interval 5000

# Evaluate existing model
python train.py --eval-only --model-path models/gomoku_agent_final.pt

# Skip ONNX export
python train.py --no-export
```

## 🌐 Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Full | Recommended |
| Firefox | ✅ Full | Good performance |
| Safari | ✅ Full | WASM support required |
| Edge | ✅ Full | Chromium-based |

**Requirements:**
- WebAssembly support (for ONNX Runtime)
- JavaScript ES6+
- LocalStorage (for stats persistence)

## ⚡ Performance

- **Model Size**: ~2-3 MB (ONNX)
- **Inference Time**: 10-50ms per move (depends on browser/hardware)
- **Memory Usage**: ~100-200 MB (browser process)

## 📝 Implementation Details

### Gomoku Rules
- 5 consecutive stones (any direction) = win
- Invalid moves = already occupied cells
- Draw = board full with no winner
- Optimal for 9×9 board: ~40-60 moves per game

### State Encoding
- **Format**: 9×9 grid flattened to (1×1×9×9)
- **Values**: 0 (empty), 1 (agent), -1 (opponent)
- **Input Shape for DQN**: (batch, 1, 9, 9)

### Action Space
- **Total Actions**: 81 (one per cell)
- **Action Index**: row × 9 + col
- **Valid Actions**: Only empty cells

### Model Architecture

**Policy Network:**
```
Input (1×9×9)
  ↓
Conv2D(1→64, kernel=3, padding=1)
  ↓
ReLU
  ↓
Conv2D(64→64, kernel=3, padding=1)
  ↓
ReLU
  ↓
Flatten → (576,)
  ↓
Dense(576→256)
  ↓
ReLU
  ↓
Dense(256→81)
  ↓
Output Q-values (81,)
```

## 🐛 Troubleshooting

### Model Not Loading in Browser
1. Check console for errors (F12)
2. Ensure `web/models/gomoku_agent.onnx` exists
3. Check CORS if accessing from different domain
4. Verify ONNX Runtime JS is loaded from CDN

### AI Taking Too Long
1. Check inference time in console (Game.makeAIMove)
2. May be slower on older devices
3. Browser performance varies

### Training Too Slow
1. Use GPU: Ensure CUDA is installed for PyTorch
2. Reduce batch size if memory errors
3. Run on machine with more cores

### Game Logic Issues
1. Check console for errors
2. Verify Python and JS implementations match
3. Test with simple scenarios first

## 📚 Documentation

- **ARCHITECTURE.md**: Detailed system design
- **TRAINING_GUIDE.md**: Extended training instructions
- **Code comments**: Inline documentation in all files

## 🔄 Extending the Project

### Add New Opponent Types
Implement in `python/opponents/` following the interface

### Improve Agent
- Use Dueling DQN (already in code, just enable)
- Add prioritized experience replay
- Implement rainbow DQN

### Enhance UI
- Add game replay
- Show AI reasoning (visualize Q-values)
- Multiplayer support
- Tournament mode

### Deploy Online
- Use web server (Firebase, Netlify, Heroku)
- Serve static files
- Optional: serverless API for statistics

## 📄 License

This project is open source. Feel free to use and modify.

## 🎓 Credits

Built with:
- **PyTorch** - Deep learning framework
- **ONNX** - Model interchange format
- **ONNX Runtime JS** - Browser inference
- **Vanilla JavaScript** - Web interface

---

**Made for learning and experimentation with RL agents!** 🚀
