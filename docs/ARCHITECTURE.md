# System Architecture - Hybrid RL Gomoku Agent

## 🏗️ High-Level Overview

This project implements a complete pipeline from RL training to web deployment:

```
DEVELOPMENT
    ↓
[Python Training Loop]
    ├─ Gomoku Environment (rules engine)
    ├─ Double DQN Agent (learner)
    ├─ Experience Replay (memory)
    ├─ Random & Self-Play Opponents (training)
    └─ Checkpoint saving (every 10k episodes)
    ↓
[Model Export]
    ├─ PyTorch → ONNX conversion
    ├─ Weight serialization
    └─ File packaging
    ↓
DEPLOYMENT
    ↓
[Web Application]
    ├─ ONNX Runtime JS (inference engine)
    ├─ Canvas Board (UI rendering)
    ├─ Game Manager (turn logic)
    └─ Player Interface (human interaction)
```

## 🎮 Gomoku Environment

**File:** `python/environment/gomoku_env.py`

### Core Concepts
- **Board**: 9×9 grid with 81 possible positions
- **State**: Flattened array: 0=empty, 1=agent, -1=opponent
- **Actions**: 81 discrete choices (one per cell)
- **Transitions**: (state, action, reward, next_state, done)

### Game Rules
```
Win Condition: 5 consecutive stones
    - Any direction (horizontal, vertical, diagonal)
    - Win detection after every move

Termination:
    - Winning move detected
    - Board becomes full
    - Invalid move attempt (returns -0.5 reward)

Turn Order:
    - Agent plays as 1
    - Opponent plays as -1
    - Alternating moves
```

### State Representation Design

```
Board Grid (9×9):
    0 1 2 3 4 5 6 7 8
  ┌─────────────────────┐
0 │ · · · ● · · · · · │
1 │ · · · · · · · · · │
2 │ · · · · ○ · · · · │
3 │ · · ● · · · · · · │
...

Encoding:
  · (empty)  = 0
  ● (agent)  = 1
  ○ (opp)    = -1

Storage: Flattened to 1D array [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,-1,...]

For DQN:
  Reshape to (1, 1, 9, 9) for Conv2D input
```

### API Interface

```python
env = GomokuEnv()
state = env.reset()                     # → (9, 9) array

# Play moves
next_state, reward, done = env.step(action=40, player=1)

# Check validity
is_valid = env.is_valid(action=40)
valid_actions = env.get_valid_actions()  # → mask array

# Query game state
env.check_winner(player=1)  # → bool
env.is_game_over()          # → bool
```

## 🧠 Double DQN Agent

**File:** `python/agent/dqn.py`

### Architecture

```
POLICY NETWORK (weights being optimized)
    Input: (batch, 1, 9, 9)
    ↓
    Conv2D(1→64, 3×3, padding=1)
    ↓
    ReLU(64)
    ↓
    Conv2D(64→64, 3×3, padding=1)
    ↓
    ReLU(64)
    ↓
    Flatten → (batch, 3584)
    ↓
    Dense(3584→256)
    ↓
    ReLU(256)
    ↓
    Dense(256→81)
    ←─ OUTPUT: Q-values (one per action)

TARGET NETWORK (slow copy, τ=0.001 soft update)
    ├─ Same architecture
    ├─ Updated less frequently
    └─ Reduces overestimation bias
```

### Double DQN Algorithm

```
Q-learning with two networks:

Standard DQN:
    Q_target = r + γ * max(Q_target(s'))

Double DQN (reduces overestimation):
    best_action = argmax(Q_policy(s'))    # policy selects
    Q_target = r + γ * Q_target(s', best_action)  # target evaluates

    ← Breaks correlation: policy network chooses best action,
      target network evaluates it
```

### Training Process

```
For each episode:
    1. Initialize state s ← env.reset()

    2. Until done:
        a) Select action: ε-greedy policy
           - With probability ε: random valid action
           - Otherwise: argmax_a Q(s, a)

        b) Execute: (s', r, done) ← env.step(action, player)

        c) Store: (s, a, r, s', done) → replay buffer

        d) Training batch every N steps:
            - Sample batch from replay buffer
            - Compute target Q-values with target network
            - Minimize loss: ||Q_policy(s,a) - target_Q||²
            - Soft update: θ_target ← 0.001*θ_policy + 0.999*θ_target

        e) Epsilon decay: ε ← ε * decay_rate

    3. Log episode metrics
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 1e-4 | Adam optimizer step size |
| Gamma (γ) | 0.99 | Future reward discount |
| Epsilon Start | 1.0 | Full exploration initially |
| Epsilon End | 0.05 | Minimum exploration |
| Epsilon Decay | 500k | Steps to transition |
| Tau (τ) | 0.001 | Target network soft update |
| Batch Size | 32 | Mini-batch for training |
| Buffer Size | 100k | Replay buffer capacity |

## 🎓 Training Pipeline

**File:** `python/training/trainer.py` + `python/train.py`

### Episode Flow

```
Episode i:
    ├─ Choose opponent (80% random, 20% self-play)
    ├─ Initialize game
    │
    ├─ While game not over:
    │   ├─ Agent move:
    │   │   ├─ Get valid actions
    │   │   ├─ Select action (ε-greedy)
    │   │   ├─ Execute in environment
    │   │   ├─ Add to replay buffer
    │   │   ├─ Train on batch
    │   │   └─ Check game end
    │   │
    │   └─ Opponent move:
    │       ├─ Get valid actions (if random agent)
    │       ├─ Select action (if self-play agent)
    │       └─ Execute in environment
    │
    └─ Record result (win/loss/draw)

Metrics collected:
    ├─ Episode reward
    ├─ Game length (moves)
    ├─ Loss value
    ├─ Win rate (every 1000 episodes)
    └─ Epsilon value (exploration rate)
```

### Training Statistics

```
Track per evaluation window (1000 episodes):
    ├─ Win rate: P(agent wins vs random)
    ├─ Average reward: mean(episode_rewards)
    ├─ Loss curve: DQN training loss trend
    └─ Epsilon schedule: exploration decay

Expected progression:
    Episodes 0-50k:    Random exploration, 10-20% win rate
    Episodes 50k-200k: Learning phase, 40-70% win rate
    Episodes 200k-500k: Refinement, 80%+ win rate
```

### Checkpointing

```
Every 10,000 episodes:
    - Save: policy network, target network, optimizer state
    - File: models/agent_ep{N}.pt
    - Can resume training from any checkpoint
```

## 🌐 Model Export (PyTorch → ONNX)

**Integration:** End of `python/train.py`

### Export Process

```python
# Create ONNX representation
torch.onnx.export(
    network,                    # PyTorch model
    dummy_input,               # Example input (1, 1, 9, 9)
    "gomoku_agent.onnx",      # Output file
    input_names=['board_state'],
    output_names=['q_values']
)
```

### Format Comparison

```
PyTorch Model:
    └─ gomoku_agent.pt
    ├─ Policy network weights
    ├─ Target network weights
    ├─ Optimizer state
    └─ Size: ~5-10 MB (includes training info)

ONNX Model:
    └─ gomoku_agent.onnx
    ├─ Inference-only (no training info)
    ├─ Input: board_state (float32, shape [1,1,9,9])
    ├─ Output: q_values (float32, shape [1,81])
    └─ Size: ~2-3 MB (optimized for inference)

Browser Runtime:
    └─ ONNX Runtime JS
    ├─ Loads ONNX model
    ├─ Executes inference
    ├─ Returns Q-values
    └─ Latency: 10-50ms per forward pass
```

## 🎨 Web Architecture

**Files:** `web/js/*`

### Application Structure

```
Main Application (main.js)
    │
    ├─ BoardRenderer (Canvas drawing)
    │   ├─ drawBoard() - grid, lines, stars
    │   ├─ drawStones() - place pieces
    │   └─ getClickPosition() - hand le input
    │
    ├─ Game (Game state machine)
    │   ├─ board[] - 81-element state
    │   ├─ makeMove() - validation, placement
    │   ├─ makeHumanMove() - player input
    │   └─ makeAIMove() - NN inference
    │
    ├─ ONNXAgent (Model inference)
    │   ├─ loadModel() - load ONNX file
    │   ├─ getAction() - forward pass
    │   └─ applyMask() - filter invalid moves
    │
    └─ Utils (Helper functions)
        ├─ checkWinner() - 5-in-a-row detection
        ├─ getValidActions() - legal moves
        └─ gameStateToString() - UI display
```

### Turn-by-Turn Flow

```
User clicks cell (row, col):
    ↓
getClickPosition() → action index
    ↓
Game.makeHumanMove(row, col):
    ├─ Validate: is cell empty?
    ├─ Place: board[action] = 1
    ├─ Check: game over?
    ├─ If not over: Game.makeAIMove()
    └─ Update UI: render and display

Game.makeAIMove():
    ├─ Get valid moves mask
    ├─ ONNXAgent.getAction(board, mask):
    │   ├─ Prepare input: reshape board to (1,1,9,9)
    │   ├─ Run inference: Q-values ← model(input)
    │   ├─ Apply mask: invalid actions → -∞
    │   └─ Return: best_action = argmax(Q-values)
    ├─ Place AI stone: board[action] = -1
    ├─ Check: game over?
    └─ Update UI

UI Update:
    ├─ Clear canvas
    ├─ Draw grid
    ├─ Draw stones from board
    └─ Update status text and turn indicator
```

## 🔄 State Consistency

### Python ↔ JavaScript Parity

Must match exactly:

✅ **Implemented consistently:**
- Board encoding: 0/1/-1 same in both
- Win detection: 5-in-a-row logic identical
- Action mapping: row×9 + col → same
- Valid moves: only empty cells

✅ **Cross-validation:**
```
Python: env.check_winner(board, player=1)
JS:     Utils.checkWinner(board, 1)
        Must return SAME result for same board

Python: board[action]=0
JS:     board[action]===0
        Empty check must match
```

## 📊 Data Flow

```
[Training Phase]

Gomoku Env
    ├─ state: (9,9)
    ├─ action: 0-80
    └─ reward: float
        ↓
    Experience
        ├─ (state, action, reward, next_state, done)
        │
        ↓ add to buffer

Replay Buffer
    └─ 100k samples
        ↓ sample batch of 32

DQN Agent
    ├─ Forward on batch
    ├─ Compute loss
    ├─ Backprop
    └─ Update weights

Checkpoints saved every 10k episodes


[Deployment Phase]

ONNX Model
    ├─ Input: (1, 1, 9, 9) float32
    ├─ Conv→FC layers
    └─ Output: (1, 81) Q-values
        ↓ loaded in browser

JavaScript
    ├─ board: [0,1,-1,0,...] × 81
    ├─ mask: [1,0,1,...] × 81
    │
    ├─ inference
    │   ├─ tensorfy board → (1,1,9,9)
    │   ├─ model.run(input)
    │   └─ get Q-values
    │
    └─ action = argmax(Q-values * mask)
```

## ⚡ Performance Optimization

### Python Backend
- Conv2D layers for spatial feature extraction
- Batch processing (32 samples)
- GPU acceleration via PyTorch/CUDA
- Efficient numpy operations

### JavaScript Frontend
- ONNX Runtime WASM backend (100x faster than JS)
- Async/await for non-blocking UI
- Canvas rendering (GPU-accelerated)
- LocalStorage for stats persistence

### Model Compression
- ONNX export removes training data (~50% reduction)
- Only inference graph needed
- Small model: 2-3 MB (fits in browser cache)

---

**Next:** See TRAINING_GUIDE.md for detailed training instructions
