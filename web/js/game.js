/**
 * Game Logic - Manages game state and turn flow
 */

const Game = {
    board: new Array(81).fill(0),
    gameActive: false,
    currentPlayer: 1, // 1 for human, -1 for AI
    moveHistory: [],
    stats: {
        gamesPlayed: 0,
        humanWins: 0,
        aiWins: 0,
        draws: 0
    },

    /**
     * Initialize new game
     */
    init() {
        this.resetBoard();
        this.gameActive = true;
        this.currentPlayer = 1; // Human starts
        this.moveHistory = [];

        // Load stats from localStorage
        this.loadStats();
        this.updateUI();

        Utils.log('New game started');
    },

    /**
     * Reset board but keep stats
     */
    resetBoard() {
        this.board = new Array(81).fill(0);
        this.moveHistory = [];
    },

    /**
     * Make a move on the board
     */
    makeMove(action, player) {
        if (!this.isValidMove(action)) {
            return false;
        }

        this.board[action] = player;
        this.moveHistory.push({ action, player, timestamp: Date.now() });

        return true;
    },

    /**
     * Check if move is valid
     */
    isValidMove(action) {
        return action >= 0 && action < 81 && this.board[action] === 0;
    },

    /**
     * Get valid moves
     */
    getValidMoves() {
        return Utils.getValidActions(this.board);
    },

    /**
     * Get game state
     */
    getGameState() {
        return Utils.checkGameState(this.board);
    },

    /**
     * Check if game is over
     */
    isGameOver() {
        const state = this.getGameState();
        return state !== 'ongoing';
    },

    /**
     * End game and record result
     */
    endGame(result) {
        this.gameActive = false;
        this.stats.gamesPlayed++;

        if (result === 'human_win') {
            this.stats.humanWins++;
        } else if (result === 'ai_win') {
            this.stats.aiWins++;
        } else if (result === 'draw') {
            this.stats.draws++;
        }

        this.saveStats();
        this.updateUI();

        Utils.log(`Game ended: ${result}`);
    },

    /**
     * Make human move
     */
    async makeHumanMove(row, col) {
        if (!this.gameActive || this.currentPlayer !== 1) {
            return false;
        }

        const action = Utils.posToAction(row, col);

        if (!this.makeMove(action, 1)) {
            return false;
        }

        this.updateUI();

        // Check if game ends
        if (this.isGameOver()) {
            const result = this.getGameState();
            this.endGame(result);
            return true;
        }

        // Switch to AI
        this.currentPlayer = -1;
        this.updateUI();

        // Let UI update before AI move
        await new Promise(resolve => setTimeout(resolve, 500));

        return await this.makeAIMove();
    },

    /**
     * Make AI move using ONNX model
     */
    async makeAIMove() {
        if (!this.gameActive || this.currentPlayer !== -1) {
            return false;
        }

        if (!ONNXAgent.isReady()) {
            console.error('AI model not ready');
            this.updateUI('Model not ready for AI move');
            return false;
        }

        try {
            const validMask = Utils.getValidActionsMask(this.board);
            const result = await ONNXAgent.getAction(this.board, validMask);
            const action = result.action;

            Utils.log(`AI selected action ${action} (Q-value: ${result.qValues[action].toFixed(3)})`);

            if (!this.makeMove(action, -1)) {
                console.error('Invalid AI move');
                return false;
            }

            this.updateUI();

            // Check if game ends
            if (this.isGameOver()) {
                const result = this.getGameState();
                this.endGame(result);
                return true;
            }

            // Switch back to human
            this.currentPlayer = 1;
            this.updateUI();

            return true;
        } catch (error) {
            console.error('AI move failed:', error);
            this.updateUI('AI move error: ' + error.message);
            return false;
        }
    },

    /**
     * Undo last move
     */
    undoMove() {
        if (this.moveHistory.length < 1) {
            return false;
        }

        // Remove last move
        const lastMove = this.moveHistory.pop();
        this.board[lastMove.action] = 0;

        this.currentPlayer = 1;
        this.updateUI();

        return true;
    },

    /**
     * Update UI display
     */
    updateUI(message = null) {
        // Update board
        BoardRenderer.render(this.board);

        // Update status
        const statusEl = document.getElementById('gameStatus');
        if (message) {
            statusEl.textContent = message;
            statusEl.className = 'error';
        } else if (!this.gameActive) {
            const gameState = this.getGameState();
            statusEl.textContent = Utils.formatGameStatus(gameState);

            if (gameState === 'human_win') {
                statusEl.className = 'active';
            } else if (gameState === 'ai_win') {
                statusEl.className = 'error';
            }
        } else {
            statusEl.textContent = 'Game in progress...';
            statusEl.className = 'active';
        }

        // Update turn indicator
        const turnEl = document.getElementById('currentTurn');
        if (this.gameActive) {
            if (this.currentPlayer === 1) {
                turnEl.innerHTML = '<span class="stone human">●</span> Your turn';
            } else {
                turnEl.innerHTML = '<span class="stone ai">●</span> AI thinking...';
            }
        } else {
            const winner = this.getGameState();
            if (winner === 'human_win') {
                turnEl.innerHTML = '<span class="stone human">●</span> You won!';
            } else if (winner === 'ai_win') {
                turnEl.innerHTML = '<span class="stone ai">●</span> AI won!';
            } else {
                turnEl.innerHTML = 'Game over - Draw';
            }
        }

        // Update stats
        document.getElementById('gamesPlayed').textContent = this.stats.gamesPlayed;
        document.getElementById('humanWins').textContent = this.stats.humanWins;
        document.getElementById('aiWins').textContent = this.stats.aiWins;
        document.getElementById('draws').textContent = this.stats.draws;
    },

    /**
     * Save stats to localStorage
     */
    saveStats() {
        localStorage.setItem('gomokuStats', JSON.stringify(this.stats));
    },

    /**
     * Load stats from localStorage
     */
    loadStats() {
        const saved = localStorage.getItem('gomokuStats');
        if (saved) {
            try {
                this.stats = JSON.parse(saved);
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }
    },

    /**
     * Clear all stats
     */
    clearStats() {
        this.stats = {
            gamesPlayed: 0,
            humanWins: 0,
            aiWins: 0,
            draws: 0
        };
        this.saveStats();
        this.updateUI();
    }
};
