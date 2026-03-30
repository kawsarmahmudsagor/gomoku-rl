/**
 * Utility functions for Gomoku game
 */

const Utils = {
    /**
     * Convert action index (0-80) to board position (row, col)
     */
    actionToPos(action) {
        return {
            row: Math.floor(action / 9),
            col: action % 9
        };
    },

    /**
     * Convert board position to action index
     */
    posToAction(row, col) {
        return row * 9 + col;
    },

    /**
     * Check if position is valid on board
     */
    isValidPos(row, col) {
        return row >= 0 && row < 9 && col >= 0 && col < 9;
    },

    /**
     * Get all valid actions (empty cells)
     */
    getValidActions(board) {
        const valid = [];
        for (let i = 0; i < 81; i++) {
            if (board[i] === 0) {
                valid.push(i);
            }
        }
        return valid;
    },

    /**
     * Create valid actions mask
     */
    getValidActionsMask(board) {
        const mask = new Array(81).fill(0);
        for (let i = 0; i < 81; i++) {
            if (board[i] === 0) {
                mask[i] = 1;
            }
        }
        return mask;
    },

    /**
     * Check 5 consecutive stones for given player
     */
    checkWinner(board, player) {
        // Convert 1D array to 2D for easier checking
        const grid = [];
        for (let i = 0; i < 9; i++) {
            grid[i] = board.slice(i * 9, (i + 1) * 9);
        }

        const WIN_LENGTH = 5;

        // Check rows
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9 - WIN_LENGTH + 1; col++) {
                if (grid[row].slice(col, col + WIN_LENGTH).every(cell => cell === player)) {
                    return true;
                }
            }
        }

        // Check columns
        for (let col = 0; col < 9; col++) {
            for (let row = 0; row < 9 - WIN_LENGTH + 1; row++) {
                let found = true;
                for (let i = 0; i < WIN_LENGTH; i++) {
                    if (grid[row + i][col] !== player) {
                        found = false;
                        break;
                    }
                }
                if (found) return true;
            }
        }

        // Check diagonals (top-left to bottom-right)
        for (let row = 0; row < 9 - WIN_LENGTH + 1; row++) {
            for (let col = 0; col < 9 - WIN_LENGTH + 1; col++) {
                let found = true;
                for (let i = 0; i < WIN_LENGTH; i++) {
                    if (grid[row + i][col + i] !== player) {
                        found = false;
                        break;
                    }
                }
                if (found) return true;
            }
        }

        // Check diagonals (top-right to bottom-left)
        for (let row = 0; row < 9 - WIN_LENGTH + 1; row++) {
            for (let col = WIN_LENGTH - 1; col < 9; col++) {
                let found = true;
                for (let i = 0; i < WIN_LENGTH; i++) {
                    if (grid[row + i][col - i] !== player) {
                        found = false;
                        break;
                    }
                }
                if (found) return true;
            }
        }

        return false;
    },

    /**
     * Check if board is full
     */
    isBoardFull(board) {
        return board.every(cell => cell !== 0);
    },

    /**
     * Get game result
     */
    checkGameState(board) {
        if (Utils.checkWinner(board, 1)) {
            return 'human_win';
        }
        if (Utils.checkWinner(board, -1)) {
            return 'ai_win';
        }
        if (Utils.isBoardFull(board)) {
            return 'draw';
        }
        return 'ongoing';
    },

    /**
     * Clone board state
     */
    cloneBoard(board) {
        return [...board];
    },

    /**
     * Format game state message
     */
    formatGameStatus(state) {
        const messages = {
            'ongoing': 'Game in progress',
            'human_win': '🎉 You won!',
            'ai_win': '🤖 AI won!',
            'draw': '🤝 It\'s a draw!'
        };
        return messages[state] || 'Unknown state';
    },

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Format numbers with commas
     */
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    /**
     * Get timestamp
     */
    getTimestamp() {
        return new Date().toLocaleTimeString();
    },

    /**
     * Log with timestamp
     */
    log(message) {
        console.log(`[${this.getTimestamp()}] ${message}`);
    }
};
