/**
 * Board Renderer - Handles Canvas drawing
 */

const BoardRenderer = {
    canvas: null,
    ctx: null,
    boardSize: 9,
    cellSize: 0,

    /**
     * Initialize canvas
     */
    init(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // Get actual canvas size
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;

        this.cellSize = this.canvas.width / (this.boardSize + 1);

        // Handle canvas resizing
        window.addEventListener('resize', () => this.onResize());

        Utils.log('Board renderer initialized');
    },

    /**
     * Handle canvas resize
     */
    onResize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.cellSize = this.canvas.width / (this.boardSize + 1);
    },

    /**
     * Draw the game board
     */
    drawBoard() {
        const ctx = this.ctx;
        const offset = this.cellSize;

        // Clear canvas
        ctx.fillStyle = '#d4af7a';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;

        for (let i = 0; i < this.boardSize; i++) {
            // Horizontal lines
            ctx.beginPath();
            ctx.moveTo(offset, offset + i * this.cellSize);
            ctx.lineTo(offset + (this.boardSize - 1) * this.cellSize, offset + i * this.cellSize);
            ctx.stroke();

            // Vertical lines
            ctx.beginPath();
            ctx.moveTo(offset + i * this.cellSize, offset);
            ctx.lineTo(offset + i * this.cellSize, offset + (this.boardSize - 1) * this.cellSize);
            ctx.stroke();
        }

        // Draw star points (for reference)
        ctx.fillStyle = '#333';
        const starPoints = [2, 6]; // Top-middle, middle, bottom-middle positions for 9x9
        for (const row of [2, 6]) {
            for (const col of [2, 6]) {
                ctx.beginPath();
                ctx.arc(
                    offset + col * this.cellSize,
                    offset + row * this.cellSize,
                    3,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            }
        }
    },

    /**
     * Draw a single stone
     */
    drawStone(row, col, player) {
        const ctx = this.ctx;
        const offset = this.cellSize;
        const x = offset + col * this.cellSize;
        const y = offset + row * this.cellSize;
        const radius = this.cellSize * 0.4;

        ctx.fillStyle = player === 1 ? '#2c3e50' : '#ecf0f1';
        ctx.strokeStyle = player === 1 ? '#000' : '#666';
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    },

    /**
     * Draw all stones on board
     */
    drawStones(board) {
        for (let action = 0; action < board.length; action++) {
            const cell = board[action];
            if (cell !== 0) {
                const pos = Utils.actionToPos(action);
                this.drawStone(pos.row, pos.col, cell);
            }
        }
    },

    /**
     * Render complete board state
     */
    render(board) {
        this.drawBoard();
        this.drawStones(board);
    },

    /**
     * Highlight a position
     */
    highlightPosition(row, col, color = 'rgba(255, 0, 0, 0.3)') {
        const ctx = this.ctx;
        const offset = this.cellSize;
        const x = offset + col * this.cellSize;
        const y = offset + row * this.cellSize;
        const size = this.cellSize * 0.3;

        ctx.fillStyle = color;
        ctx.fillRect(x - size, y - size, size * 2, size * 2);
    },

    /**
     * Get click position on board
     */
    getClickPosition(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const offset = this.cellSize;
        const col = Math.round((x - offset) / this.cellSize);
        const row = Math.round((y - offset) / this.cellSize);

        if (Utils.isValidPos(row, col)) {
            return { row, col };
        }
        return null;
    },

    /**
     * Draw indicator for last move
     */
    drawLastMove(row, col) {
        const ctx = this.ctx;
        const offset = this.cellSize;
        const x = offset + col * this.cellSize;
        const y = offset + row * this.cellSize;
        const size = this.cellSize * 0.15;

        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 3;
        ctx.strokeRect(x - size, y - size, size * 2, size * 2);
    }
};
