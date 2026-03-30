/**
 * Main Application Entry Point
 */

document.addEventListener('DOMContentLoaded', async () => {
    Utils.log('Application starting...');

    try {
        // Initialize board renderer
        BoardRenderer.init('gameBoard');

        // Initialize game
        Game.init();

        // Load ONNX model
        updateModelStatus('loading');
        await ONNXAgent.loadModel();
        updateModelStatus('ready');

        // Setup event listeners
        setupEventListeners();

        // Initial UI render
        Game.updateUI();

        Utils.log('✓ Application ready');
    } catch (error) {
        console.error('Failed to initialize application:', error);
        updateModelStatus('error');
        document.getElementById('modelStatus').textContent = 'Error: ' + error.message;
    }
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Canvas click for moves
    const canvas = document.getElementById('gameBoard');
    canvas.addEventListener('click', handleCanvasClick);

    // Control buttons
    document.getElementById('newGameBtn').addEventListener('click', handleNewGame);
    document.getElementById('resetBtn').addEventListener('click', handleResetBoard);
    document.getElementById('undoBtn').addEventListener('click', handleUndoMove);

    Utils.log('Event listeners attached');
}

/**
 * Handle canvas clicks (human moves)
 */
async function handleCanvasClick(event) {
    if (!Game.gameActive || Game.currentPlayer !== 1) {
        return;
    }

    const pos = BoardRenderer.getClickPosition(event);
    if (!pos) {
        return;
    }

    // Make human move
    const success = await Game.makeHumanMove(pos.row, pos.col);

    if (!success) {
        alert('Invalid move! Cell already occupied.');
    }
}

/**
 * Handle new game
 */
async function handleNewGame() {
    Game.init();
    Game.updateUI();
    Utils.log('New game started');
}

/**
 * Handle reset board
 */
function handleResetBoard() {
    if (confirm('Are you sure you want to reset the board?')) {
        Game.resetBoard();
        Game.gameActive = false;
        Game.updateUI();
        Utils.log('Board reset');
    }
}

/**
 * Handle undo move
 */
function handleUndoMove() {
    if (Game.moveHistory.length === 0) {
        alert('No moves to undo');
        return;
    }

    Game.undoMove();
    Utils.log('Move undone');
}

/**
 * Update model status display
 */
function updateModelStatus(status) {
    const el = document.getElementById('modelStatus');

    const messages = {
        'loading': '⏳ Loading ONNX model...',
        'ready': '✓ Model ready (ONNX Runtime JS)',
        'error': '✗ Model loading failed'
    };

    el.textContent = messages[status] || 'Unknown status';
    el.className = status;
}

// Export for console debugging
window.Game = Game;
window.ONNXAgent = ONNXAgent;
window.BoardRenderer = BoardRenderer;
window.Utils = Utils;

Utils.log('Global exports available: Game, ONNXAgent, BoardRenderer, Utils');
