/**
 * ONNX Agent - Handles model loading and inference
 */

const ONNXAgent = {
    session: null,
    modelLoaded: false,
    modelPath: 'models/gomoku_agent.onnx',

    /**
     * Load ONNX model
     */
    async loadModel(modelPath = null) {
        try {
            if (modelPath) {
                this.modelPath = modelPath;
            }

            Utils.log(`Loading ONNX model from ${this.modelPath}`);

            // Check if model file exists
            const response = await fetch(this.modelPath, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`Model file not found: ${this.modelPath}`);
            }

            // Create ONNX Runtime session
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            this.modelLoaded = true;
            Utils.log('✓ ONNX model loaded successfully');

            // Log model details
            const inputs = this.session.inputNames;
            const outputs = this.session.outputNames;
            Utils.log(`Model inputs: ${inputs}`);
            Utils.log(`Model outputs: ${outputs}`);

            return true;
        } catch (error) {
            console.error('Failed to load ONNX model:', error);
            this.modelLoaded = false;
            throw error;
        }
    },

    /**
     * Get agent action using model inference
     */
    async getAction(board, validActionsMask = null) {
        if (!this.modelLoaded || !this.session) {
            throw new Error('ONNX model not loaded');
        }

        try {
            // Prepare input: reshape board to (1, 1, 9, 9)
            const boardTensor = this.prepareBoardInput(board);

            // Run inference
            const start = performance.now();
            const feeds = {
                'board_state': boardTensor
            };

            const results = await this.session.run(feeds);
            const elapsed = performance.now() - start;

            // Get Q-values
            const qValuesData = results['q_values'].data;
            let qValues = new Float32Array(qValuesData);

            // Apply valid actions mask if provided
            if (validActionsMask) {
                qValues = this.applyMask(qValues, validActionsMask);
            }

            // Select best action
            let bestAction = 0;
            let bestQ = qValues[0];

            for (let i = 1; i < 81; i++) {
                if (qValues[i] > bestQ) {
                    bestQ = qValues[i];
                    bestAction = i;
                }
            }

            return {
                action: bestAction,
                qValues: qValues,
                inferenceTime: elapsed
            };
        } catch (error) {
            console.error('Inference failed:', error);
            throw error;
        }
    },

    /**
     * Prepare board input for ONNX model
     */
    prepareBoardInput(board) {
        // Reshape 1D board (81) to 4D tensor (1, 1, 9, 9)
        const input = new Float32Array(81);

        for (let i = 0; i < 81; i++) {
            input[i] = board[i];
        }

        // Create tensor with shape (1, 1, 9, 9)
        return new ort.Tensor('float32', input, [1, 1, 9, 9]);
    },

    /**
     * Apply valid actions mask to Q-values
     */
    applyMask(qValues, mask) {
        const masked = new Float32Array(qValues);
        const INF = 1e10;

        for (let i = 0; i < 81; i++) {
            if (mask[i] === 0) {
                masked[i] = -INF;
            }
        }

        return masked;
    },

    /**
     * Get model status
     */
    getStatus() {
        return {
            loaded: this.modelLoaded,
            hasSession: this.session !== null,
            modelPath: this.modelPath
        };
    },

    /**
     * Check if model is ready
     */
    isReady() {
        return this.modelLoaded && this.session !== null;
    }
};
