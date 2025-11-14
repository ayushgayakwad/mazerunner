# mazerunner

This project is an interactive, browser-based visualization of a Deep Q-Network (DQN) agent learning to solve a 10x10 maze. The entire reinforcement learning process, from model creation to training and inference, runs directly in your browser using TensorFlow.js.

You can create your own maze by setting a start point, a goal, and drawing obstacles, then watch as the agent learns the optimal path.

![Screenshot of the DQN Maze Learner in action](https://github.com/ayushgayakwad/mazerunner/blob/main/screenshots/mazerunner_screenshot.png?raw=true)

## Features

* **Interactive Maze Setup:** Click on the grid to place the **Start** (🔵), **Goal** (🟢), and **Obstacles** (🟥).
* **Live Training:** Watch the agent learn in real-time over 2,000 episodes.
* **Dual Visualization:**
    * **Agent's Path:** The left panel shows the agent's current *best-guess* path (using a greedy policy) as it trains. It updates periodically to show the agent's improving strategy.
    * **Q-Value Heatmap:** The right panel visualizes the neural network's learned Q-values for each state. The color (black -> red -> yellow) indicates the network's prediction of the *future reward* for being in that cell.
* **Real-time Stats:** Monitor key training metrics:
    * **Episode:** The current training iteration.
    * **Loss:** The model's prediction error (Mean Squared Error).
    * **Epsilon (ε):** The agent's current exploration rate. A value of 1.0 is 100% random actions, while 0.01 is 1% random.
    * **Replay Samples:** The number of experiences (`<state, action, reward, nextState>`) stored in the replay buffer.
* **Zero Installation:** Runs entirely in the browser. No dependencies or build steps required.

---

## How to Use

1.  Download the `mazerunner.html` file.
2.  Open the file in any modern web browser (like Chrome, Firefox, or Edge).
3.  **Click a cell** to set the **Start** position (🔵).
4.  **Click another cell** to set the **Goal** position (🟢).
5.  **Click any other cells** to toggle **Obstacles** (🟥).
6.  Press the **"Start Training"** button.
7.  Watch the agent learn! The left grid will show its current path, and the right grid will show its "brain" (the Q-values) evolving.
8.  Once training is complete, the left grid will display the **final best path** found by the agent.

---

## How It Works (Technical Details)

This project implements a standard Deep Q-Network (DQN) agent.

* **Agent:** The "brain" is a simple feed-forward neural network created with TensorFlow.js.
* **State:** The agent's state is its current position on the 10x10 grid, represented as a one-hot encoded vector of size 100 (e.g., being at cell `(0, 5)` is state `5`, represented by a vector with a `1` at the 5th index and `0`s elsewhere).
* **Actions:** The agent can take one of 4 discrete actions: `Up`, `Down`, `Left`, or `Right`.
* **Neural Network:** The Q-Network has the following architecture:
    * Input Layer: 100 units (for the one-hot state)
    * Hidden Layer 1: 64 units (ReLU activation)
    * Hidden Layer 2: 32 units (ReLU activation)
    * Output Layer: 4 units (Linear activation) - one for each action's predicted Q-value.
* **Rewards:** The agent receives rewards based on its actions:
    * Reach the **Goal**: `+1.0`
    * Hit an **Obstacle**: `-0.5`
    * Take any other **Step**: `-0.01` (a small penalty to encourage finding the shortest path).
* **Training Loop (DQN Algorithm):**
    1.  **Exploration:** The agent uses an **epsilon-greedy** strategy. It either chooses a random action (to explore) or the best action predicted by its network (to exploit). The `epsilon` (exploration rate) starts at `1.0` and decays to `0.01`.
    2.  **Replay Buffer:** Every experience—(`state`, `action`, `reward`, `nextState`, `terminated`)—is stored in a replay buffer.
    3.  **Batch Training:** During training, the agent samples a random `BATCH_SIZE` (64) of experiences from this buffer to train its Q-network. This breaks the correlation between consecutive steps and stabilizes learning.
    4.  **Target Network:** A second "target" network is used to calculate the target Q-values. This network's weights are frozen and only updated to match the main Q-network's weights every 5 episodes. This provides a stable target for the loss calculation and prevents the model from "chasing its own tail."

---

## Technologies Used

* **HTML5**
    Provides the core semantic structure for the web page.

* **Tailwind CSS (via CDN)**
    A utility-first CSS framework used for all layout, spacing, typography, and component styling.

* **JavaScript (ES6+ Module)**
    This is the engine of the entire application, running inside a single `<script type="module">` tag. It is responsible for:
    * **DOM Manipulation:** Dynamically generating the grid cells and handling all user click events for setting up the maze.
    * **Environment Logic:** Defining the maze environment, including states, actions, and the reward system (the `step` function).
    * **DQN Algorithm:** Implementing the core reinforcement learning logic, including the experience replay buffer (`replayBuffer`), the epsilon-greedy policy (`getAction`), and the main training loop (`startTraining`).
    * **Visualization:** Updating the CSS classes on the grids in real-time to show the agent's current path, the final best path, and the Q-value heatmap.
    * **State Management:** Tracking all hyperparameters (like `GAMMA`, `LEARNING_RATE`), the current grid setup, and the agent's training progress.

* **TensorFlow.js (via CDN)**
    The machine learning library that powers the agent's "brain." It is loaded via CDN and is used for all neural network operations:
    * **Model Creation:** Defining the architecture for both the Q-Network (`qNet`) and the Target Network (`targetNet`) using `tf.sequential` and `tf.layers.dense`.
    * **Training:** Compiling the model (`model.compile`) with the Adam optimizer and 'meanSquaredError' loss. It's used to train the Q-Network on batches of data from the replay buffer using `qNet.fit`.
    * **Inference:** Using `qNet.predict` to get the Q-values for a given state. This is crucial for the `getAction` function to decide the best action (exploitation).
    * **Tensor Management:** Efficiently creating and disposing of tensors (e.g., `tf.oneHot` for states, `tf.tensor1d`) and using `tf.tidy` to manage memory and prevent leaks during the training loop.
