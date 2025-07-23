## Bot Decision-Making Process

The core decision-making process of the bot is a brute-force algorithm that evaluates the **expected return** of each valid tile placement from the tiles currently in hand.

---

## Key Rules and Assumptions Enabling Brute Force

The following rules and constraints make a brute-force strategy both feasible and effective:

1. **Time Limit**  
   - The per-turn time allowance is 1 second, with a total cap of 8 seconds per game.
   - This provides sufficient time for thorough evaluation of all moves.

2. **Fixed Number of Rounds**  
   - A maximum of 21 rounds per game allows ~0.38 seconds per move on average.

3. **Reasonable Search Space**  
   - With 84 tiles in the game and manual testing showing ~100–200 valid placements per round, it's computationally reasonable to evaluate every placement.

4. **4-Player Game Dynamics**  
   - Due to this game is played by 4 players, the board state changes significantly between rounds, reducing the effectiveness of long-term opponent tracking and allowing the assumption that decisions made in every round are independent.

---

## Bot structure

All evaluation is performed during the **tile placement phase**, and the resulting information is passed to the **meeple placement phase** through the `bot_state` class. In the river stage, the bot aims to create alternating turns and only place meeples on city and monastery.

### Placement Evaluation

Each valid tile placement is scored by the `placement_value_estimator`, which considers:

- **Connected Structure Analysis**  
  Detects and analyzes cities, roads, monasteries, etc., measuring:
  - Structure size  
  - Ownership
  - Number of open edges

- **Completion Probability**  
  Uses combinatorics and remaining tile counts to estimate the chance of completing open features.

- **Scoring Components**:
  - **Potential Points**: Assumes this could be the final round and maximizes the score accordingly.
  - **Completion Likelihood**: Completion probabilities are added as weights to prioritize easier-to-complete structures.
  - **Meeple Cost**: A linear penalty is applied based on remaining meeples.
  - **Monastery Consideration**: Tracks how many monasteries are on the board and who owns them. Prioritizes placements that support my monasteries and avoid helping enemy ones.
  - **Meeple Placement Condition**: Meeples are only placed if the expected return of that placement exceeds the meeple cost.

### Selection

The move with the **highest weighted score** is selected, and the `bot_state` is updated for the subsequent meeple placement phase.

### Combinatorics Use

Probabilities are calculated using combinatorics to estimate:
- The chance of drawing all required tiles to complete a structure based on the available tiles and remaining rounds of the game. This means finding the probability of completing each edges and then using combinatorics to estimate the total probability of being able to complete all edges.
- This will help the bot in prioritising placing tiles near structures that are easier to complete

---

## Endgame Strategy

In later stages of the game, an **"endgame mode"** is activated:
- Reduces the weight of meeple cost and encourages using remaining meeples before the game ends

---

## Ineffective Strategies

### Blocking or Sharing Cities

Tactics like blocking enemy cities or forcing city merges were explored, but ultimately discarded due to:

- Limited rounds available for both blocking and scoring
- Low point gain relative to effort
- Only affecting one opponent out of three

---

## Debugging and Strategy Testing Utilities

To support development, the following python scripts were created:

- **`parse_game_log.py`**  
  Converts `game.json` into readable move logs in the terminal (used before visualizers were available).

- **`run_game.py`**  
  Simulates games multiple times to evaluate strategy performance through:
  - Win rate analysis  
  - Average score comparison  
  Particularly helpful in later stages in which the bot has become a "black box" by evaluating all these factors mentioned above, and improvements in minor changes are hard to detected (e.g. running 400 games to assess performance improvements).
