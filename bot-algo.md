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

## Placement Evaluation

For each valid tile placement, the bot uses `placement_value_estimator` to assign a score, factoring in:

#### Connected Structure Analysis

The bot traverses all features (cities, roads, monasteries, etc.) connected to the placement, gathering:

- The total size of the structure (including emblem bonuses for cities)  
- Which players have meeples on the structure  
- The number and coordinates of open edges (unfinished parts)

#### Completion Probability

For each open edge, the bot estimates the probability of completion by:

- Counting how many remaining tiles (in the deck and in hand) could fit and complete the structure at that spot  
- Using combinatorics to estimate the chance of drawing all needed tiles within the remaining rounds

#### Scoring Components:

- **Potential Points**: Calculates the score for completing the structure, including bonuses (like emblems for cities), and assumes the best-case scenario for the current round.  
- **Completion Likelihood**: Weights the score by the probability of completion, so easier-to-complete structures are prioritized.  
- **Meeple Cost**: Applies a penalty for using a meeple, which increases as the number of available meeples decreases (and is adjusted in endgame mode).  
- **Monastery Support**: Considers how many tiles surround monasteries (both mine and opponents’), and scores placements that help my monasteries or hinder opponents.  
- **Meeple Placement Decision**: Only recommends placing a meeple if the expected value (potential points × completion probability − meeple cost) is positive.

### Selection

The move with the **highest weighted score** is selected, and the `bot_state` is updated for the subsequent meeple placement phase.

### Combinatorics Use

Probabilities are calculated using combinatorics to estimate:
- The chance of drawing all required tiles to complete a structure based on the available tiles and remaining rounds of the game. This means finding the probability of completing each edges and then using combinatorics to estimate the total probability of being able to complete all edges.
- This will help the bot in prioritising placing tiles near structures that are easier to complete

---

## Endgame Strategy

In later stages of the game, an **"endgame mode"** is activated based on the highest scorer or remaining rounds:
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
