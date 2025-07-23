import subprocess
import re
from collections import defaultdict
from pathlib import Path

ENGINE_LOG = Path("/Users/maxyi/Desktop/public-carcassonne-game-engine/output/engine.log")
SUBMISSION_LOG_TEMPLATE = "/Users/maxyi/Desktop/public-carcassonne-game-engine/output/submission_{}.log"

def run_simulations(n=100):
    win_count = defaultdict(int)
    total_score = defaultdict(int)
    failed_matches = 0

    for i in range(n):
        print(f"\nRunning simulation {i + 1}/{n}")
        
        result = subprocess.run(
            ["python3", "match_simulator.py", "--submissions", "2:bot3_rework_prob.py", "2:bot3_gd_wr.py", "--engine"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        output = result.stdout + result.stderr

        match_success = re.search(
            r"match complete, outcome was {result_type='SUCCESS'.*?ranking=\[(.*?)\].*?score={(.*?)}",
            output
        )

        if match_success:
            ranking_str = match_success.group(1)
            scores_str = match_success.group(2)

            ranking = list(map(int, ranking_str.split(",")))
            scores = dict(map(lambda x: (int(x.split(":")[0].strip()), int(x.split(":")[1].strip())),
                              scores_str.split(",")))

            winner = ranking[0]
            win_count[winner] += 1
            for pid, score in scores.items():
                total_score[pid] += score
        else:
            failed_matches += 1
            print("[!] Match was not successful.")
            
            # Extract ban details if available
            match_fail = re.search(
                r"match complete, outcome was {result_type='PLAYER_BANNED'.*?player=(\d).*?}",
                output
            )
            if match_fail:
                banned_pid = int(match_fail.group(1))

                # Read last line of engine log
                try:
                    engine_lines = ENGINE_LOG.read_text().splitlines()
                    if engine_lines:
                        print("[engine.log] Last line:")
                        print(engine_lines[-1])
                except Exception as e:
                    print(f"[engine.log] Error reading log: {e}")

                # Read last 3 lines of player's log
                submission_log_path = Path(SUBMISSION_LOG_TEMPLATE.format(banned_pid))
                try:
                    submission_lines = submission_log_path.read_text().splitlines()
                    print(f"[submission_{banned_pid}.log] Last 3 lines:")
                    for line in submission_lines[-3:]:
                        print(line)
                except Exception as e:
                    print(f"[submission_{banned_pid}.log] Error reading log: {e}")
            else:
                print("Could not determine banned player.")

    # Summary
    print("\n--- Simulation Summary ---")
    total_matches = n
    successful_matches = total_matches - failed_matches

    for pid in sorted(set(win_count.keys()).union(total_score.keys())):
        winrate = win_count[pid] / successful_matches * 100
        avg_score = total_score[pid] / successful_matches if successful_matches else 0
        print(f"Bot {pid}: Win Rate = {winrate:.2f}%, Average Score = {avg_score:.2f}")

    print(f"\nFailed matches: {failed_matches}/{total_matches}")

if __name__ == "__main__":
    run_simulations(200)
