import requests
import chess.pgn
import io
import json
import re

STUDY_IDS = [
    # --- ENDGAMES (Essential for coaching logic) ---
    "wukLYIXj", # Beginner Endgames You Must Know (NoseKnowsAll)
    "djBRTwos", # Intermediate Endgames You Must Know
    "UsqmCsgC", # Intermediate Endgames (Alternate)
    "UO2zqigQ", # Advanced Endgames You Must Know
    "1w0yWZvx", # Advanced Endgames (Part 2)
    "bnboDhFM", # Rook Endgames You Must Know
    "dYFcDtRq", # Pawns aren't people! (Pawn play)
    "h3ccaYFE", # Always sacrifice the exchange (Material imbalances)

    # --- STRATEGY & MIDDLEGAME (The "Why" of chess) ---
    "LAV8k5kM", # Morphy Simulator (Attacking principles)
    "kNn68T8l", # Bishops | Slice through the opposition
    "KSDGZjnf", # Knights | How to dominate your opponents
    "kI8ikTU4", # Knights (Alternate version)
    "T3ixjwmg", # Light and Dark Squares (Color complexes)
    "PLK3NjDd", # Light and Dark Squares (Alternate)
    "kjBSgqoA", # Talk to your pieces! (Developing plans)
    "xx2nOKJv", # Most Common Mistakes in Chess (Mr_Penings)
    "w2JcfP5K", # The Most Instructive Games of Chess Ever Played (Chernev)
    "YtBYXc3m", # Beautiful Checkmates (Pattern recognition)

    # --- OPENINGS (Solid & Instructional, not traps) ---
    "h4GuSZh3", # English Opening Repertoire (Mr_Penings)
    "8SEqMHi5", # English Opening (Alternate)
    "ZWHbJIPd", # Nimzo/Bogo Indian Repertoire
    "x75NW8ek", # The Basic Catalan (GM Craze)
    "vJsZScnC", # Italian Opening (LeninPerez)
    "BkaKI2VK", # Ruy Lopez (LeninPerez)
    "JM9AjCVp", # Sicilian Najdorf (LeninPerez)
    "SNOYkgFo", # King's Gambit (Aggressive but standard theory)
    "VSmBjmFj", # EPIC London System Games
    "bmWAylqe", # Repertoire for the D4 Player
    "sEo8o4Rm", # 1. e4 Nc6 (Nimzowitsch Defense - Solid offbeat)

    # --- MASTER GAMES & SPEEDRUNS (High level commentary) ---
    "s6JNESzr", # Naroditsky's Best Games
    "vEEsw5wy", # Daniel Naroditsky Speedrun Notes (Part 1)
    "mJnrrOEZ", # Sensei Speedrun Games
    "inBWS4oN", # Crushing the Englund Gambit (Refuting a trap, very instructional)
    "PhBeOxb9", # Only 6 Centipawn Loss (Perfect game analysis)
    "DXMZf4cl", # Ben Finegold's Favorite Game
    "84InnEl6", # NoseKnowsAll vs Famous People
]

OUTPUT_FILE = "chess_coaching_data.jsonl"
MIN_COMMENT_LENGTH = 20  # Filter out "lol", "blunder", etc.
# ---------------------

def clean_comment(comment):
    # Remove engine evals like [%eval 0.33] or [%clk 0:05:00]
    comment = re.sub(r"\[%.*?\]", "", comment)
    # Remove multiple spaces/newlines
    comment = " ".join(comment.split())
    return comment

def process_study(study_id):
    print(f"Downloading Study: {study_id}...")
    # Lichess API to export all chapters of a study as one big PGN
    url = f"https://lichess.org/study/{study_id}.pgn"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to download {study_id}")
        return []

    pgn_data = io.StringIO(response.text)
    data_pairs = []

    while True:
        game = chess.pgn.read_game(pgn_data)
        if game is None:
            break  # End of file

        board = game.board()
        
        # Iterate through every move in the game
        for node in game.mainline():
            comment = node.comment.strip()
            
            # THE FILTER: Only keep if it has a real text comment
            if comment:
                clean_text = clean_comment(comment)
                
                # Check if it's long enough to be useful
                if len(clean_text) > MIN_COMMENT_LENGTH:
                    entry = {
                        "input": board.fen(), # The Board State
                        "output": clean_text  # The Coach's explanation
                    }
                    data_pairs.append(entry)
            
            # Apply the move to update the board for the next loop
            board.push(node.move)
            
    return data_pairs

def main():
    all_data = []
    
    for study_id in STUDY_IDS:
        data = process_study(study_id)
        all_data.extend(data)
        print(f" -> Found {len(data)} valid training examples.")

    # Save to JSONL
    print(f"\nSaving {len(all_data)} total examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_data:
            json.dump(entry, f)
            f.write("\n")
    print("Done!")

if __name__ == "__main__":
    main()