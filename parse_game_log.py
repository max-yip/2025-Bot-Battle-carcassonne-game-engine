import json

# Load the JSON file
with open("/Users/maxyi/Desktop/public-carcassonne-game-engine/output/game.json", "r") as f:
    events = json.load(f)

# Helper function to pretty-print an event
def print_event(event):
    event_type = event.get("event_type", "Unknown")
    print(f"\n=== {event_type.upper()} ===")

    if event_type == "event_game_started":
        print(f"Turn Order: {event['turn_order']}")
        for p in event['players']:
            print(f"Player {p['player_id']} (Team {p['team_id']}): {p['points']} points, {p['num_meeples']} meeples")
    elif event_type == "event_starting_tile_placed":
        tile = event['tile_placed']
        print(f"Starting Tile: {tile['tile_type']} at {tile['pos']} (Rotation: {tile['rotation']})")
    elif event_type == "event_player_drew_tiles":
        print(f"Player {event['player_id']} drew {event['num_tiles']} tile(s):")
        for tile in event['tiles']:
            print(f"  - {tile['tile_type']} at {tile['pos']} (Rotation: {tile['rotation']})")
    elif event_type == "event_player_banned":
        print(f"Player {event['player_id']} was banned due to {event['ban_type']}")
        print(f"Reason: {event['reason']}")
        if 'details' in event:
            print(f"Details: {event['details'].get('query_type', '')}")
    else:
        print(json.dumps(event, indent=2))  # Fallback for unknown types

# Print all events
for event in events:
    print_event(event)
