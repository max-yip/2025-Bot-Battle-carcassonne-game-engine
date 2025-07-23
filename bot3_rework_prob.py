from helper.game import Game
from lib.interact.tile import Tile
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.events.moves.typing import MoveType
from lib.interact.tile import TileModifier
from lib.models.tile_model import TileModel

from collections import defaultdict, deque, Counter
from lib.config.map_config import MAX_MAP_LENGTH, MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType
from itertools import combinations
from copy import deepcopy
from math import ceil


# State object to track bot's current context and last actions
class BotState:
    def __init__(self):
        # The last tile placed, used for meeple placement
        self.last_tile: TileModel | None = None

        # Whether the bot should place a meeple after placing a tile
        self.should_i_place_meeple = False

        # The feature (road/city/etc) the bot is interested in
        self.edge_for_placing_meeple = None

        #use to store any adjacent empty coords
        self.candidate_coords = set()

        self.valid_coords = defaultdict(list)
        # {tile A: [(rotation, (x,y)), (rotation, (x,y)), ...]

        self.placement_scores = defaultdict(list)
        # Each tile: (score, tile_model, should_place_meeple, meeple_edge)

        # temp data storage
        self.endgame_mode = False
        self.highest_scorer = None

        # map definitions
        self.CITY_1SIDE = {'D', 'E', 'H', 'I', 'J', 'K', 'L'}
        self.CITY_2SIDES = {'F', 'G', 'M', 'N', 'O', 'P'}
        self.CITY_3SIDES = {'Q', 'R', 'S', 'T'}
        self.CITY_4SIDES = {'C'}
        self.ROAD_START = {'A', 'L', 'S', 'T', 'W', 'X'}
        self.TURNING_RIVER = {"R3", "R5", "R7", "R9"}
    
    def update_endgame_mode(self, game: Game)->None:
        my_id = game.state.me.player_id
        max_points = 0

        for player_id, player in game.state.players.items():
            if player.points > max_points:
                max_points = player.points
                self.highest_scorer = player_id

        if max_points >= 40 or game.state.round >= 18:
            print(f"endgame_mode ON, highest_scorer = {self.highest_scorer}", flush = True)
            self.endgame_mode = True

    def get_score_params(self, game: Game) -> list:
        my_meeples = game.state.me.num_meeples

        """Scoring Parameters"""
        ROAD_SCORE = 1 #rather use open edges to determine completion probability
        CITY_SCORE = 1 #emblem will assume the city size = 2; for open city edge, find completion probability (max +0.3)
        CITY_COMPLETION_SCORE = 2 #for each tile
        MONASTERY_SCORE = 1
        SURROUNDING_MONASTERY = 1
        MEEPLE_COST = 1
        if my_meeples >= 5:
            MEEPLE_COST = 1
        elif self.endgame_mode:
            MEEPLE_COST = 1
        # elif my_meeples == 1:
        #     MEEPLE_COST = 4
        else:
            MEEPLE_COST = 5 - my_meeples
        
        return [ROAD_SCORE, CITY_SCORE, CITY_COMPLETION_SCORE, 
                MONASTERY_SCORE, SURROUNDING_MONASTERY, MEEPLE_COST]

    def reset_bot_state(self):
        self.last_tile = None
        self.should_i_place_meeple = False
        self.edge_for_placing_meeple = None
        self.candidate_coords.clear()
        self.valid_coords.clear()
        self.placement_scores.clear()
        self.endgame_mode = False
        self.highest_scorer = None

# Main loop for the bot: receives queries and responds with moves
def main():
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()

        # Decide what move to make based on the query type
        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    return handle_place_meeple(game, bot_state, q)

        # Send the chosen move to the game engine
        game.send_move(choose_move(query))


"""
Place Tile Function
"""
# Called from main function for placing tiles
def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    """
    STEP 1: SETUP - Get player info and tile hand (Completed)
    """
    # region Access all player models
    all_players = game.state.players  # Dict[int, PlayerModel]
    me = game.state.me
    my_player_id = me.player_id
    my_tiles = game.state.my_tiles
    my_meeples = me.num_meeples

    print(f"round: {game.state.round}", flush=True)
    print(f"I am player {my_player_id}", flush=True)
    for player_id, player in all_players.items():
        print(f"Player {player_id} has {player.points}", flush=True)
    print(f"my current score is {me.points}", flush=True)
    print(f"I have {my_meeples} meeples left.", flush=True)
    print(f"I have these tiles on my hand: {[tile.tile_type for tile in my_tiles]}", flush=True)

    bot_state.update_endgame_mode(game)
    # endregion

    """
    STEP 2: RIVER STAGE - find last tile and place with correct orientation (Completed)
    """
    # region Check if river stage
    if len(my_tiles) == 1 and len(my_tiles[0].tile_type) >1 and my_tiles[0].tile_type[0] == "R":
        tile_model, tile_index = river_tile_placement(game, bot_state, my_tiles)
        return game.move_place_tile(query, tile_model, tile_index)
    # endregion

    """
    STEP 3: ANALYZE BOARD STAGE - loop through all placed tiles and find valid spots 
    """
    get_all_valid_placements(game, bot_state)

    """
    STEP 4: Placement Value Estimator
    """
    placement_value_estimator(game, bot_state)

    """
    3d. return the tile placement with most score
    """
    move = select_best_tile_move(game, bot_state, query, my_tiles)
    if move:
        print("move successful", flush=True)
        return move

    raise ValueError("No valid tile placement found with any rotation.")



"""
Place Meeple Function
"""
# Called from main function for placing meeples
def handle_place_meeple(game: Game, bot_state: BotState, query: QueryPlaceMeeple) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """
    STAGE 5: PLACE MEEPLE
    place meeple based on the information from bot_state
    """
    assert bot_state.last_tile is not None
    curr_tile = bot_state.last_tile

    me = game.state.me
    my_meeples = me.num_meeples


    # make sure the edge is after rotation
    if bot_state.should_i_place_meeple and bot_state.edge_for_placing_meeple is not None and my_meeples > 0:
        print(f"placing meeple on {bot_state.edge_for_placing_meeple} on tile {curr_tile}", flush=True)
        # Place meeple on the given edge
        edge = bot_state.edge_for_placing_meeple
        bot_state.reset_bot_state()
        return game.move_place_meeple(query, curr_tile, placed_on=edge)
    
    else:
        print(f"should_i_place_meeple: {bot_state.should_i_place_meeple}, "
              f"edge_for_placing_meeple: {bot_state.edge_for_placing_meeple}, "
              f"curr_tile: {curr_tile}", flush=True)
        print("No meeple placement needed, passing.", flush=True)
        # Pass if no meeple placement is needed
        bot_state.reset_bot_state()
        return game.move_place_meeple_pass(query)



"""--------------------------------------------------Helper Functions------------------------------------------------"""
"""
General Helpers
"""
def my_can_place_tile_at(game: Game, tile: Tile, x: int, y: int) -> bool:
        if game.state.map._grid[y][x]:
            return False  # Already occupied

        directions = {
            (0, -1): "top_edge",
            (1, 0): "right_edge",
            (0, 1): "bottom_edge",
            (-1, 0): "left_edge",
        }

        edge_opposite = {
            "top_edge": "bottom_edge",
            "bottom_edge": "top_edge",
            "left_edge": "right_edge",
            "right_edge": "left_edge",
        }

        # print(f"can I place tile at {x, y}?", flush=True)
        has_any_neighbor = False

        for (dx, dy), edge in directions.items():
            nx, ny = x + dx, y + dy
            if not (0 <= ny < len(game.state.map._grid) and 0 <= nx < len(game.state.map._grid[0])):
                continue
            neighbor_tile = game.state.map._grid[ny][nx]
            if neighbor_tile is None:
                continue
            has_any_neighbor = True
            if not StructureType.is_compatible(
                    tile.internal_edges[edge],
                    neighbor_tile.internal_edges[edge_opposite[edge]],
                ):
                return False
        else:
            if has_any_neighbor:
                print(f"yes, tile {tile.tile_type} with rotation {tile.rotation}", flush=True)
                return True
        return False

def count_monastery_surrounding_tiles(game: Game, x, y): # assume x y is the coord of the monastery
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    deltas = [(-1, -1), (0, -1), (1, -1),
              (-1,  0),         (1,  0),
              (-1,  1), (0,  1), (1,  1)]

    count = 0
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= ny < height and 0 <= nx < width:
            if grid[ny][nx] is not None:
                count += 1

    return count

def surrounding_monastery(game: Game, x, y)->dict: # assume x y is the coord of the monastery
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    my_player_id = game.state.me.player_id

    deltas = [(-1, -1), (0, -1), (1, -1),
              (-1,  0),         (1,  0),
              (-1,  1), (0,  1), (1,  1)]

    monasteries = {}
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= ny < height and 0 <= nx < width:
            tile = grid[ny][nx]
            if tile and MONASTARY_IDENTIFIER in tile.internal_edges:
                claims = game.state._get_claims_objs(tile, MONASTARY_IDENTIFIER)
                if claims:
                    monasteries[tile] = claims

    return monasteries

def filter_candidate_placements(game: Game, bot_state: BotState) -> None:
    bot_state.valid_coords.clear()
    my_tiles = game.state.my_tiles

    for tile in my_tiles:
        for x, y in bot_state.candidate_coords:
            tile.placed_pos = (x, y)  # Set position for checking
            for r in range(4):
                if my_can_place_tile_at(game, tile, x, y):
                    bot_state.valid_coords[tile].append((r, (x, y)))
                    # print(f"Valid placement for {tile.tile_type} at ({x}, {y}) with rotation {r}", flush=True)
                tile.rotate_clockwise(1)  # Rotate for next check
            tile.placed_pos = None  # Reset position after checking all coords and rotations
            reset_tile_rotation(tile)  # Reset rotation to 0 for next tile

def get_all_valid_placements(game: Game, bot_state: BotState) -> bool:
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # top, right, bottom, left

    for tile in game.state.map.placed_tiles:
        x, y = tile.placed_pos

        for dir_idx, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] is None:
                bot_state.candidate_coords.add((nx, ny))
    
    filter_candidate_placements(game, bot_state)
    print(f"Valid placements: {bot_state.valid_coords}", flush=True)

def reset_tile_rotation(tile: Tile):
    """
    Reset the tile rotation to 0.
    """
    if tile.rotation != 0:
       tile.rotate_clockwise((4 - tile.rotation) % 4)

def get_coord_in_direction(pos: tuple[int, int], direction: str) -> tuple[int, int]:
    x, y = pos
    if direction == "top_edge":
        return (x, y - 1)
    elif direction == "bottom_edge":
        return (x, y + 1)
    elif direction == "left_edge":
        return (x - 1, y)
    elif direction == "right_edge":
        return (x + 1, y)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
def prob_get_all_tiles(probs: list[float], rounds: int) -> float:
    """
    Calculates the probability of getting at tiles that can complete the edges in the remaining rounds
    with given individual draw probabilities in a number of rounds.
    
    :param probs: List of required tile draw probabilities (e.g. [0.4, 0.3, 0.2])
    :param rounds: Number of total draws (e.g. 5)
    :return: Probability of drawing all tile types at least once
    """
    if not probs:
        return 1.0

    n = len(probs) 
    total_p = sum(probs)
    p_other = 1 - total_p
    total_prob = 0.0
    rounds = min(rounds, ceil(len(probs) * 1.5))

    # Inclusion-Exclusion Principle
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            p_miss = 1.0 - sum(probs[i] for i in combo)
            term = (p_miss) ** rounds
            if k % 2 == 1:
                total_prob += term
            else:
                total_prob -= term

    return 1 - total_prob

"""
River Stage Helper Functions
"""
#region
# Helper function to get placeable structures for a tile (before placing)
def get_placeable_structures_for_tile(tile: Tile) -> dict[str, StructureType]:
    """
    Returns all structures (edges and center) that could be claimed on this tile,
    even if it is not yet placed.
    """
    # Start with all internal edges (e.g., 'N', 'E', 'S', 'W')
    structures = dict(tile.internal_edges)  # edge: StructureType

    # Add monastery if the tile has the correct modifier
    if tile.tile_type == "R8":
        print("monastery in river")
        structures[MONASTARY_IDENTIFIER] = StructureType.MONASTARY

    # Filter only claimable structure types
    return structures

# Place meeple strat during river stage
def assess_meeple_river_stage(game: Game, bot_state: BotState, curr_tile, my_meeples):
    """
    Assess whether to place a meeple on the last placed tile in the river stage.
    """
    # since we only have 3 tiles, place meeple on at most two of them
    if my_meeples > 4:
    # Prioritize placing on city, then monastary, then roadstart (no road)
        structures = get_placeable_structures_for_tile(curr_tile)
        print(structures, flush = True)

        structPriority = [StructureType.CITY, StructureType.MONASTARY]

        for struct_type in structPriority:
            for edge, structure in structures.items():
                if structure == struct_type:
                    bot_state.edge_for_placing_meeple = edge
                    bot_state.should_i_place_meeple = True
                    return
    # If no suitable structure found, do not place a meeple
    bot_state.should_i_place_meeple = False
    bot_state.edge_for_placing_meeple = None

def get_river_edges(tile):
    return [e for e in Tile.get_edges() if tile.internal_edges[e] == StructureType.RIVER]

# Place river tile (alternate turns logic)
def river_tile_placement (game: Game, bot_state: BotState, my_tiles):
    """
    Find the open river edge and place the tile there, handling curved river alternation.
    """
    #map config
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    # dict of directions and edge names
    directions = {
            (0, -1): "top_edge",
            (1, 0): "right_edge",
            (0, 1): "bottom_edge",
            (-1, 0): "left_edge",
        }
    #only one tile in hand during river stage
    my_tile = my_tiles[0]

    # 1. Last placed river tile (it works so dont change)
    prev_tile = game.state.map.placed_tiles[-1]
    x, y = prev_tile.placed_pos
    prev_tile_on_map = game.state.map._grid[y][x]

    # 2. Find open river edge (only one)
    open_edge = None
    for (dx, dy), edge in directions.items():
        nx, ny = x + dx, y + dy

        if prev_tile_on_map.internal_edges[edge] == StructureType.RIVER:
            if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] is None:
                open_edge = (nx, ny, edge)
    

    print(open_edge, flush=True)

    
    # 3. Curved river tile logic
    curved_river_tiles = bot_state.TURNING_RIVER

    opposite_edge = {
        "top_edge": "bottom_edge",
        "bottom_edge": "top_edge",
        "left_edge": "right_edge",
        "right_edge": "left_edge",
    }

    # Default: Try all rotations at the open edge, but check river edge alignment
    nx, ny, open_edge_name = open_edge
    curr_tile_river_edge = opposite_edge[open_edge_name]

    if my_tile.tile_type in curved_river_tiles:

        print("I have a curved tile", flush=True)

        # Find the previous curved river tile (excluding the last placed, which is the current spot)
        last_turn_tile = None
        last_turn_river_edges = []

        for i in range(len(game.state.map.placed_tiles)-1, -1, -1):
            candidate = game.state.map.placed_tiles[i]

            if candidate.tile_type in curved_river_tiles:
                last_turn_tile = candidate
                for (dx, dy), edge in directions.items():
                    if last_turn_tile.internal_edges[edge] == StructureType.RIVER:
                        last_turn_river_edges.append(edge)

                print("last turn tile found", flush=True)
                break

        if last_turn_tile:
            for r in range(4):
                if (my_tile.internal_edges[curr_tile_river_edge] == StructureType.RIVER and
                    prev_tile.internal_edges[open_edge_name] == StructureType.RIVER):
                    curr_river_edges = get_river_edges(my_tile)
                    # Check if the tile is "flipped" (no shared edge direction)
                    flipped = not any(e in curr_river_edges for e in last_turn_river_edges)

                    if flipped:
                        """
                        Valid placement found
                        """
                        my_tile.placed_pos = (nx, ny)
                        assess_meeple_river_stage(game, bot_state, my_tile, game.state.me.num_meeples)
                        bot_state.last_tile = my_tile._to_model()
                        return my_tile._to_model(), 0
                    
                my_tile.rotate_clockwise(1)
            raise ValueError("No valid curved river tile placement found.")


    #default
    for r in range(4):
        if (my_tile.internal_edges[curr_tile_river_edge] == StructureType.RIVER and
            prev_tile.internal_edges[open_edge_name] == StructureType.RIVER):

            """
            Valid placement found
            """
            my_tile.placed_pos = (nx, ny)
            assess_meeple_river_stage(game, bot_state, my_tile, game.state.me.num_meeples)
            bot_state.last_tile = my_tile._to_model()
            return [my_tile._to_model(), 0]
        
        my_tile.rotate_clockwise(1)

    raise ValueError("No valid river tile placement found.")
#endregion

"""
Placement Value Estimator
"""
def placement_value_estimator(game: Game, bot_state: BotState):
    my_tiles = game.state.my_tiles
    for curr_tile, valid_placements in bot_state.valid_coords.items():
        curr_tile_index = my_tiles.index(curr_tile)

        for r, (x,y) in valid_placements:
            # reset rotation
            reset_tile_rotation(curr_tile)
            # assign rotation
            curr_tile.rotate_clockwise(r)
            curr_tile.placed_pos = (x, y)

            connected_structures = analyse_connected_structures(game, bot_state, curr_tile)
            # print(f"{connected_structures}", flush = True)

            score, place_meeple, meeple_edge = get_tile_placement_score(game, bot_state, connected_structures, curr_tile)
            # print(f"Score for {curr_tile.tile_type} at {x}{y} rot{r} = {score}", flush=True)

            bot_state.placement_scores[curr_tile_index].append(
                (score, curr_tile._to_model(), place_meeple, meeple_edge)
            )
            curr_tile.placed_pos = None
            reset_tile_rotation(curr_tile)


def analyse_connected_structures(game: Game, bot_state: BotState, curr_tile: Tile) -> dict:
    connected_structures = {}
    all_edges = [e for e in curr_tile.internal_edges.keys() if e != MONASTARY_IDENTIFIER]

    visited_edges = set()

    curr_tile_pos = curr_tile.placed_pos

    for edge in all_edges:
        if edge in visited_edges:
            continue
        
        if curr_tile.internal_edges[edge] == StructureType.GRASS:
            continue
        
        connected_tiles = list(game.state._traverse_connected_component(curr_tile, edge))
        print(connected_tiles, flush=True)

        #avoid adding up score for same structure (in no meeple placed)
        connected_edges = {e for t, e in connected_tiles if t == curr_tile}
        if connected_edges:
            visited_edges.update(connected_edges)

        unique_tiles = set()
        adj_empty_coords = []
        structure = curr_tile.internal_edges[edge]
        players = defaultdict(list)
        additional_size = 0 #for emblem cities

        for connected_tile, e in connected_tiles:
            unique_tiles.add(connected_tile)

            #find openings
            pos = connected_tile.placed_pos
            cell_next_to_edge = connected_tile.get_external_tile(e, pos, game.state.map._grid)
            adj_cell_coord = get_coord_in_direction(pos, e)

            if cell_next_to_edge is None and adj_cell_coord != curr_tile_pos:
                adj_empty_coords.append(adj_cell_coord)

            # Emblem bonus for cities
            if (connected_tile.modifiers and 
                connected_tile.internal_edges[e] == StructureType.CITY and
                TileModifier.EMBLEM in connected_tile.modifiers):
                additional_size+=1

            meeple = connected_tile.internal_claims[e]
            if meeple is not None:
                players[meeple.player_id].append(meeple)

        connected_structures[edge] = (len(unique_tiles)+additional_size, structure, players, adj_empty_coords)

    #monastery "edge"
    if TileModifier.MONASTARY in curr_tile.modifiers:
        x, y = curr_tile.placed_pos
        surrounding_size = count_monastery_surrounding_tiles(game, x, y)
        size = surrounding_size
        connected_structures[MONASTARY_IDENTIFIER] = (size, StructureType.MONASTARY, None, None)

    return connected_structures

#region
def get_tile_placement_score(game: Game, bot_state: BotState, connected_structures: dict, tile: Tile):
    # Initialize placement score and flags
    total_score = 0
    place_meeple = False
    meeple_edge = None
    my_player_id = game.state.me.player_id
    my_meeples = game.state.me.num_meeples

    meeple_edge_scores = defaultdict(int)

    # handle non-monastery stuff first
    for edge, info in connected_structures.items():
        size, structure, players, open_edge_coords = info


        remaining_rounds = 20-game.state.round
        # probability of completing each edge as a list
        prob_list = prob_of_completing_open_edges(game, bot_state, tile, info)
        # probability of completing all the open edges
        total_prob = prob_get_all_tiles(prob_list, remaining_rounds)
        print(f"rnds: {remaining_rounds}, prob: {prob_list}, total: {total_prob}", flush=True)


        if structure != StructureType.GRASS:
            if not players and my_meeples>0:
                # unoccupied
                place_meeple_score = meeple_edge_score(game, bot_state, tile, edge, info, total_prob)
                print(f"try placing meeple here would yield {place_meeple_score}", flush=True)
                meeple_edge_scores[edge] = place_meeple_score

            elif players: #occupied by someone, just check the edge
                total_score += no_meeple_edge_score(game, bot_state, tile, edge, info, total_prob)


    # handle monasteries
    total_score += surrounding_monasteries_score(game, bot_state, tile)

    if meeple_edge_scores:
        best_edge_to_place_meeple = max(meeple_edge_scores, key = lambda k: meeple_edge_scores[k])
        best_score_to_place_meeple = meeple_edge_scores[best_edge_to_place_meeple]

        if best_score_to_place_meeple > 0:
            total_score += best_score_to_place_meeple
            place_meeple = True
            meeple_edge = best_edge_to_place_meeple

    # Return the final score, meeple placement decision, and selected edge (if any)
    return total_score, place_meeple, meeple_edge

def meeple_edge_score(game: Game, bot_state: BotState, tile:Tile, edge, info, total_prob) -> float:
    params = bot_state.get_score_params(game)
    ROAD_SCORE = params[0]
    CITY_SCORE = params[1]
    CITY_COMPLETION_SCORE = params[2]
    MONASTERY_SCORE = params[3]
    SURROUNDING_MONASTERY = params[4]
    MEEPLE_COST = params[5]

    size, structure, players, open_edge_coords = info
    will_be_completed = False
    if not open_edge_coords:
        will_be_completed = True
    score = MEEPLE_COST * (-1)

    if structure == StructureType.MONASTARY:
        score += size*SURROUNDING_MONASTERY + MONASTERY_SCORE
        if will_be_completed:
            score += MEEPLE_COST
        return score

    if structure == StructureType.CITY:
        if will_be_completed:
            score += size * CITY_COMPLETION_SCORE + MEEPLE_COST
        else:
            score += size * CITY_SCORE
            score += (MEEPLE_COST+size) * total_prob
        return score

    elif structure == StructureType.ROAD or structure == StructureType.ROAD_START:
        score += size * ROAD_SCORE

        if will_be_completed:
            score += MEEPLE_COST
        else:
            score += MEEPLE_COST * total_prob
        return score
    
    return score

def prob_of_completing_open_edges(game: Game, bot_state: BotState, tile: Tile, info)->list:
    size, structure, players, open_edge_coords = info
    prob_list = []
    # find how many tiles can place next to the edge of the open edges
    if not open_edge_coords:
        return

    coords_count = Counter(open_edge_coords)

    for coord, count in coords_count.items():
        print(f"check here {coord}", flush=True)
        p_complete = p_complete_this_edge(game, bot_state, structure, coord, count, tile)
        prob_list.append(p_complete)
        
    return prob_list

def p_complete_this_edge (game: Game, bot_state: BotState, structure, coord, count, curr_tile: Tile) -> float:
    important_tiles = set()
    remaining_rounds = 20-game.state.round
    if structure == StructureType.CITY:
        if count == 1:
            important_tiles = bot_state.CITY_1SIDE
        elif count == 2:
            important_tiles = bot_state.CITY_2SIDES
        elif count == 3:
            important_tiles = bot_state.CITY_3SIDES
        elif count == 4:
            important_tiles = bot_state.CITY_4SIDES
    elif structure == StructureType.ROAD or structure == StructureType.ROAD_START:
        important_tiles = bot_state.ROAD_START
    else:
        return 0.0

    x, y = coord

    #check my hand
    my_tiles = game.state.my_tiles
    for tile in my_tiles:

        if tile == curr_tile:
            continue
        
        temp_tile = deepcopy(tile)
        if game.can_place_tile_at(temp_tile, x, y) and tile.tile_type in important_tiles:
            return 1.0
            

    # check all available_tiles
    available_tiles_by_type = game.state.map.available_tiles_by_type
    total_remaining = len(game.state.map.available_tiles)
    matching = 0
    
    for tile_type, tile_list in available_tiles_by_type.items():
        if len(tile_list) == 0:
            continue
        if tile_type in important_tiles:
            # matching += len(tile_list)
            temp_tile = deepcopy(tile_list[0])
            if game.can_place_tile_at(temp_tile, x, y):
                matching += len(tile_list)

    if total_remaining == 0:
        return 0.0

    #hopeful factor (hope that this edge can be completed)
    # if matching == 0:
    #     if total_remaining > 40:
    #         return 0.015
    #     elif total_remaining > 20:
    #         return 0.008
    #     else:
    #         return 0.0
    
    print(f"matching {matching}/{total_remaining}", flush=True)
    return matching/total_remaining

def no_meeple_edge_score(game: Game, bot_state: BotState, tile:Tile, edge, info, total_prob) -> float:
    size, structure, players, open_edge_coords = info
    params = bot_state.get_score_params(game)
    ROAD_SCORE = params[0]
    CITY_SCORE = params[1]
    CITY_COMPLETION_SCORE = params[2]
    MONASTERY_SCORE = params[3]
    SURROUNDING_MONASTERY = params[4]
    MEEPLE_COST = params[5]

    size, structure, players, open_edge_coords = info
    will_be_completed = False
    if not open_edge_coords:
        will_be_completed = True

    my_player_id = game.state.me.player_id
    my_meeples = 0
    enemies_meeples = 0
    
    for pid, meeples in players.items():
        if pid == my_player_id:
            my_meeples = len(meeples)
        else:
            enemies_meeples += len(meeples)

    score = 0
    if structure == StructureType.CITY:
        if TileModifier.EMBLEM in tile.modifiers:
            CITY_SCORE += 1
        if will_be_completed:
            if my_meeples > 0:
                score += (size * CITY_COMPLETION_SCORE + MEEPLE_COST*my_meeples)
            elif enemies_meeples > 0:
                score -= (size * CITY_COMPLETION_SCORE + MEEPLE_COST*enemies_meeples)
        else:
            if my_meeples > 0:
                score += (CITY_SCORE + MEEPLE_COST*my_meeples*total_prob)
            elif enemies_meeples > 0:
                score -= (CITY_SCORE + MEEPLE_COST*enemies_meeples*total_prob)
        return score

    elif structure == StructureType.ROAD or structure == StructureType.ROAD_START:
        if will_be_completed:
            if my_meeples > 0:
                score += (size * ROAD_SCORE + MEEPLE_COST*my_meeples)
            elif enemies_meeples > 0:
                score -= (size * ROAD_SCORE + MEEPLE_COST*enemies_meeples)
        else:
            if my_meeples > 0:
                score += (ROAD_SCORE + MEEPLE_COST*my_meeples*total_prob)
            elif enemies_meeples > 0:
                score -= (ROAD_SCORE + MEEPLE_COST*enemies_meeples*total_prob)
        return score
    
    return score

def surrounding_monasteries_score(game: Game, bot_state: BotState, tile: Tile) -> float:
    x, y = tile.placed_pos
    monasteries = surrounding_monastery(game, x, y)  # expected to return dict[Tile, dict[int, list[Meeple]]]
    my_player_id = game.state.me.player_id
    score = 0.0

    for surrounding_tile, claims in monasteries.items():
        nx, ny = surrounding_tile.placed_pos
        surround = count_monastery_surrounding_tiles(game, nx, ny)
        if claims:
            if my_player_id in claims:
                score += surround + 1/8
            else:
                score -= surround + 1/8

    return score

def select_best_tile_move(game, bot_state, query, my_tiles):
    best_entry = None
    max_score = float('-inf')
    best_tile_index = None
    best_rotation = None
    for tile_index, score_data_list in bot_state.placement_scores.items():
        for score, t, m, e in score_data_list:
            # t is a TileModel, which has .rotation
            if best_entry is None or score > max_score:
                max_score = score
                best_tile_index = tile_index
                best_entry = (t, m, e)
                best_rotation = t.rotation

    if best_tile_index is None:
        raise ValueError("bug in select_best_tile_move.")

    tile_model, should_place_meeple, meeple_edge  = best_entry
    print(tile_model, flush=True)
    bot_state.last_tile = tile_model
    bot_state.should_i_place_meeple = should_place_meeple
    bot_state.edge_for_placing_meeple = meeple_edge

    # --- Important: Ensure the tile in my_tiles is rotated to match tile_model.rotation ---
    tile = my_tiles[best_tile_index]
    reset_tile_rotation(tile)
    tile.rotate_clockwise(tile_model.rotation)

    return game.move_place_tile(query, tile_model, best_tile_index)


if __name__ == "__main__":
    main()