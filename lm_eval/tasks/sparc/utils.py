"""SPaRC task utilities for lm-evaluation-harness.

The prompt builder and the path validator below are copied from the reference
implementation, because the harness cannot import the ``sparc-puzzle`` package:

* prompt:       https://github.com/lkaesberg/SPaRC/blob/6ecd5d9/sparc/prompts/single_shot.py
* paper prompt: https://github.com/lkaesberg/SPaRC/blob/6ecd5d9/sparc/prompts/single_shot_paper.py
* validator:    https://github.com/lkaesberg/SPaRC/blob/6ecd5d9/sparc/validation.py

The logic is unchanged; only type hints and formatting were adapted to the
harness's linters. The default prompt states the star, polyshape and negative
polyshape rules the way the puzzles were built. The paper's prompt words them
differently and is kept for ``sparc_paper``; it differs only in those three lines. A path counts as correct only if it exactly matches one of
the puzzle's stored solutions (``validate_solution``). ``analyze_path`` gives
the structural checks behind the paper's "invalid path" rate.
"""

import json
import re

import datasets


SYSTEM_PROMPT = "You are an expert at solving puzzles games."


# ---------------------------------------------------------------------------
# Prompt builder (copied from sparc/prompts/single_shot.py)
# ---------------------------------------------------------------------------


def _get_polyshapes_str(puzzle_data: dict) -> str:
    """Extract polyshapes string from puzzle data."""
    polyshapes_str = ""
    if puzzle_data.get("polyshapes"):
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = "\n".join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    return polyshapes_str


def get_prompt(puzzle_data: dict) -> dict:
    """
    Generate the complete prompt dict for single-shot puzzle solving.

    Args:
        puzzle_data: The puzzle data dictionary

    Returns:
        Dict with 'system' and 'user' message content
    """
    grid_size = puzzle_data.get("grid_size", {"width": 0, "height": 0})
    puzzle_array = puzzle_data.get("puzzle_array", [])
    grid_str = "\n".join(map(str, puzzle_array))
    start_pos = None
    end_pos = None
    for y, row in enumerate(puzzle_array):
        for x, cell in enumerate(row):
            if cell == "S":
                start_pos = f"({x}, {y})"
            elif cell == "E":
                end_pos = f"({x}, {y})"

    polyshapes_str = _get_polyshapes_str(puzzle_data)

    user_content = f"""
    ## Objective
    You are a specialized AI proficient in spatial reasoning and solving puzzles from the game 'The Witness'. Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    ## Core Concepts & Grid Basics
    *   **Grid Dimensions:** The puzzle grid has {grid_size["width"]} columns and {grid_size["height"]} rows.
    *   **Coordinate System:** Nodes are identified by `(x, y)` coordinates. `(0,0)` is the top-left node. `x` increases to the right, `y` increases downwards.
    *   **Path:** The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    *   **No Revisits:** The path **CANNOT** visit the same node more than once.
    *   **Valid Path Cells:** The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    *   **Rule Cells:** Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.
    *   **Regions:** The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.

    ## Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y

    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black

    ## Detailed Solving Rules
    The drawn path must satisfy **ALL** applicable constraints:

    1.  **Path Constraints:**
        *   Path **MUST** start at `S` and end at `E`.
        *   Path connects adjacent nodes (horizontal/vertical moves only).
        *   Nodes **CANNOT** be revisited.
        *   Path **MUST** pass through all Dot (`.`) cells.
        *   Path **CANNOT** pass through any Gap (`G`) cells.

    2.  **Region-Based Rules** (Apply to areas enclosed by the path):
        *   **Squares (`o-X`):** All squares within a single region **MUST** be the same color. Squares of different colors **MUST** be separated into different regions by the path.
        *   **Stars (`*-X`):** Within a single region, each star symbol **MUST** be paired with exactly **ONE** other symbol *of the same color*. Any rule symbol can be the partner: a star, square, triangle, polyshape or negative polyshape. Other colors within the region are irrelevant to this specific star's rule.
        *   **Polyshapes (`P-X-Y`):** The region containing this symbol **MUST** have exactly the specified shape (defined in Polyshape Definitions): the shape fills the whole region, with no cells left over. If multiple positive polyshapes are in one region, their shapes together must fill the region exactly, without overlapping. Rotation of polyshapes is NOT allowed. They must fit in their given orientation. The color of a polyshape does not matter for this rule.
        *   **Negative Polyshapes (`Y-X-Y`):** These subtract area from the positive polyshapes in the same region. The positive shapes, minus the cells covered by the negative shapes, must fill the region exactly, so the region's size is the total size of the positive shapes minus the total size of the negative shapes. A negative shape does not have to match the shape or color of any positive shape. If the negative shapes cancel the positive shapes completely, the region has no shape constraint.

    3.  **Path-Based Rules (Edge Touching):**
        *   **Triangles (`A-X`, `B-X`, `C-X`, `D-X`):** The path **MUST** touch a specific number of edges of the cell containing the triangle symbol.
            *   `A-X` (1): Path touches **EXACTLY 1** edge of the triangle's cell.
            *   `B-X` (2): Path touches **EXACTLY 2** edges of the triangle's cell.
            *   `C-X` (3): Path touches **EXACTLY 3** edges of the triangle's cell.
            *   `D-X` (4): Path touches **EXACTLY 4** edges (fully surrounds) the triangle's cell.

    ## EXAMPLE PUZZLE GRID:

    ["+",".","+","+","+","E","+"]
    ["+","C-R","+","o-K","+","o-K","+"]
    ["S","+","+","+","+","+","+"]
    ["+","P-G-112","+","*-G","+","P-B-624","+"]
    ["+","+","+","+","+","+","+"]
    ["+","*-G","+","*-G","+","o-K","+"]
    ["+","+","+",".","+","+","+"]

    EXAMPLE POLYSHAPE DEFINITIONS:
    Shape 112:
    [0,1,0,0]
    [0,1,0,0]
    [0,1,0,0]
    [0,0,0,0]

    Shape 624:
    [0,1,0,0]
    [0,1,1,0]
    [0,1,0,0]
    [0,0,0,0]

    EXAMPLE SOLUTION:

    We start at (0,2) and draw a line to (0,0).
    We then draw a line to (2,0) to reach the dot at (1,0) and surround the 3 count triangle.
    We then draw a line to (2,2) here we go down to touch the third side of the triangle cell and therefore validate the 3 count triangle.
    We continue down to (2,6) to validate the polyshape 112 and also the green star with the green polyshape
    After this we draw a line to (4,6) to start validating the polyshape 624 by surrounding it.
    Therefore we have to draw a line to (6,4) over (4,4) which creates a region for the stone at (5,5) which validates the stone.
    We continue up to (6,2) for the polyshape 624 and then go to (4,2) and after this to (4,0) to finaly validate the polyshape 624.
    This also validates the two green stars at (3,3) and (3,5) with each other and the black stone at (3,1) because its the only stone in its region.
    This line also creates a region for the black stone at (5,1) because its the only stone in its region.
    Now we can draw a line to (5,0) to reach the end node.

    #### (0,2),(0,1),(0,0),(1,0),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(3,6),(4,6),(4,5),(4,4),(5,4),(6,4),(6,3),(6,2),(5,2),(4,2),(4,1),(4,0),(5,0)

    ## Puzzle Input Data
    *   **Start Node:** {start_pos}
    *   **End Node:** {end_pos}
    *   **Grid Layout:**
        ```
        {grid_str}
        ```
    *   **Polyshape Definitions (if applicable):**
        *   Shapes are defined by 2D arrays where '1' indicates an occupied cell and '0' indicates an empty cell.
        ```
        {polyshapes_str}
        ```

    ## Task & Output Format
    1.  **Solve the Puzzle:** Determine the valid path from the Start Node to the End Node that satisfies all rules.
    2.  **Explain Reasoning:** Provide a step-by-step explanation of your thought process. Detail key deductions, how constraints were applied, and any backtracking or choices made.
    3.  **Provide Solution Path:** After the reasoning, output the exact marker string `####` followed immediately by the solution path as a list of node coordinates `(x, y)`. Include all intermediate nodes from start to end.

    **Example Solution Path Format:**
    ####
    (0,0), (1,0), (2,0), (2,1), ...
    """

    return {
        "system": "You are an expert at solving puzzles games.",
        "user": user_content,
    }


# ---------------------------------------------------------------------------
# Validator (copied from sparc/validation.py)
# ---------------------------------------------------------------------------


def extract_solution_path(
    solution_text: str, puzzle_data: dict | None = None
) -> list[dict[str, int]] | None:
    """Extract solution path from LLM's response

    Args:
        solution_text: Text response from the LLM
        puzzle_data: Optional puzzle data dict to extract end point from

    Returns:
        List of coordinate dicts or None if no path found
    """
    # First, check if "Solution" appears in the text
    solution_marker = "####"
    if solution_marker in solution_text:
        # Only process text after "####"
        solution_part = solution_text.split(solution_marker)[-1]
    else:
        # If no solution marker, use the full text
        solution_part = solution_text

    # Look for coordinate patterns like (0,0) -> (0,1) or similar
    # Pattern for (x,y) or (x, y) coordinates
    coord_pattern = r"\((\d+),\s*(\d+)\)"
    coords = re.findall(coord_pattern, solution_part)

    if coords:
        # Extract end point from puzzle data if provided
        end_point = None
        if puzzle_data:
            puzzle_array = puzzle_data.get("puzzle_array", [])
            for y, row in enumerate(puzzle_array):
                for x, cell in enumerate(row):
                    if cell == "E":
                        end_point = {"x": x, "y": y}
                        break
                if end_point:
                    break

        # Convert string coordinates to integer dicts
        path = []
        for x, y in coords:
            point = {"x": int(x), "y": int(y)}
            path.append(point)

            # Stop extracting if we've reached the end point
            if (
                end_point
                and point["x"] == end_point["x"]
                and point["y"] == end_point["y"]
            ):
                break

        return path

    # If no coordinates found, return None
    return None


def validate_solution(
    extracted_path: list[dict[str, int]] | None, puzzle_data: dict
) -> bool:
    """Validate if the solution is valid by comparing with known solutions

    Args:
        extracted_path: Optional list of coordinate dicts (can be None if no path extracted)
        puzzle_data: Dictionary containing puzzle data including solutions

    Returns:
        Boolean indicating if the solution is valid. Returns False if no valid path is provided.
    """
    # If no path was extracted, validation fails immediately.
    if not extracted_path:
        return False

    extracted_path = [(p["x"], p["y"]) for p in extracted_path]
    if len(extracted_path) < 2:
        return False

    # Check against all valid solutions in the database
    all_solutions = puzzle_data.get("solutions", [])
    if not all_solutions:
        return False

    # For each solution in the database, check if our path matches
    for solution in all_solutions:
        solution_path = [(p["x"], p["y"]) for p in solution["path"]]

        # Check if the paths match exactly
        if extracted_path == solution_path:
            return True

    return False


def analyze_path(solution_path: list[dict[str, int]] | None, puzzle: dict) -> dict:
    """Analyze the solution path for detailed validation metrics

    Args:
        solution_path: Optional list of coordinate dicts representing the solution path (can be None)
        puzzle: Puzzle dictionary with puzzle array and metadata

    Returns:
        Dictionary with detailed path analysis results
    """
    # Early return with all metrics False if no path was provided
    if not solution_path:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    solution_path = [(p["x"], p["y"]) for p in solution_path]

    # Get puzzle information
    puzzle_array = puzzle.get("puzzle_array", [])
    if not puzzle_array:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    # Find start and end points
    start_point = None
    end_point = None
    for y, row in enumerate(puzzle_array):
        for x, cell in enumerate(row):
            if cell == "S":
                start_point = (x, y)
            elif cell == "E":
                end_point = (x, y)

    if not start_point or not end_point:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    # Check if path starts at start and ends at exit
    starts_at_start = len(solution_path) > 0 and solution_path[0] == start_point
    ends_at_exit = len(solution_path) > 0 and solution_path[-1] == end_point
    starts_at_start_ends_at_exit = starts_at_start and ends_at_exit

    # Check if path is connected (no gaps)
    connected_line = True
    for i in range(1, len(solution_path)):
        prev_x, prev_y = solution_path[i - 1]
        curr_x, curr_y = solution_path[i]
        # Check if adjacent (Manhattan distance of 1)
        if abs(prev_x - curr_x) + abs(prev_y - curr_y) != 1:
            connected_line = False
            break

    # Check if path doesn't intersect with itself
    non_intersecting_line = len(set(solution_path)) == len(solution_path)

    # Check if there's a connected path from start to exit
    start_to_exit_connected = starts_at_start_ends_at_exit and connected_line

    # Identify rule cells as cells where both coordinates are odd
    rule_cells = set()
    for y in range(len(puzzle_array)):
        for x in range(len(puzzle_array[0]) if len(puzzle_array) > 0 else 0):
            if x % 2 == 1 and y % 2 == 1:
                rule_cells.add((x, y))

    # Check if path crosses rule cells
    no_rule_crossing = not any((x, y) in rule_cells for x, y in solution_path[1:-1])

    # Check if path is fully valid
    fully_valid_path = (
        starts_at_start_ends_at_exit
        and connected_line
        and non_intersecting_line
        and no_rule_crossing
    )

    return {
        "starts_at_start_ends_at_exit": starts_at_start_ends_at_exit,
        "connected_line": connected_line,
        "non_intersecting_line": non_intersecting_line,
        "start_to_exit_connected": start_to_exit_connected,
        "no_rule_crossing": no_rule_crossing,
        "fully_valid_path": fully_valid_path,
    }


# ---------------------------------------------------------------------------
# Harness hooks
# ---------------------------------------------------------------------------


# The paper's wording of the three rules, used by ``sparc_paper``.
PAPER_RULES = (
    (
        """*   **Stars (`*-X`):** Within a single region, each star symbol **MUST** be paired with exactly **ONE** other symbol *of the same color*. Any rule symbol can be the partner: a star, square, triangle, polyshape or negative polyshape. Other colors within the region are irrelevant to this specific star's rule.""",
        """*   **Stars (`*-X`):** Within a single region, each star symbol **MUST** be paired with exactly **ONE** other element (star or square) *of the same color*. Other colors within the region are irrelevant to this specific star's rule.""",
    ),
    (
        """*   **Polyshapes (`P-X-Y`):** The region containing this symbol **MUST** have exactly the specified shape (defined in Polyshape Definitions): the shape fills the whole region, with no cells left over. If multiple positive polyshapes are in one region, their shapes together must fill the region exactly, without overlapping. Rotation of polyshapes is NOT allowed. They must fit in their given orientation. The color of a polyshape does not matter for this rule.""",
        """*   **Polyshapes (`P-X-Y`):** The region containing this symbol **MUST** be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries. If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is NOT allowed. They must fit within the provided space in their given orientation.""",
    ),
    (
        """*   **Negative Polyshapes (`Y-X-Y`):** These subtract area from the positive polyshapes in the same region. The positive shapes, minus the cells covered by the negative shapes, must fill the region exactly, so the region's size is the total size of the positive shapes minus the total size of the negative shapes. A negative shape does not have to match the shape or color of any positive shape. If the negative shapes cancel the positive shapes completely, the region has no shape constraint.""",
        """*   **Negative Polyshapes (`Y-X-Y`):** These "subtract" shape requirements, typically within the same region as corresponding positive polyshapes. A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region. If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one. Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.""",
    ),
)


def get_paper_prompt(puzzle_data: dict) -> dict:
    """The prompt with the paper's wording of the star, polyshape and negative polyshape rules."""
    prompt = get_prompt(puzzle_data)
    user = prompt["user"]
    for new, old in PAPER_RULES:
        user = user.replace(new, old)
    return {"system": prompt["system"], "user": user}


def doc_to_text(doc: dict) -> str:
    """User message of the SPaRC single-shot prompt (system prompt is the YAML description)."""
    return get_prompt(doc)["user"]


def doc_to_text_paper(doc: dict) -> str:
    """User message of the paper's single-shot prompt, for ``sparc_paper``."""
    return get_paper_prompt(doc)["user"]


def doc_to_target(doc: dict) -> str:
    """First stored solution, formatted like the expected answer (reference only)."""
    path = doc["solutions"][0]["path"]
    return "#### " + ", ".join(f"({p['x']},{p['y']})" for p in path)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Interleave difficulty levels so ``--limit n`` covers all levels evenly.

    Puzzles keep their dataset order within each level; the output order is
    level 1, 2, 3, 4, 5, level 1, 2, ... until every level is exhausted.
    """
    by_level: dict[int, list[int]] = {}
    for idx, level in enumerate(dataset["difficulty_level"]):
        by_level.setdefault(level, []).append(idx)
    queues = [by_level[level] for level in sorted(by_level)]
    order = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    return dataset.select(order)


def process_hard(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep only difficulty-level-5 puzzles (the paper's "hard" subset)."""
    return dataset.filter(lambda doc: doc["difficulty_level"] == 5)


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Score one generation: exact-match accuracy and the paper's invalid-path flag."""
    response = results[0] or ""
    path = extract_solution_path(response, doc)
    solved = validate_solution(path, doc)
    analysis = analyze_path(path, doc)
    return {
        "acc": float(solved),
        "invalid_path": float(not analysis["fully_valid_path"]),
    }


def process_results_by_level(doc: dict, results: list[str]) -> dict[str, float]:
    """``process_results`` plus an ``acc_level_<k>`` entry for the puzzle's difficulty level."""
    out = process_results(doc, results)
    out[f"acc_level_{doc['difficulty_level']}"] = out["acc"]
    return out
