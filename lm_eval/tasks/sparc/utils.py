import re
import json
import datasets
from typing import List, Dict, Any, Tuple, Optional


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process the SPaRC dataset documents."""
    
    def _process_doc(doc):
        # Extract and clean the grid visualization if present
        if 'text_visualization' in doc:
            doc['visualization'] = doc['text_visualization']
        
        # Ensure solutions is a list
        if 'solutions' in doc and isinstance(doc['solutions'], str):
            try:
                doc['solutions'] = json.loads(doc['solutions'])
            except json.JSONDecodeError:
                doc['solutions'] = [doc['solutions']]
        elif 'solutions' in doc and not isinstance(doc['solutions'], list):
            doc['solutions'] = [doc['solutions']]
        
        # Convert polyshapes if it's a string
        if 'polyshapes' in doc and isinstance(doc['polyshapes'], str):
            try:
                doc['polyshapes'] = json.loads(doc['polyshapes'])
            except json.JSONDecodeError:
                doc['polyshapes'] = {}
        
        # Extract grid size
        if 'grid_size' in doc:
            if isinstance(doc['grid_size'], dict):
                doc['grid_height'] = doc['grid_size'].get('height', 10)
                doc['grid_width'] = doc['grid_size'].get('width', 10)
            else:
                doc['grid_height'] = 10
                doc['grid_width'] = 10
        
        return doc
    
    return dataset.map(_process_doc)


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Convert a document to input text for the model using comprehensive prompt format."""
    
    grid_size = doc.get("grid_size", {"width": 0, "height": 0})
    puzzle_array = doc.get("puzzle_array", [])
    grid_str = "\n".join(map(str, puzzle_array))
    start_pos = None
    end_pos = None
    for y, row in enumerate(puzzle_array):
        for x, cell in enumerate(row):
            if cell == "S":
                start_pos = f"({x}, {y})"
            elif cell == "E":
                end_pos = f"({x}, {y})"

    polyshapes_str = ""
    if "polyshapes" in doc and doc["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_data = doc["polyshapes"]
        if isinstance(polyshapes_data, str):
            try:
                polyshapes_json = json.loads(polyshapes_data)
            except json.JSONDecodeError:
                polyshapes_json = {}
        else:
            polyshapes_json = polyshapes_data
            
        for shape_id, shape_def in polyshapes_json.items():
            polyshapes_str += f"Shape {shape_id}:\n"
            if isinstance(shape_def, list):
                polyshapes_str += '\n'.join(str(row) for row in shape_def)
            else:
                polyshapes_str += str(shape_def)
            polyshapes_str += "\n\n"
    
    return f"""
## Objective
You are a specialized AI proficient in spatial reasoning and solving puzzles from the game 'The Witness'. Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

## Core Concepts & Grid Basics
*   **Grid Dimensions:** The puzzle grid has {grid_size['width']} columns and {grid_size['height']} rows.
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
    *   **Stars (`*-X`):** Within a single region, each star symbol **MUST** be paired with exactly **ONE** other element (star or square) *of the same color*. Other colors within the region are irrelevant to this specific star's rule.
    *   **Polyshapes (`P-X-Y`):** The region containing this symbol **MUST** be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries. If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is generally allowed unless context implies otherwise.
    *   **Negative Polyshapes (`Y-X-Y`):** These "subtract" shape requirements, typically within the same region as corresponding positive polyshapes. A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region. If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one. Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

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
[(0, 0), (1, 0), (2, 0), (2, 1), ...]
"""


def doc_to_target(doc: Dict[str, Any]) -> str:
    """Extract the target solution and puzzle data for validation."""
    # Create a comprehensive target that includes both the solution and puzzle data
    target_data = {
        "solutions": doc.get('solutions', []),
        "puzzle_array": doc.get('puzzle_array', []),
        "grid_size": doc.get('grid_size', {}),
        "polyshapes": doc.get('polyshapes', {}),
        # Include difficulty information - try multiple possible field names
        "difficulty": doc.get('difficulty_level', "unknown")
    }
    
    # Return as JSON string so metrics can parse it
    return json.dumps(target_data)


def solve_rate_by_difficulty(predictions, references=None, **kwargs):
    """Metric that tracks solve rates broken down by difficulty level."""
    if not predictions or not references:
        return {"overall_solve_rate": 0.0}
    
    # Initialize tracking dictionaries
    difficulty_stats = {}
    total_count = len(predictions)
    total_solved = 0
    
    for pred, ref in zip(predictions, references):
        # Handle list responses
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        elif not isinstance(pred, str):
            pred = str(pred)
        
        # Extract path from prediction
        path = extract_solution_path(pred)
        
        # Parse puzzle data from reference
        try:
            puzzle_data = json.loads(ref) if isinstance(ref, str) else ref
            
            # Get difficulty information
            difficulty = puzzle_data.get('difficulty')
            if difficulty is None:
                difficulty = "unknown"
            else:
                difficulty = str(difficulty)
            
            # Initialize difficulty tracking if not seen before
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {
                    "total": 0,
                    "solved": 0
                }
            
            difficulty_stats[difficulty]["total"] += 1
            
            # Check if this puzzle was solved correctly
            is_solved = False
            if path:
                is_solved = validate_solution(path, puzzle_data)
                if is_solved:
                    difficulty_stats[difficulty]["solved"] += 1
                    total_solved += 1
            
        except (json.JSONDecodeError, TypeError):
            # Handle parsing errors by tracking as unknown difficulty
            if "unknown" not in difficulty_stats:
                difficulty_stats["unknown"] = {"total": 0, "solved": 0}
            difficulty_stats["unknown"]["total"] += 1
    
    # Calculate solve rates for each difficulty (only percentages)
    result = {
        "overall_solve_rate": total_solved / total_count if total_count > 0 else 0.0
    }
    
    for difficulty, stats in difficulty_stats.items():
        solve_rate = stats["solved"] / stats["total"] if stats["total"] > 0 else 0.0
        result[f"solve_rate_difficulty_{difficulty}"] = solve_rate
    
    return result


def aggregate_difficulty_analysis(items, **kwargs):
    """Aggregate solve rate by difficulty analysis across multiple items."""
    if not items:
        return {"overall_solve_rate": 0.0}
    
    # The items should be the results from solve_rate_by_difficulty
    # Just take the first item since it already contains aggregated results
    if isinstance(items, list) and len(items) > 0:
        return items[0]
    else:
        return items


def extract_solution_path(
    solution_text: str, puzzle_data: Dict = None
) -> Optional[List[Dict[str, int]]]:
    """Extract solution path from LLM's response

    Args:
        solution_text: Text response from the LLM
        puzzle_data: Optional puzzle data dict to extract end point from

    Returns:
        List of coordinate dicts or None if no path found
    """
    print("--------------------------------")
    print(solution_text)
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


def validate_solution(extracted_path: Optional[List[Dict[str, int]]], puzzle_data: Dict) -> bool:
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


def analyze_path(solution_path: Optional[List[Dict[str, int]]], puzzle: Dict) -> Dict:
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


def contains_coordinates(predictions, references=None, **kwargs):
    """Metric that checks if model output contains coordinate patterns."""
    if not predictions:
        return 0.0
    
    coord_count = 0
    total_count = len(predictions)
    
    for pred in predictions:
        # Handle list responses
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        elif not isinstance(pred, str):
            pred = str(pred)
        
        # Check if prediction contains coordinate patterns
        coord_pattern = r"\((\d+),\s*(\d+)\)"
        match = re.search(coord_pattern, pred)
        
        if match:
            coord_count += 1
    
    final_score = coord_count / total_count if total_count > 0 else 0.0
    return final_score


def path_validity_score(predictions, references=None, **kwargs):
    """Metric that validates extracted paths against known solutions."""
    if not predictions or not references:
        return 0.0
    
    valid_count = 0
    total_count = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Handle list responses
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        elif not isinstance(pred, str):
            pred = str(pred)
        
        # Extract path from prediction
        path = extract_solution_path(pred)
        
        # Parse puzzle data from reference
        try:
            puzzle_data = json.loads(ref) if isinstance(ref, str) else ref
            
            if path:
                is_valid = validate_solution(path, puzzle_data)
                if is_valid:
                    valid_count += 1
            
        except (json.JSONDecodeError, TypeError):
            # Give partial credit for extracting a path with multiple coordinates
            if path and len(path) > 1:
                valid_count += 0.5
    
    final_score = valid_count / total_count if total_count > 0 else 0.0
    return final_score


def spatial_reasoning_analysis(predictions, references=None, **kwargs):
    """Detailed spatial reasoning analysis metric."""
    if not predictions or not references:
        return {
            "starts_at_start_ends_at_exit": 0.0,
            "connected_line": 0.0,
            "non_intersecting_line": 0.0,
            "start_to_exit_connected": 0.0,
            "no_rule_crossing": 0.0,
            "fully_valid_path": 0.0,
            "path_extraction_rate": 0.0,
        }
    
    total_count = len(predictions)
    
    # Initialize metric counters
    metrics = {
        "starts_at_start_ends_at_exit": 0,
        "connected_line": 0,
        "non_intersecting_line": 0,
        "start_to_exit_connected": 0,
        "no_rule_crossing": 0,
        "fully_valid_path": 0,
        "path_extraction_rate": 0,
    }
    
    for pred, ref in zip(predictions, references):
        # Handle list responses
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        elif not isinstance(pred, str):
            pred = str(pred)
        
        # Extract path from prediction
        path = extract_solution_path(pred)
        
        if path:
            metrics["path_extraction_rate"] += 1
            
            # Parse puzzle data from reference for detailed analysis
            try:
                puzzle_data = json.loads(ref) if isinstance(ref, str) else ref
                
                analysis = analyze_path(path, puzzle_data)
                
                # Update counters based on analysis
                for key in ["starts_at_start_ends_at_exit", "connected_line", 
                           "non_intersecting_line", "start_to_exit_connected", 
                           "no_rule_crossing", "fully_valid_path"]:
                    if analysis.get(key, False):
                        metrics[key] += 1
                        
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Convert counts to ratios
    final_metrics = {key: count / total_count for key, count in metrics.items()}
    return final_metrics


def aggregate_spatial_analysis(items, **kwargs):
    """Aggregate spatial reasoning analysis across multiple items."""
    if not items:
        return {
            "starts_at_start_ends_at_exit": 0.0,
            "connected_line": 0.0,
            "non_intersecting_line": 0.0,
            "start_to_exit_connected": 0.0,
            "no_rule_crossing": 0.0,
            "fully_valid_path": 0.0,
            "path_extraction_rate": 0.0,
        }
    
    # The items should be the results from spatial_reasoning_analysis
    # Just take the first item since it already contains aggregated results
    if isinstance(items, list) and len(items) > 0:
        return items[0]
    else:
        return items 