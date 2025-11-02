import re
import random
import ast
import operator


def doc_to_text(doc):
    """Format the prompt for countdown task.
    
    Args:
        doc: The document dictionary containing 'target' and 'numbers' (or 'nums')
    
    Returns:
        Formatted prompt string
    """
    # Handle both 'numbers' and 'nums' field names
    numbers = doc.get('numbers', doc.get('nums', []))
    target = doc.get('target', doc.get('target_number'))
    
    # Format numbers as comma-separated string
    numbers_str = ', '.join(str(n) for n in numbers)
    
    return f"Given the numbers {numbers_str}, how can you use each number exactly once with basic arithmetic operations (+, -, *, /) to reach the target {target}?\n\nAnswer:"


def extract_solution(solution_str):
    """Extract the equation from the solution string.
    
    This function tries multiple strategies to extract an arithmetic equation:
    1. Look for <answer>...</answer> tags
    2. Extract from assistant responses
    3. Find equation patterns in lines
    4. Extract from the whole string as fallback
    """
    if not solution_str or not solution_str.strip():
        return None
    
    # Try to find answer in <answer> tags first
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE))
    if matches:
        content = matches[-1].group(1).strip()
        # Clean up the extracted content
        # Remove any trailing "= target" or similar
        content = re.sub(r'\s*=\s*\d+\s*$', '', content)
        if content and re.search(r'[\+\-\*/]', content) and re.search(r'\d', content):
            return content
    
    # Remove everything before the first "Assistant:" if present
    working_str = solution_str
    if "Assistant:" in working_str:
        working_str = working_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in working_str.lower():
        working_str = working_str.split("<|im_start|>assistant", 1)[1]
    
    # Try to extract equation-like pattern from lines (check from last line backwards)
    lines = working_str.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        
        # Remove common prefixes
        line = re.sub(r'^(Answer|Equation|Solution|Result):\s*', '', line, flags=re.IGNORECASE)
        
        # Clean up: remove trailing "= number" patterns (the result)
        line_clean = re.sub(r'\s*=\s*\d+\.?\d*\s*$', '', line)
        
        # Look for equation patterns: numbers, operators, parentheses
        # Must have at least one operator and one number
        if re.search(r'[\+\-\*/]', line_clean) and re.search(r'\d', line_clean):
            # Extract continuous equation-like sequences
            equation_match = re.search(r'([\d+\-*/().\s]+)', line_clean)
            if equation_match:
                potential_eq = equation_match.group(1).strip()
                # Verify it's a valid-looking equation
                if (re.search(r'[\+\-\*/]', potential_eq) and 
                    re.search(r'\d', potential_eq) and
                    len(potential_eq) >= 3):
                    return potential_eq
    
    # If no structured format found, try to extract from the whole string
    # Look for the longest sequence that matches equation pattern
    equation_match = re.search(r'([\d+\-*/().\s]{5,})', working_str)
    if equation_match:
        potential_eq = equation_match.group(1).strip()
        # Clean it up
        potential_eq = re.sub(r'\s*=\s*\d+\.?\d*\s*$', '', potential_eq)
        if (re.search(r'[\+\-\*/]', potential_eq) and 
            re.search(r'\d', potential_eq) and
            len(potential_eq) >= 3):
            return potential_eq
    
    return None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once.
    
    Note: This checks that the numbers used in the equation match the available numbers,
    but allows for flexibility in case not all numbers need to be used (partial solutions).
    """
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available and used at most once
        available_sorted = sorted(available_numbers)
        numbers_sorted = sorted(numbers_in_eq)
        
        # Each number in the equation must be from available numbers
        # and each should be used exactly once
        return numbers_sorted == available_sorted
    except Exception:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions.
    
    Args:
        equation_str: String containing arithmetic expression
        
    Returns:
        Numeric result of the equation, or None if evaluation fails
    """
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            return None

        # Check for division by zero patterns (simple check)
        # More thorough checking happens during evaluation
        if '/0' in equation_str.replace(' ', ''):
            return None

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        
        # Check if result is a valid number
        if not isinstance(result, (int, float)):
            return None
            
        # Check for inf or nan
        if isinstance(result, float) and (result != result or result == float('inf') or result == float('-inf')):
            return None
            
        return result
    except Exception:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def process_results(doc, results):
    """Process results for the countdown task.
    
    Args:
        doc: The document dictionary containing 'target' and 'numbers' (or 'nums')
        results: List of generated strings (one per repeat)
    
    Returns:
        Dictionary with metric scores
    """
    # Get the first (or only) result
    solution_str = results[0] if results else ""
    
    # Prepare ground truth - handle both 'numbers' and 'nums' field names
    numbers = doc.get('numbers', doc.get('nums', []))
    target = doc.get('target', doc.get('target_number'))
    
    ground_truth = {
        'target': target,
        'numbers': numbers
    }
    
    # Compute score
    score = compute_score(solution_str, ground_truth)
    
    return {
        'countdown_score': score
    }