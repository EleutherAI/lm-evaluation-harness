"""
Extraction functions for MolecularIQ chemistry tasks.
Unified extraction using <answer></answer> tags.
"""

import ast
import json
import re


def extract_moleculariq_answer(response: str) -> str | int | float | dict | list | None:
    """
    Unified extraction function for MolecularIQ.

    Handles multiple common formats in priority order:
    1. Strip thinking blocks (</think>)
    2. <answer>...</answer> tags (standard format)
    3. JSON objects/arrays via extract_answer_only logic
    4. Fallback to general extraction patterns

    Args:
        response: Raw model response string

    Returns:
        Extracted answer (string, number, dict, or list), or None if extraction failed
    """
    if not response or not isinstance(response, str):
        return None

    # Step 1: Strip thinking blocks to get the answer portion
    content = response.strip()
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    # Step 2: Check for <answer></answer> tags (standard format)
    if "<answer>" in content and "</answer>" in content:
        answer = content.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer

    # Step 3: Fall back to JSON/dict detection via extract_answer_only
    json_answer = extract_answer_only(content)
    if json_answer is not None:
        return json_answer

    # Step 4: Fall back to general extraction patterns (boxed, bold, etc.)
    return extract_general_answer(response)


def extract_general_answer(response):
    """
    Simplified answer extraction for chemistry CoT traces and cleanup.
    """
    # Handle thinking mode
    if "</think>" in response:
        content = response.split("</think>")[-1].strip()
    else:
        content = response.strip()

    answer = None

    if "<answer>" in content and "</answer>" in content:
        answer = content.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        # Check for boxed or bold answers
        answer = extract_last_formatted_answer(content)

    if answer is not None:
        cleaned = clean_extracted_answer(answer)
        if cleaned is not None:
            return convert_to_appropriate_type(cleaned)

    # Fallback: try other patterns
    return extract_fallback_patterns(content)


def extract_last_formatted_answer(content):
    r"""
    Extract the last formatted answer from the content.
    Looks for **content**, \\boxed{content}, and (content) patterns.
    """
    all_matches = []

    # Find positions of bold matches **content**
    for match in re.finditer(r"\*\*(.+?)\*\*", content):
        all_matches.append((match.start(), match.group(1)))

    # Find positions of boxed matches \\boxed{content}
    pattern = r"\\boxed\{"
    for match in re.finditer(pattern, content):
        start = match.end()
        brace_count = 1
        end = start

        while end < len(content) and brace_count > 0:
            if content[end] == "{":
                brace_count += 1
            elif content[end] == "}":
                brace_count -= 1
            end += 1

        if brace_count == 0:
            extracted = content[start : end - 1]
            all_matches.append((match.start(), extracted))

    if all_matches:
        all_matches.sort(key=lambda x: x[0])
        last_answer = all_matches[-1][1].strip()
        return last_answer

    return None


def extract_fallback_patterns(content):
    """Fallback extraction methods."""
    # Content in quotes
    quoted = re.findall(r'"([^"]+)"', content)
    if quoted:
        return clean_extracted_answer(quoted[-1])

    single_quoted = re.findall(r"'([^']+)'", content)
    if single_quoted:
        return clean_extracted_answer(single_quoted[-1])

    # Numbers with units
    numbers_with_units = re.findall(r"([0-9.]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)", content)
    if numbers_with_units:
        return f"{numbers_with_units[-1][0]} {numbers_with_units[-1][1]}"

    # Just numbers
    numbers = re.findall(r"([0-9.]+(?:\.[0-9]+)?)", content)
    if numbers:
        return numbers[-1]

    return None


def remove_latex_commands(text):
    r"""Remove LaTeX commands like \\text{}, \\mathrm{}, \\mathbf{}."""
    if not text:
        return text

    latex_commands = ["text", "mathrm", "mathbf"]
    result = text

    for cmd in latex_commands:
        pattern = rf"\\{cmd}\{{"

        while True:
            match = re.search(pattern, result)
            if not match:
                break

            start = match.end()
            brace_count = 1
            end = start

            while end < len(result) and brace_count > 0:
                if result[end] == "{":
                    brace_count += 1
                elif result[end] == "}":
                    brace_count -= 1
                end += 1

            if brace_count == 0:
                before = result[: match.start()]
                content = result[start : end - 1]
                after = result[end:]
                result = before + content + after
            else:
                break

    return result


def clean_extracted_answer(answer):
    """Clean extracted answers."""
    if not answer:
        return None

    answer = answer.strip()

    # Remove common artifacts
    answer = re.sub(r"^\$+|\$+$", "", answer)
    answer = re.sub(r"^\[|\]$", "", answer)
    answer = re.sub(r"[.,;:]+$", "", answer)

    if not answer.startswith(("\\text{", "\\mathrm{", "\\mathbf{")):
        answer = re.sub(r"^\{|\}$", "", answer)

    answer = remove_latex_commands(answer)
    answer = normalize_sub_super_scripts(answer)
    answer = remove_chemistry_units(answer)

    return answer.strip()


def remove_chemistry_units(answer):
    """Remove common chemistry units from numerical answers."""
    if not answer:
        return answer

    unit_pattern = (
        r"^([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*(?:"
        + r"Å²|Å\^2|A²|A\^2|"
        + r"g/mol|Da|kDa|amu|"
        + r"logP|cLogP|"
        + r"mol/L|mmol/L|"
        + r"kJ/mol|kcal/mol|"
        + r"°C|°F|"
        + r"mmHg|"
        + r"μM|μL|μg|μm|μs|μA|"
        + r"uM|uL|ug|um|us|uA|"
        + r"mM|mL|mg|mm|ms|mV|mA|"
        + r"nM|nm|ns|"
        + r"pM|Pa|"
        + r"kPa|kDa|kJ|kg|km|kHz|"
        + r"MPa|MHz|"
        + r"GHz|THz|"
        + r"eV|"
        + r"atm|bar|torr|"
        + r"cal|kcal|"
        + r"min|hr|hours|"
        + r"percent|units|"
        + r"M|L|K|J|V|A|C|F|g|m|s|h|%"
        + r")(?:\s*[.,;:]?)*$"
    )

    match = re.search(unit_pattern, answer, re.IGNORECASE | re.UNICODE)
    if match:
        return match.group(1)

    return answer


def convert_to_appropriate_type(answer):
    """Convert answer to appropriate type (int, float, or string)."""
    if not answer:
        return answer

    answer_str = str(answer).strip()

    # Try integer first
    try:
        if "." not in answer_str:
            return int(answer_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(answer_str)
    except ValueError:
        pass

    try:
        return ast.literal_eval(answer_str)
    except (ValueError, SyntaxError):
        pass

    return answer_str


def normalize_sub_super_scripts(text):
    """Normalize Unicode subscript and superscript digits to standard digits."""
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
    return text.translate(subscripts).translate(superscripts)


def extract_answer_only(response):
    """
    Enhanced extraction function that extracts content in the following priority:
    1. Content between <answer></answer> tags (highest priority)
    2. Valid JSON objects or arrays
    3. Python dictionary/list formats
    """
    if not response or not isinstance(response, str):
        return None

    # Priority 1: Check for answer tags
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].split("</answer>")[0]
        return answer

    # Priority 2 & 3: Find and validate JSON/dict/list/tuple structures
    candidates = []

    def find_balanced_structures(text):
        """Find all balanced bracket/brace structures in text."""
        structures = []
        stack = {"[": [], "{": [], "(": []}
        reverse_matching = {"]": "[", "}": "{", ")": "("}

        i = 0
        while i < len(text):
            char = text[i]

            if char == '"':
                i += 1
                while i < len(text):
                    if text[i] == '"' and (i == 0 or text[i - 1] != "\\"):
                        break
                    i += 1
                i += 1
                continue
            elif char == "'":
                if i > 0 and text[i - 1] in ["{", "[", ":", ",", " ", "("]:
                    j = i + 1
                    found_close = False
                    while j < len(text) and j < i + 1000:
                        if (
                            text[j] == "'"
                            and (j == 0 or text[j - 1] != "\\")
                            and j + 1 < len(text)
                            and text[j + 1] in ["}", "]", ":", ",", " ", ")"]
                        ):
                            found_close = True
                            break
                        j += 1

                    if found_close:
                        i = j + 1
                        continue

            if char in stack:
                stack[char].append(i)
            elif char in reverse_matching:
                opener = reverse_matching[char]
                if stack[opener]:
                    start = stack[opener].pop()
                    complete_structure = text[start : i + 1]
                    structures.append((start, complete_structure))

            i += 1

        return structures

    structures = find_balanced_structures(response)

    # Filter out nested structures
    filtered_structures = []
    for i, (start1, struct1) in enumerate(structures):
        is_nested = False
        for j, (start2, struct2) in enumerate(structures):
            if i != j:
                end1 = start1 + len(struct1)
                end2 = start2 + len(struct2)
                if start2 <= start1 and end2 >= end1 and len(struct2) > len(struct1):
                    is_nested = True
                    break
        if not is_nested:
            filtered_structures.append((start1, struct1))

    for start_pos, candidate in filtered_structures:
        if candidate.startswith("{"):
            structure_type = "dict"
        elif candidate.startswith("["):
            structure_type = "list"
        elif candidate.startswith("("):
            structure_type = "tuple"
        else:
            continue

        if structure_type in ["dict", "list"]:
            try:
                json.loads(candidate)
                candidates.append((start_pos, candidate, "json"))
                continue
            except (json.JSONDecodeError, ValueError):
                pass

        try:
            ast.literal_eval(candidate)
            candidates.append((start_pos, candidate, "python"))
        except (ValueError, SyntaxError, TypeError):
            if (
                structure_type == "dict"
                and ":" in candidate
                and '"' not in candidate
                and "'" not in candidate
            ):
                candidates.append((start_pos, candidate, "simple_dict"))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    return None
