"""Utility functions for Phonebook Lookup evaluation."""

import re
import random
from typing import List, Dict, Any, Tuple


def generate_phone_number() -> str:
    """Generate a random phone number in XXX-XXX-XXXX format."""
    area = random.randint(200, 999)
    exchange = random.randint(200, 999)
    number = random.randint(0, 9999)
    return f"{area}-{exchange}-{number:04d}"


def generate_name(first_names: List[str], last_names: List[str]) -> str:
    """Generate a random full name."""
    first = random.choice(first_names)
    last = random.choice(last_names)
    return f"{first} {last}"


def create_phonebook_entry(name: str, phone: str) -> str:
    """Create a single phonebook entry."""
    return f"Name: {name}, Phone: {phone}"


def extract_phone_number(text: str) -> str:
    """Extract phone number from model output."""
    # Look for phone number patterns
    patterns = [
        r"(\d{3}-\d{3}-\d{4})",  # XXX-XXX-XXXX
        r"(\d{3}\.\d{3}\.\d{4})",  # XXX.XXX.XXXX
        r"(\d{3}\s\d{3}\s\d{4})",  # XXX XXX XXXX
        r"(\d{10})",  # XXXXXXXXXX
        r"phone(?:\s+number)?(?:\s+is)?[:\s]*([^\n,]+)",  # "phone is ..."
        r"number(?:\s+is)?[:\s]*([^\n,]+)",  # "number is ..."
        r"^([^\n,]+)$",  # Just the answer
    ]

    text = text.strip()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            # Normalize format if it's a valid phone number
            digits = re.sub(r"\D", "", result)
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            return result

    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Check if predicted phone number matches ground truth."""
    pred = extract_phone_number(prediction)

    # Normalize both for comparison
    pred_digits = re.sub(r"\D", "", pred)
    truth_digits = re.sub(r"\D", "", ground_truth)

    return 1.0 if pred_digits == truth_digits else 0.0


def create_phonebook_context(
    entries: List[Tuple[str, str]], target_name: str, target_position: str = "random"
) -> Tuple[str, str, str]:
    """
    Create a phonebook context with target at specified position.

    Args:
        entries: List of (name, phone) tuples
        target_name: Name to query for
        target_position: Where to place target ("beginning", "middle", "end", "random")

    Returns:
        Tuple of (context, question, answer)
    """
    # Find target entry
    target_entry = None
    target_phone = None
    other_entries = []

    for name, phone in entries:
        if name == target_name:
            target_entry = create_phonebook_entry(name, phone)
            target_phone = phone
        else:
            other_entries.append(create_phonebook_entry(name, phone))

    if not target_entry:
        raise ValueError(f"Target name '{target_name}' not found in entries")

    # Position the target entry
    if target_position == "beginning":
        position_idx = 0
    elif target_position == "end":
        position_idx = len(other_entries)
    elif target_position == "middle":
        position_idx = len(other_entries) // 2
    else:  # random
        position_idx = random.randint(0, len(other_entries))

    # Build phonebook
    phonebook_entries = (
        other_entries[:position_idx] + [target_entry] + other_entries[position_idx:]
    )

    context = "Phonebook:\n" + "\n".join(phonebook_entries)
    question = f"What is {target_name}'s phone number?"

    return context, question, target_phone


def process_results_gen(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """Process generation results for Phonebook Lookup tasks."""
    prediction = results[0] if results else ""

    # Get ground truth phone number
    answer = doc.get("answer", doc.get("phone", ""))

    # Compute exact match score
    score = exact_match(prediction, answer)

    # Also track position for analysis
    position = doc.get("position", "unknown")

    return {"acc": score, f"acc_{position}": score}  # Position-specific accuracy


def create_prompt(doc: Dict[str, Any]) -> str:
    """Create prompt for Phonebook Lookup tasks."""
    context = doc.get("context", "")
    question = doc.get("question", "")

    prompt = f"""{context}

Question: {question}
Answer:"""

    return prompt


# Sample name lists for generation
FIRST_NAMES = [
    "James",
    "Mary",
    "John",
    "Patricia",
    "Robert",
    "Jennifer",
    "Michael",
    "Linda",
    "William",
    "Elizabeth",
    "David",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Christopher",
    "Nancy",
    "Daniel",
    "Lisa",
    "Matthew",
    "Betty",
    "Anthony",
    "Helen",
    "Mark",
    "Sandra",
    "Donald",
    "Donna",
    "Steven",
    "Carol",
    "Kenneth",
    "Ruth",
    "Andrew",
    "Sharon",
    "Joshua",
    "Michelle",
    "Kevin",
    "Laura",
    "Brian",
    "Sarah",
    "George",
    "Kimberly",
    "Edward",
    "Deborah",
    "Ronald",
    "Dorothy",
    "Timothy",
    "Lisa",
    "Jason",
    "Nancy",
    "Jeffrey",
    "Karen",
    "Ryan",
    "Betty",
    "Jacob",
    "Helen",
    "Gary",
    "Sandra",
    "Nicholas",
    "Donna",
    "Eric",
    "Carol",
    "Jonathan",
    "Ruth",
    "Stephen",
    "Sharon",
    "Larry",
    "Michelle",
    "Justin",
    "Laura",
    "Scott",
    "Sarah",
    "Brandon",
    "Kimberly",
    "Benjamin",
    "Deborah",
    "Samuel",
    "Jessica",
    "Frank",
    "Shirley",
    "Gregory",
    "Cynthia",
    "Raymond",
    "Angela",
    "Alexander",
    "Melissa",
    "Patrick",
    "Brenda",
    "Jack",
    "Emma",
    "Dennis",
    "Amy",
    "Jerry",
    "Anna",
    "Tyler",
    "Rebecca",
    "Aaron",
    "Virginia",
    "Jose",
    "Kathleen",
]

LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
    "Wright",
    "Scott",
    "Torres",
    "Nguyen",
    "Hill",
    "Flores",
    "Green",
    "Adams",
    "Nelson",
    "Baker",
    "Hall",
    "Rivera",
    "Campbell",
    "Mitchell",
    "Carter",
    "Roberts",
    "Gomez",
    "Phillips",
    "Evans",
    "Turner",
    "Diaz",
    "Parker",
    "Cruz",
    "Edwards",
    "Collins",
    "Reyes",
    "Stewart",
    "Morris",
    "Morales",
    "Murphy",
    "Cook",
    "Rogers",
    "Gutierrez",
    "Ortiz",
    "Morgan",
    "Cooper",
    "Peterson",
    "Bailey",
    "Reed",
    "Kelly",
    "Howard",
    "Ramos",
    "Kim",
    "Cox",
    "Ward",
    "Richardson",
    "Watson",
    "Brooks",
    "Chavez",
    "Wood",
    "James",
    "Bennett",
    "Gray",
    "Mendoza",
    "Ruiz",
    "Hughes",
    "Price",
    "Alvarez",
    "Castillo",
    "Sanders",
    "Patel",
    "Myers",
    "Long",
    "Ross",
    "Foster",
    "Jimenez",
    "Powell",
    "Jenkins",
    "Perry",
]


def generate_phonebook_data(
    num_entries: int, target_position: str = "random"
) -> Dict[str, Any]:
    """Generate a complete phonebook lookup instance."""
    # Generate unique names
    names_phones = []
    used_names = set()

    while len(names_phones) < num_entries:
        name = generate_name(FIRST_NAMES, LAST_NAMES)
        if name not in used_names:
            used_names.add(name)
            phone = generate_phone_number()
            names_phones.append((name, phone))

    # Select target
    target_idx = random.randint(0, num_entries - 1)
    target_name = names_phones[target_idx][0]

    # Create context
    context, question, answer = create_phonebook_context(
        names_phones, target_name, target_position
    )

    return {
        "context": context,
        "question": question,
        "answer": answer,
        "position": target_position,
        "num_entries": num_entries,
        "target_name": target_name,
    }
