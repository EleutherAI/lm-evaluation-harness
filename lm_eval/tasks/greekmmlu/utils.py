"""
Greek MMLU utility functions for formatting questions and choices.
"""

PROMPT = "Αυτό είναι μια ερώτηση {}. Επίλεξε τη σωστή απάντηση!\n\nΕρώτηση: {}\n{}\n\n Απάντηση:"


# subjects_gr
subjects_gr = {
    "Economics": "Οικονομικών",
    "Education": "Παιδαγωγικής",
    "Medicine": "Ιατρικής",
    "Electrical Engineering": "Ηλεκτρολόγων Μηχανικών",
    "Greek Mythology": "Ελληνικής Μυθολογίας",
    "Computer Networks & Security": "Δικτύων Υπολογιστών και Ασφάλειας",
    "Law": "Νομικής",
    "Physics": "Φυσικής",
    "Government and Politics": "Διακυβέρνησης και Πολιτικής",
    "Art": "Τέχνης",
    "Greek Literature": "Νεοελληνικής Λογοτεχνίας",
    "World History": "Παγκόσμιας Ιστορίας",
    "General Knowledge": "Γενικών Γνώσεων",
    "World Religions": "Παγκόσμιων Θρησκειών",
    "Mathematics": "Μαθηματικών",
    "Clinical Knowledge": "Κλινικών Γνώσεων",
    "Driving Rules": "Κανόνων Οδικής Κυκλοφορίας",
    "Biology": "Βιολογίας",
    "Civil Engineering": "Πολιτικών Μηχανικών",
    "Computer Science": "Επιστήμης Υπολογιστών",
    "Geography": "Γεωγραφίας",
    "Chemistry": "Χημείας",
    "Prehistory": "Προϊστορίας",
    "Agriculture": "Γεωργίας",
    "Modern Greek Language": "Νεοελληνικής Γλώσσας",
    "Accounting": "Λογιστικής",
    "Greek History": "Ελληνικής Ιστορίας",
    "Management": "Διοίκησης Επιχειρήσεων",
    "Greek Traditions": "Ελληνικών Παραδόσεων",
    "Maritime Safety and Rescue Operations": "Ναυαγοσωστικων Λειτουργιών και Ασφάλειας στη Θάλασσα",
}


# Greek choice labels
LABELS = ["Α.", "Β.", "Γ.", "Δ."]


def doc_to_text(doc):
    """
    Format the question with choices for Greek MMLU.

    Args:
        doc: Dictionary with 'question' and 'choices' fields

    Returns:
        Formatted question string
    """
    question = doc["question"]
    choices = doc["choices"]
    subject = doc["subject"]

    # Convert English subject to Greek
    subject_gr = subjects_gr.get(subject, subject)

    # Format choices with Greek labels
    formatted_choices = []
    for i, choice in enumerate(choices):
        formatted_choices.append(f"{LABELS[i]} {choice}")

    choices_text = "\n".join(formatted_choices)

    return PROMPT.format(subject_gr, question, choices_text)


def doc_to_choice(doc):
    """
    Extract choice labels based on number of choices.

    Args:
        doc: Dictionary with 'choices' field

    Returns:
        List of choice labels (e.g., ['Α', 'Β', 'Γ', 'Δ'])
    """
    num_choices = len(doc["choices"])
    return [LABELS[i][0] for i in range(num_choices)]
