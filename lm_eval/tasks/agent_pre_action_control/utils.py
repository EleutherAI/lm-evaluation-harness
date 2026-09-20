ROUTE_CHOICES = ["accept", "ask", "defer", "refuse"]


def doc_to_text(doc):
    messages = doc["input"]
    rendered_messages = []
    for message in messages:
        role = message["role"].upper()
        rendered_messages.append(f"{role}:\n{message['content']}")
    return "\n\n".join(rendered_messages) + "\n\nAnswer:"


def doc_to_choice(doc):
    return ROUTE_CHOICES


def doc_to_target(doc):
    return ROUTE_CHOICES.index(doc["ideal"])
