import re


def parse_thinking(response):
        
    # Strip everything up to and including the first </think>
    if "</think>" in response:
        response = response.split("</think>", 1)[1].lstrip()

    # Extract the content of the first <answer>...</answer> pair if present
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if answer_blocks:
        response = answer_blocks[0].strip()

    # If LaTeX boxed answers exist, take the first one inside the (possibly reduced) response
    matches = re.findall(r"\\boxed\{(.*?)\}", response, re.DOTALL)
    if matches:
        response = matches[0].strip()
        
    return response
