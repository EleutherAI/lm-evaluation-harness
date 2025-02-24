# Chat Template Delimiter Handling Update

## Overview
This change modifies how delimiters are handled when applying chat templates in the request construction process for likelihood and multiple-choice based tasks. When `apply_chat_template` is set to `True`, the target delimiter is now set to an empty string instead of using the configured delimiter.

## Background
By default, the system uses a target delimiter (typically a whitespace " ") between the context and target text when constructing prompts. The full string is constructed as:
```
doc_to_text(doc) + target_delimiter + doc_to_target(doc)
```

While this worked well for base models where we wanted the model to predict a single whitespace followed by the answer, chat models have their own formatting conventions that handle spacing differently.

## The Change
- When `apply_chat_template=True`, the target delimiter is now empty ("") instead of the default whitespace
- This prevents interference between chat template formatting and the default delimiter system
- Particularly important for multiple choice tasks where the template itself handles spacing

## Example
```
# Before (with default delimiter " ")
<user>Question: What color is the sky?\nAnswer:<assistant> blue

# After
<user>Question: What color is the sky?\nAnswer:<assistant>blue
```
