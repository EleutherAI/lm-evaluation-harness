# Common Pitfalls and Troubleshooting Guide

This document highlights common pitfalls and troubleshooting tips when using this library. We'll continue to add more tips as we discover them.

## YAML Configuration Issues

### Newline Characters in YAML (`\n`)

**Problem:** When specifying newline characters in YAML, they may be interpreted incorrectly depending on how you format them.

```yaml
# ❌ WRONG: Single quotes don't process escape sequences
generation_kwargs:
  until: ['\n']  # Gets parsed as the literal characters '\' and 'n' i.e "\\n"

```
```yaml
# ✅ RIGHT: Use double quotes for escape sequences
generation_kwargs:
  until: ["\n"]  # Gets parsed as an actual newline character

```

**Solutions:**
- Use double quotes for strings containing escape sequences
- For multiline content, use YAML's block scalars (`|` or `>`)
- When generating YAML programmatically, be careful with how template engines handle escape sequences

### Quoting in YAML

**When to use different types of quotes:**

- **No quotes**: Simple values (numbers, booleans, alphanumeric strings without special characters)
  ```yaml
  simple_value: plain text
  number: 42

  ```

- **Single quotes (')**:
  - Preserves literal values
  - Use when you need special characters to be treated literally
  - Escape single quotes by doubling them: `'It''s working'`
  ```yaml
  literal_string: 'The newline character \n is not processed here'
  path: 'C:\Users\name'  # Backslashes preserved

  ```

- **Double quotes (")**:
  - Processes escape sequences like `\n`, `\t`, etc.
  - Use for strings that need special characters interpreted
  - Escape double quotes with backslash: `"He said \"Hello\""`
  ```yaml
  processed_string: "First line\nSecond line"  # Creates actual newline
  unicode: "Copyright symbol: \u00A9"  # Unicode character

  ```
