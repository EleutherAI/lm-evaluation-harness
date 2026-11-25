import ast
import re


# Options look like "a ) 4 , 3 , b ) 4 , 4 , ... , e ) 4 , 2". An option can contain
# " , " itself, so match the five options by their "a )" to "e )" markers instead of
# cutting each one at the first comma.
_OPTIONS = re.compile(
    r"a \) (.*?)\s*,\s*b \) (.*?)\s*,\s*c \) (.*?)\s*,\s*d \) (.*?)\s*,\s*e \) (.*)"
)


def doc_to_choice(doc):
    options = doc["options"].strip()
    # Some rows store the options as a list literal: "['a ) 2 : 3', ..., 'e ) 8 : 5']".
    if options.startswith("["):
        options = " , ".join(ast.literal_eval(options))
    return [c.rstrip(" ,") for c in _OPTIONS.fullmatch(options).groups()]
