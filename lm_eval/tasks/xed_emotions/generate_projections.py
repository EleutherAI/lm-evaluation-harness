langs = [
    "ar",
    "bg",
    "br",
    "bs",
    "cn",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "gl",
    "he",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "mk",
    "ml",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "th",
    "tr",
    "tw",
    "vi",
    "zh"
]

if __name__ == "__main__":

    # Generate projections

    for lang in langs:
        if lang == "fi" or lang == "en":
            continue
        contents = f"""
include: ../xed_emotions.yaml

task: xed_emotions_{lang}

metadata:
  lang: "{lang}"
  """
        with open(f"projections/xed_emotions_{lang}.yaml", "w") as f:
            f.write(contents)

    # Meta task

    with open("groups/xed_emotions_all.yaml", "w") as f:
        f.write("""
group: xed_emotions_all
task:
{}
""".format("\n".join([f"    - xed_emotions_{lang}" for lang in langs])))