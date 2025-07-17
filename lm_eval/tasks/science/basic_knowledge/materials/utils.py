# These Q&A sets all differ slightly in set up, so a function is created for each of them.

def process_lammps_vasp(dataset):
    def format_row(row):
        return {
            "Question": (row.get("Question") or "").strip(),
            "Option_A": (row.get("Option_A") or "").strip(),
            "Option_B": (row.get("Option_B") or "").strip(),
            "Option_C": (row.get("Option_C") or "").strip(),
            "Option_D": (row.get("Option_D") or "").strip(),
            "Option_E": (row.get("Option_E") or "").strip(),
            "Option_F": (row.get("Option_F") or "").strip(),
            "Answer": (row.get("Answer") or "").strip(),
            "Comment": (row.get("Comment") or "").strip(),  # VASP / LAMMPS
        }
    return dataset.map(format_row)

def process_mof_synthesis_qa(dataset):
    def format_row(row):
        return {
            "Question": row["Question"].strip(),
            "Option_A": row["Option_A"].strip(),
            "Option_B": row["Option_B"].strip(),
            "Option_C": row["Option_C"].strip(),
            "Option_D": row["Option_D"].strip(),
            "Option_E": row.get("Option_E", "").strip(),
            "Answer": row["Answer"].strip()  # no XML tags!
        }
    return dataset.map(format_row)

def process_battery_electrolyte_qa(dataset):
    def format_row(row):
        return {
            "Question": row["Question"].strip(),
            "Option_A": row["Option_A"].strip(),
            "Option_B": row["Option_B"].strip(),
            "Option_C": row["Option_C"].strip(),
            "Option_D": row.get("Option_D", "").strip(),
            "Answer": row["Answer"].strip()  # ← no <answer> wrapper here!
        }
    return dataset.map(format_row)

def process_biomaterials_qa(dataset):
    def format_row(row):
        # helper to safely pull and strip any field
        def s(key):
            return (row.get(key) or "").strip()

        return {
            "Question": s("Question"),
            "Option_A": s("Option_A"),
            "Option_B": s("Option_B"),
            "Option_C": s("Option_C"),
            "Option_D": s("Option_D"),  # now None → "" → strip() → ""
            "Answer"  : s("Answer"),
        }
    return dataset.map(format_row)
   
def process_composites_qa(dataset):
    def format_row(row):
        # helper to safely pull and strip any field
        def s(key):
            return (row.get(key) or "").strip()

        return {
            "Question": s("Question"),
            "Option_A": s("Option_A"),
            "Option_B": s("Option_B"),
            "Option_C": s("Option_C"),
            "Option_D": s("Option_D"),  # now None → "" → strip() → ""
            "Answer"  : s("Answer"),
        }
    return dataset.map(format_row)

def process_materials_science_qa(dataset):
    def format_row(row):
        # helper to safely pull and strip any field
        def s(key):
            return (row.get(key) or "").strip()

        return {
            "Question": s("Question"),
            "Option_A": s("Option_A"),
            "Option_B": s("Option_B"),
            "Option_C": s("Option_C"),
            "Option_D": s("Option_D"),  # now None → "" → strip() → ""
            "Answer"  : s("Answer"),
        }
    return dataset.map(format_row)