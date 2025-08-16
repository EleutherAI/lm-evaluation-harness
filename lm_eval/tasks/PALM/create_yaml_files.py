import yaml
from pathlib import Path


def create_yaml_files():
    countries = [
        "Saudi Arabia", "Syria", "Egypt", "Jordan", "Djibouti", "Somalia",
        "Sudan", "Yemen", "General", "Tunisia", "Mauritania", "Morocco",
        "Iraq", "Palestine", "UAE", "Kuwait", "Qatar", "Algeria", "Comors",
        "Bahrain", "Lebanon", "Libya", "Oman"
    ]
    for country in countries:
        filename = f"{country.lower().replace(' ', '_')}"
        filepath = Path(__file__).parent / f"palm_{filename}.yaml"
        with open(filepath, "w") as fout:
            fout.writelines([
                "include: _default_template_yaml\n",
                f"task: palm_{filename}\n",
                f"process_docs: !function utils.get_{filename}_samples\n",
            ])
        print(f"""
def get_{filename}_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "{country}")
""")
        
    for country in countries:
        filename = f"{country.lower().replace(' ', '_')}"
        print(f"- palm_{filename}")

    

if __name__ == "__main__":
    create_yaml_files()