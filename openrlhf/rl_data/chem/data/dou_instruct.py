import csv
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Sample data from zinc.csv
data = pd.read_csv("/scratch365/kguo2/Kehan/openrlhf/rl_data/zinc.csv", nrows=200)

data = data[["smiles", "logP", "qed", "SAS"]].to_dict(orient="records")
data = [entry for entry in data if entry["logP"] < 5]
# only pick 50 easy molecules
# data = data[data["logP"] < 5]




def calculate_dou(smiles):
    """Calculate the Degree of Unsaturation (DoU) for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"

    # Count atoms
    num_c = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    num_h = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')
    num_n = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')

    # Calculate DoU
    dou = num_c - num_h / 2 + num_n / 2 + 1
    return dou

def generate_qa_pairs(data):
    qa_pairs = []
    for entry in data:
        smiles = entry["smiles"]
        dou = calculate_dou(smiles)
        is_saturated = dou == 0

        # Question about DoU
        question_dou = f"What is the degree of unsaturation for the molecule with SMILES {smiles}?"
        answer_dou = f"The answer is: {dou}."

        # Question about saturation
        question_saturation = f"Is the molecule with SMILES {smiles} saturated?"
        answer_saturation = f"The answer is: {'saturated' if is_saturated else 'unsaturated'}."

        qa_pairs.append({"Question": question_dou, "Answer": answer_dou, "Difficulty": "hard","Domain": "chemistry","Task": "Degree of Unsaturation","Type": "Advanced Reasoning"})
        qa_pairs.append({"Question": question_saturation, "Answer": answer_saturation, "Difficulty": "medium","Domain": "chemistry","Task": "Degree of Unsaturation","Type": "Advanced Reasoning"})

    return qa_pairs

# Generate QA pairs
qa_pairs = generate_qa_pairs(data)

# save to json
with open("dou_qa_pairs.jsonl", "w") as f:
    for qa in qa_pairs:
        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
# Print the QA pairs
# for i, qa in enumerate(qa_pairs, 1):
#     print(f"Q{i}: {qa['question']}")
#     print(f"A{i}: {qa['answer']}\n")

def smiles_to_iupac(smiles):
    """Convert SMILES to IUPAC name or molecular formula using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    
    # Generate molecular formula as a placeholder for IUPAC name
    iupac_name = rdMolDescriptors.CalcMolFormula(mol)
    return iupac_name

def generate_name_conversion_qa_pairs(data):
    qa_pairs = []
    for entry in data:
        smiles = entry["smiles"]
        iupac_name = smiles_to_iupac(smiles)

        # Instruction QA pair for name conversion
        question = f"Convert the following SMILES to its IUPAC name: {smiles}"
        answer = f"The answer is: {iupac_name}."

        qa_pairs.append({"Question": question, "Answer": answer, "Difficulty": "easy","Domain": "chemistry","Task": "Name Conversion","Type": "Basic Knowledge"})

    return qa_pairs

# Generate name conversion QA pairs
name_conversion_qa_pairs = generate_name_conversion_qa_pairs(data)

# Save to JSON
with open("name_conversion_qa_pairs.jsonl", "w") as f:
    for qa in name_conversion_qa_pairs:
        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

# Print the QA pairs
print(len(name_conversion_qa_pairs))
