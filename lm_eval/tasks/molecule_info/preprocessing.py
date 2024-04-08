import rdkit.Chem as Chem
import datasets

def process_docs(dataset: datasets.Dataset):
    # def _helper(doc):
    #   # modifies the contents of a single
    #   # document in our dataset.
    #   doc["choices"] = [doc["choice1"], doc["choice2"], doc["wrong_answer"]]
    #   doc["gold"] = doc["label"]
    #   return doc

    def get_related_atom_masses(entry, rounded=True):
        """
        Get the masses of atoms related to the entry.
        """
        mol = Chem.MolFromSmiles(entry['smiles'])
        # restore the hydrogen atoms
        mol = Chem.AddHs(mol)
        set_of_atoms = set()
        atom_count = {}
        for atom in mol.GetAtoms():
            set_of_atoms.add(atom.GetSymbol())
            atom_count[atom.GetSymbol()] = atom_count.get(atom.GetSymbol(), 0) + 1

        dict_of_masses = {}
        pt = Chem.GetPeriodicTable()
        for atom in set_of_atoms:
            dict_of_masses[atom] = pt.GetMostCommonIsotopeMass(atom)
            if rounded:
                dict_of_masses[atom] = int(round(dict_of_masses[atom]))

        # substitute the atom names with A, B, C, etc.
        substition_dict = {atom: f"{chr(65+i)}" for i, atom in enumerate(set_of_atoms)}
        entry['related_atom_masses'] = "(Atom masses: " + '; '.join([f"{atom} - {mass}" for atom, mass in dict_of_masses.items()]) + ")"
        entry['mol_mass_calculation'] = f"({'+'.join([f'{count}*{mass}' for atom, mass in dict_of_masses.items() for atom2, count in atom_count.items() if atom == atom2])})"
        entry['rounded_mol_mass'] = str(sum([mass*count for atom, mass in dict_of_masses.items() for atom2, count in atom_count.items() if atom == atom2]))
        entry['substituted_formula'] = ''
        for i, atom in enumerate(set_of_atoms):
            entry['substituted_formula'] += f"{atom_count[atom]} {substition_dict[atom]}" + ('' if atom_count[atom] == 1 else "'s")
            if i < len(set_of_atoms) - 2:
                entry['substituted_formula'] += ", "
            elif i == len(set_of_atoms) - 2:
                entry['substituted_formula'] += " and "
        entry['substituted_related_atom_masses'] = "(Object weights: " + '; '.join([f"{substition_dict[atom]} - {mass}" for atom, mass in dict_of_masses.items()]) + ")"

        return entry

    return dataset.map(get_related_atom_masses) # returns back a datasets.Dataset object

