import logging

eval_logger = logging.getLogger(__name__)

def process_smiles(doc, results):
    reference = doc["SMILES"]
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError(
            "This evaluation requires RDKit. Please install rdkit via `conda install -c conda-forge rdkit`"
        )
    mols = results[0]
    if len(mols) == 0:
        return {"acc": 0.0}
    if len(mols) > 1:
        eval_logger.info("Multiple molecules found in response. Only the first molecule will be used.")
        smiles = mols[0]
    else:
        smiles = mols[0]
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"acc": 0.0}
    return {
        "acc": 1.0 if Chem.MolToSmiles(mol) == reference else 0.0,
    }
