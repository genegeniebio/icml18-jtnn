from rdkit.Chem import Descriptors, MolFromSmiles

from utils import sascorer

smiles = ['CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1']

targets = []

for smile in smiles:
    mol = MolFromSmiles(smile)
    logp = Descriptors.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    targets.append((smile, logp - sa))

for x, y in sorted(targets, key=lambda x: x[1]):
    print x, y
