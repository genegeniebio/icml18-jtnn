'''
synbiochem (c) University of Manchester 2018

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
from collections import defaultdict

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import rdkit.Chem as Chem


MST_MAX_WEIGHT = 100
MAX_NCAND = 2000


def set_atommap(mol, num=0):
    '''Set atom map.'''
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_mol(smiles):
    '''Get mol from smiles.'''
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        Chem.Kekulize(mol)

    return mol


def get_smiles(mol):
    '''Get smiles from mol.'''
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol):
    '''sanitize.'''
    return get_mol(get_smiles(mol))


def decode_stereo(smiles2d):
    '''Decode stereochemistry.'''
    mol = Chem.MolFromSmiles(smiles2d)
    dec_isomers = list(Chem.EnumerateStereoisomers.EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(
        mol, isomericSmiles=True)) for mol in dec_isomers]

    smiles3d = [Chem.MolToSmiles(mol, isomericSmiles=True)
                for mol in dec_isomers]

    chiral_n = [atom.GetIdx()
                for atom in dec_isomers[0].GetAtoms()
                if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == 'N']

    if chiral_n:
        for mol in dec_isomers:
            for idx in chiral_n:
                mol.GetAtomWithIdx(idx).SetChiralTag(
                    Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3d.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3d


def copy_atom(atom):
    '''Copy atom.'''
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    '''Copy edit molecule.'''
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        new_mol.AddBond(bond.GetBeginAtom().GetIdx(),
                        bond.GetEndAtom().GetIdx(),
                        bond.GetBondType())
    return new_mol


def get_clique_mol(mol, atoms):
    '''Get clique molecule.'''
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol


def tree_decomp(mol):
    '''Tree decomposition.'''
    if mol.GetNumAtoms() == 1:
        return [[0]], []

    cliques = []

    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append([bond.GetBeginAtom().GetIdx(),
                            bond.GetEndAtom().GetIdx()])

    cliques.extend([list(x) for x in Chem.GetSymmSSSR(mol)])

    nei_list = [[] for i in xrange(mol.GetNumAtoms())]

    for i in xrange(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms
    for i in xrange(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2:
                    continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]

    nei_list = [[] for i in xrange(mol.GetNumAtoms())]

    for i in xrange(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build edges and add singleton cliques
    edges = defaultdict(int)

    for atom in xrange(mol.GetNumAtoms()):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        # In general, if len(cnei) >= 3, a singleton should be added, but 1
        # bond + 2 ring is currently not dealt with.
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])

            for c_1 in cnei:
                edges[(c_1, len(cliques) - 1)] = 1
        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom])

            for c_1 in cnei:
                edges[(c_1, len(cliques) - 1)] = MST_MAX_WEIGHT - 1
        else:
            for i in xrange(len(cnei)):
                for j in xrange(i + 1, len(cnei)):
                    inter = set(cliques[cnei[i]]) & set(cliques[cnei[j]])

                    if edges[(cnei[i], cnei[j])] < len(inter):
                        # cnei[i] < cnei[j] by construction
                        edges[(cnei[i], cnei[j])] = len(inter)

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.iteritems()]

    if not edges:
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in xrange(len(row))]
    return (cliques, edges)


def atom_equal(atom_1, atom_2):
    '''Atoms equal?'''
    return atom_1.GetSymbol() == atom_2.GetSymbol() and \
        atom_1.GetFormalCharge() == atom_2.GetFormalCharge()

# Bond type not considered because all aromatic (so SINGLE matches DOUBLE)


def ring_bond_equal(bond_1, bond_2, reverse=False):
    '''Ring bond equal?'''
    bond_1 = (bond_1.GetBeginAtom(), bond_1.GetEndAtom())

    if reverse:
        bond_2 = (bond_2.GetEndAtom(), bond_2.GetBeginAtom())
    else:
        bond_2 = (bond_2.GetBeginAtom(), bond_2.GetEndAtom())

    return atom_equal(bond_1[0], bond_2[0]) \
        and atom_equal(bond_1[1], bond_2[1])


def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
    '''Attach mols.'''
    prev_nids = [node.nid for node in prev_nodes]

    for nei_node in prev_nodes + neighbors:
        amap = nei_amap[nei_node.get_nid()]

        for atom in nei_node.get_mol().GetAtoms():
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        if nei_node.get_mol().GetNumBonds() == 0:
            nei_atom = nei_node.get_mol().GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_node.get_mol().GetBonds():
                atom_1 = amap[bond.GetBeginAtom().GetIdx()]
                atom_2 = amap[bond.GetEndAtom().GetIdx()]

                if not ctr_mol.GetBondBetweenAtoms(atom_1, atom_2):
                    ctr_mol.AddBond(atom_1, atom_2, bond.GetBondType())
                elif nei_node.get_nid() in prev_nids:  # father node overrides
                    ctr_mol.RemoveBond(atom_1, atom_2)
                    ctr_mol.AddBond(atom_1, atom_2, bond.GetBondType())

    return ctr_mol


def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    '''Local attach.'''
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei.get_nid(): {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()

# This version records idx mapping between ctr_mol and nei_mol


def enum_attach(ctr_mol, nei_node, amap, singletons):
    '''Enumerate attach.'''
    att_confs = []
    black_list = [atom_idx for nei_id, atom_idx,
                  _ in amap if nei_id in singletons]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx()
                 not in black_list]
    ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

    if nei_node.get_mol().GetNumBonds() == 0:  # neighbor singleton
        nei_atom = nei_node.get_mol().GetAtomWithIdx(0)
        used_list = [atom_idx for _, atom_idx, _ in amap]
        for atom in ctr_atoms:
            if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                new_amap = amap + [(nei_node.get_nid(), atom.GetIdx(), 0)]
                att_confs.append(new_amap)

    elif nei_node.get_mol().GetNumBonds() == 1:  # neighbor is a bond
        bond = nei_node.get_mol().GetBondWithIdx(0)

        for atom in ctr_atoms:
            # Optimize if atom is carbon (other atoms may change valence)
            if atom.GetAtomicNum() == 6 \
                    and atom.GetTotalNumHs() < int(bond.GetBondTypeAsDouble()):
                continue
            if atom_equal(atom, bond.GetBeginAtom()):
                new_amap = amap + [(nei_node.get_nid(),
                                    atom.GetIdx(),
                                    bond.GetEndAtom().GetIdx())]
                att_confs.append(new_amap)
            elif atom_equal(atom, bond.GetEndAtom()):
                new_amap = amap + [(nei_node.get_nid(),
                                    atom.GetIdx(),
                                    bond.GetEndAtom().GetIdx())]
                att_confs.append(new_amap)
    else:
        # intersection is an atom
        for atom_1 in ctr_atoms:
            for atom_2 in nei_node.get_mol().GetAtoms():
                if atom_equal(atom_1, atom_2):
                    # Optimize if atom is carbon (other atoms may change
                    # valence)
                    if atom_1.GetAtomicNum() == 6 and \
                            atom_1.GetTotalNumHs() \
                            + atom_2.GetTotalNumHs() < 4:
                        continue
                    new_amap = amap + [(nei_node.get_nid(),
                                        atom_1.GetIdx(),
                                        atom_2.GetIdx())]
                    att_confs.append(new_amap)

        # intersection is an bond
        if ctr_mol.GetNumBonds() > 1:
            for bond_1 in ctr_bonds:
                for bond_2 in nei_node.get_mol().GetBonds():
                    if ring_bond_equal(bond_1, bond_2):
                        new_amap = amap + [(nei_node.get_nid(),
                                            bond_1.GetBeginAtom().GetIdx(),
                                            bond_2.GetBeginAtom().GetIdx()),
                                           (nei_node.get_nid(),
                                            bond_1.GetEndAtom().GetIdx(),
                                            bond_2.GetEndAtom().GetIdx())]
                        att_confs.append(new_amap)

                    if ring_bond_equal(bond_1, bond_2, reverse=True):
                        new_amap = amap + [(nei_node.get_nid(),
                                            bond_1.GetBeginAtom().GetIdx(),
                                            bond_2.GetEndAtom().GetIdx()),
                                           (nei_node.get_nid(),
                                            bond_1.GetEndAtom().GetIdx(),
                                            bond_2.GetBeginAtom().GetIdx())]
                        att_confs.append(new_amap)
    return att_confs

# Try rings first: Speed-Up


def enum_assemble(node, neighbors, prev_nodes=None, prev_amap=None):
    '''Enum assemble.'''
    if not prev_nodes:
        prev_nodes = []

    if not prev_amap:
        prev_amap = []

    all_attach_confs = []
    singletons = [nei_node.get_nid()
                  for nei_node in neighbors + prev_nodes
                  if nei_node.get_mol().GetNumAtoms() == 1]

    def search(cur_amap, depth):
        '''search.'''
        if len(all_attach_confs) > MAX_NCAND:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        cand_amap = enum_attach(node.get_mol(), nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach(
                node.get_mol(), neighbors[:depth + 1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue
            smiles = get_smiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            candidates.append(amap)

        if not candidates:
            return

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)
    cand_smiles = set()
    candidates = []
    for amap in all_attach_confs:
        cand_mol = local_attach(node.get_mol(), neighbors, prev_nodes, amap)
        cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles:
            continue
        cand_smiles.add(smiles)
        Chem.Kekulize(cand_mol)
        candidates.append((smiles, cand_mol, amap))

    return candidates
