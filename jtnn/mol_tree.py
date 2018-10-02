# coding=utf-8

"""
synbiochem (c) University of Manchester 2018

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
"""
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=wrong-import-order
from collections import defaultdict
from itertools import combinations

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from jtnn import chemutils
import rdkit.Chem as Chem
# from rdkit.Chem import Draw


class MolTreeNode(object):
    """Class to represent a molecular tree node."""

    def __init__(self, smiles, clique=None):
        self.__smiles = smiles
        self.__mol = chemutils.get_mol(self.__smiles)
        self.__clique = list(clique if clique else [])
        self.__neighbors = []
        self.__node_id = None
        self.__cands = []
        self.__cand_mols = []
        self.__label = None
        self.__label_mol = None

    def get_smiles(self):
        """Get smiles."""
        return self.__smiles

    def get_mol(self):
        """Get mol."""
        return self.__mol

    def get_clique(self):
        """Get clique."""
        return self.__clique

    def get_neighbors(self):
        """Get clique."""
        return self.__neighbors

    def get_node_id(self):
        """Get node_id."""
        return self.__node_id

    def get_candidates(self):
        """Get candidates."""
        return self.__cands

    def get_label(self):
        """Get label."""
        return self.__label

    def is_leaf(self):
        """Is the node a leaf?"""
        return len(self.__neighbors) == 1

    def add_neighbor(self, neighbor):
        """Add neighbour."""
        self.__neighbors.append(neighbor)

    def set_node_id(self, node_id):
        """Set node_id."""
        self.__node_id = node_id

    def recover(self, original_mol):
        """Recover."""
        clique = list(self.__clique)

        if not self.is_leaf():
            for clq_idx in self.__clique:
                original_mol.GetAtomWithIdx(
                    clq_idx).SetAtomMapNum(self.__node_id)

        for neighbour in self.__neighbors:
            clique.extend(neighbour.get_clique())

            # Leaf node, no need to mark:
            if neighbour.is_leaf():
                continue

            for clq_idx in neighbour.get_clique():
                # Allow singleton node override the atom mapping:
                if clq_idx not in self.__clique or \
                        len(neighbour.get_clique()) == 1:
                    atom = original_mol.GetAtomWithIdx(clq_idx)
                    atom.SetAtomMapNum(neighbour.get_node_id())

        clique = list(set(clique))

        label_mol = _get_clique_mol(original_mol, clique)

        self.__label = Chem.MolToSmiles(
            Chem.MolFromSmiles(chemutils.get_smiles(label_mol)))
        self.__label_mol = chemutils.get_mol(self.__label)

        for clq_idx in clique:
            original_mol.GetAtomWithIdx(clq_idx).SetAtomMapNum(0)

        return self.__label

    def assemble(self):
        """Assemble."""
        neighbors = sorted([nei for nei in self.__neighbors
                            if nei.get_mol().GetNumAtoms() > 1],
                           key=lambda x: x.get_mol().GetNumAtoms(),
                           reverse=True)

        singletons = [nei for nei in self.__neighbors
                      if nei.get_mol().GetNumAtoms() == 1]

        neighbors = singletons + neighbors

        cands = chemutils.enum_assemble(self, neighbors)

        if cands:
            self.__cands, self.__cand_mols, _ = zip(*cands)
            self.__cands = list(self.__cands)
            self.__cand_mols = list(self.__cand_mols)

    def __repr__(self):
        return '%s (%s)' % (self.__smiles, str(self.__clique))


class MolTree(object):
    """Class to represent a molecular tree."""

    def __init__(self, smiles):
        self.__smiles = smiles
        self.__mol = chemutils.get_mol(smiles)
        self.__nodes = []

        # Stereo generation:
        mol = Chem.MolFromSmiles(smiles)
        self.__smiles3d = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.__stereo_cands = chemutils.decode_stereo(Chem.MolToSmiles(mol))

        # Calculate cliques and edges:
        cliques, edges = _tree_decomp(self.__mol)

        # Add nodes:
        root = 0

        for clq_idx, clique in enumerate(cliques):
            clq_mol = _get_clique_mol(self.__mol, clique)
            self.__nodes.append(MolTreeNode(chemutils.get_smiles(clq_mol),
                                            clique))

            if min(clique) == 0:
                root = clq_idx

        # Add edges:
        for edge_x, edge_y in edges:
            self.__nodes[edge_x].add_neighbor(self.__nodes[edge_y])
            self.__nodes[edge_y].add_neighbor(self.__nodes[edge_x])

        if root > 0:
            self.__nodes[0], self.__nodes[root] = \
                self.__nodes[root], self.__nodes[0]

        for i, node in enumerate(self.__nodes):
            node.set_node_id(i + 1)

            # Leaf node mol is not marked:
            if not node.is_leaf():
                chemutils.set_atommap(node.get_mol(), node.get_node_id())

    def get_smiles(self):
        """Get smiles."""
        return self.__smiles

    def get_nodes(self):
        """Get nodes."""
        return self.__nodes

    def size(self):
        """Get size."""
        return len(self.__nodes)

    def recover(self):
        """Recover."""
        for node in self.__nodes:
            node.recover(self.__mol)

    def assemble(self):
        """Assemble."""
        for node in self.__nodes:
            node.assemble()


def _tree_decomp(mol):
    """Tree decomposition."""
    if mol.GetNumAtoms() == 1:
        cliques = [[0]]
        edges = []
    else:
        # For 'cliques' read bonds, paired by atom idx:
        cliques = [[bond.GetBeginAtom().GetIdx(),
                    bond.GetEndAtom().GetIdx()]
                   for bond in mol.GetBonds()
                   if not bond.IsInRing()]

        # Add rings:
        cliques.extend([list(x) for x in Chem.GetSymmSSSR(mol)])

        # Merge rings:
        cliques = _merge_rings(mol, cliques)

        edges = _get_edges(mol, cliques)

        if edges:
            edges = _calc_min_span_tree(edges, cliques)

    return cliques, edges


def _merge_rings(mol, cliques):
    """Merge rings with intersection > 2 atoms."""
    neighbours = _get_neighbours(mol, cliques)

    for idx, clique in enumerate(cliques):
        if len(clique) <= 2:
            continue

        for atom in clique:
            for j in neighbours[atom]:
                if idx >= j or len(cliques[j]) <= 2:
                    continue

                inter = set(clique) & set(cliques[j])

                if len(inter) > 2:
                    clique.extend(cliques[j])
                    clique = list(set(clique))
                    cliques[j] = []

    # Remove empty cliques:
    return [clique for clique in cliques if clique]


def _get_edges(mol, cliques):
    """Build edges between cliques and add singleton cliques."""
    edges = defaultdict(int)
    neighbours = _get_neighbours(mol, cliques)
    max_weight = 128

    for atom_idx, neighbour in enumerate(neighbours):
        if len(neighbour) <= 1:
            continue

        bonds = [clq_idx for clq_idx in neighbour
                 if len(cliques[clq_idx]) == 2]

        rings = [clq_idx for clq_idx in neighbour
                 if len(cliques[clq_idx]) > 4]

        # In general, if len(neighbour) >= 3, a singleton should be added,
        # but 1 bond + 2 ring is currently not dealt with:
        if len(bonds) > 2 or (len(bonds) == 2 and len(neighbour) > 2):
            cliques.append([atom_idx])

            for clq_idx in neighbour:
                edges[(clq_idx, len(cliques) - 1)] = 1

        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom_idx])

            for clq_idx in neighbour:
                edges[(clq_idx, len(cliques) - 1)] = max_weight - 1
        else:
            for (clq_idx_1, clq_idx_2) in combinations(neighbour, r=2):
                inter = set(cliques[clq_idx_1]) & set(cliques[clq_idx_2])

                if edges[(clq_idx_1, clq_idx_2)] < len(inter):
                    # clq_idx_1 < clq_idx_2 by construction
                    edges[(clq_idx_1, clq_idx_2)] = len(inter)

    return [clq_idxs + (max_weight - weight,)
            for clq_idxs, weight in edges.items()]


def _calc_min_span_tree(edges, cliques):
    """Calculate minimum spanning tree."""
    row, col, weight = zip(*edges)

    clique_graph = csr_matrix((weight,
                               (row, col)),
                              shape=(len(cliques), len(cliques)))

    junc_tree = minimum_spanning_tree(clique_graph)
    return zip(*junc_tree.nonzero())


def _get_neighbours(mol, cliques):
    """Neighbours are a list of cliques that each atom belongs to."""
    neighbours = [[] for _ in range(mol.GetNumAtoms())]

    for idx, clique in enumerate(cliques):
        for atom in clique:
            neighbours[atom].append(idx)

    return neighbours


def _get_clique_mol(mol, atoms):
    """Get clique molecule."""
    clq_smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    clq_mol = Chem.MolFromSmiles(clq_smiles, sanitize=False)
    clq_mol = chemutils.copy_edit_mol(clq_mol).GetMol()
    return chemutils.sanitize(clq_mol)


def _get_vocabulary(filename, max_tree_width=15):
    """Get vocabulary."""
    vocabulary = set()

    with open(filename) as fle:
        for line in fle:
            nodes = MolTree(line.split()[0]).get_nodes()

            for node in nodes:
                if node.get_mol().GetNumAtoms() > max_tree_width:
                    raise Exception(
                        'Molecule %s has a high tree-width' % node.smiles)
                # else:
                vocabulary.add(node)

    return vocabulary
