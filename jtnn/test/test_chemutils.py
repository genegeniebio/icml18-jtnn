'''
synbiochem (c) University of Manchester 2018

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
import unittest

import rdkit.Chem

from jtnn.chemutils import copy_edit_mol, set_atommap
from jtnn.mol_tree import MolTree


class Test(unittest.TestCase):
    '''Test class for chemutils.'''

    @classmethod
    def setUpClass(cls):
        cls.__smiles = ['CCC(C)CO',     # chiral
                        'Oc1ccccc1']    # ring (phenol)

    def test_tree(self):
        '''test_tree.'''
        for smiles in self.__smiles:
            tree = MolTree(smiles)

            self.assertTrue(tree.get_nodes())

            for node in tree.get_nodes():
                self.assertTrue(node.get_smiles())
                self.assertTrue(all([neighbour.get_smiles()
                                     for neighbour in node.get_neighbors()]))

    def test_decode(self):
        '''test_decode.'''
        for smiles in self.__smiles:
            tree = MolTree(smiles)
            tree.recover()

            cur_mol = copy_edit_mol(tree.get_nodes()[0].get_mol())
            global_amap = [{}] + [{} for _ in tree.get_nodes()]
            global_amap[1] = {atom.GetIdx(): atom.GetIdx()
                              for atom in cur_mol.GetAtoms()}

            cur_mol = cur_mol.GetMol()
            cur_mol = rdkit.Chem.MolFromSmiles(rdkit.Chem.MolToSmiles(cur_mol))
            set_atommap(cur_mol)
            dec_smiles = rdkit.Chem.MolToSmiles(cur_mol)

            gold_smiles = \
                rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(smiles))

            self.assertEqual(gold_smiles, dec_smiles)

    def test_enum(self):
        '''test_enum.'''
        for smiles in self.__smiles:
            tree = MolTree(smiles)
            tree.recover()
            tree.assemble()
            for node in tree.get_nodes():
                if node.get_label() not in node.get_candidates():
                    print tree.get_smiles()
                    print node.get_smiles(), \
                        [x.get_smiles() for x in node.get_neighbors()]
                    print node.get_label(), len(node.get_candidates())


if __name__ == '__main__':
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
