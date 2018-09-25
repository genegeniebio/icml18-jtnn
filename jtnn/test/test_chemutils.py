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
        cls.__smiles = ['O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1',
                        'O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2',
                        'ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3',
                        'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br',
                        'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1',
                        'O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1']

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
