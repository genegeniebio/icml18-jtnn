'''
synbiochem (c) University of Manchester 2018

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
import copy
import rdkit.Chem as Chem


class Vocabulary(object):
    '''Class to represent a vocabulary.'''

    def __init__(self, vocab):
        self.__vocab = vocab
        self.__vmap = {x: i for i, x in enumerate(self.__vocab)}
        self.__slots = [[(atm.GetSymbol(),
                          atm.GetFormalCharge(),
                          atm.GetTotalNumHs())
                         for atm in Chem.MolFromSmiles(smiles).GetAtoms()]
                        for smiles in self.__vocab]

    def get_index(self, smiles):
        '''Get index.'''
        return self.__vmap[smiles]

    def get_smiles(self, idx):
        '''Get smiles.'''
        return self.__vocab[idx]

    def get_slots(self, idx):
        '''Get slots.'''
        return copy.deepcopy(self.__slots[idx])

    def size(self):
        '''Get size.'''
        return len(self.__vocab)
