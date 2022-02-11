from pathlib import Path
import pytest

from rp2biosensor import RP2Objects


@pytest.fixture
def smiles_to_test():
    return [
        # (not yet canonized, canonized)
        ('[H]OC(=O)C([H])(O[H])C([H])([H])[H]','CC(O)C(=O)O'),
        ('[H]O[H]', 'O'),
        ('[H]OC([H])=O', 'O=CO'),
    ]


def test_canonize_smiles(smiles_to_test):
    for case in smiles_to_test:
        query, ans = case
        assert RP2Objects.canonize_smiles(query) == ans


class TestIDsHandler:
    def test_new_id(self):
        idh = RP2Objects.IDsHandler()
        assert idh.make_new_id() == 'ID_0000000001'


class TestCompound:

    smi = '[H]O[H]'

    def test_init_ids(self):
        # One should initialize the ID handler
        with pytest.raises(AttributeError):
            c = RP2Objects.Compound(smiles=self.smi)

    def test_init(self):
        # This is better
        RP2Objects.Compound.init_id_handler()
        c = RP2Objects.Compound(smiles=self.smi)
        assert c.uid == 'CMPD_0000000001'
    
    def test_compute_struct(self):
        RP2Objects.Compound.init_id_handler()
        c = RP2Objects.Compound(smiles=self.smi)
        c.compute_structures()
        assert c.smiles == 'O'

    def test_cids(self):
        RP2Objects.Compound.init_id_handler()
        c = RP2Objects.Compound()
        c.add_cid('B')
        c.add_cid('A')
        assert c.get_cids() == ['A', 'B']


class TestTransformatoin:
    # To be done
    pass


class TestRP2parser:
    # To be done
    pass


class TestRetroGraph:
    # To be done
    pass