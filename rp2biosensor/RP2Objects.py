"""Convert output of the RetroPath2.0 workflow.

Copyright (C) 2016-2017 Thomas Duigou, JL Faulon's research group, INRA

Use of this source code is governed by the MIT license that can be found in the
LICENSE.txt file.

"""

from __future__ import annotations

import sys
import csv
import json
import urllib
import logging

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.AllChem import Compute2DCoords
from rr_cache import rrCache 


class IDsHandler:
    """Handler in order to generate IDs."""

    def __init__(self, length: int=10, prefix: str='ID', sep: str='_') -> IDsHandler:
        """Buid an IDsHandler object

        Parameters
        ----------
        length : int, optional
            number of digit to be used in the ID, by default 10
        prefix : str, optional
            prefix to be used, by default 'ID'
        sep : str, optional
            separator to be used between prefix and digits, by default '_'

        Returns
        -------
        IDsHandler
            IDsHandler object
        """
        self.cpt = 1  # Counter of ID (first generated will be 1)
        self.length = length  # Length of the number part of the ID
        self.prefix = prefix  # Prefixe of each ID
        self.sep = sep  # Separator between prefix and number parts

    def make_new_id(self) -> str:
        """Return a new ID and update the counter.

        Returns
        -------
        str
            a new ID
        """
        number_part = "0" * (self.length - len(str(self.cpt))) + str(self.cpt)
        new_id = self.prefix + self.sep + number_part
        self.cpt = self.cpt + 1
        return new_id


class Compound(object):
    """Class handling info on compounds.

    The key information is the SMILES representation of the compound,
    i.e. the SMILES should be used in order to distinct compounds.
    """

    @classmethod
    def init_id_handler(cls) -> None:
        """Init ID handler"""
        # Class attribute that handle the IDs of compounds
        cls.ids_handler = IDsHandler(length=10, prefix='CMPD')

    def __init__(self, smiles: str) -> Compound:
        """Build a Compound object

        Parameters
        ----------
        smiles : str
            SMILES depiction

        Returns
        -------
        Compound
            a Compound object
        """
        self.cids = []
        self.uid = self.ids_handler.make_new_id()
        self.smiles = smiles
        self.original_smiles = smiles
        self.inchi = None
        self.inchikey = None
        self.is_sink = False
        self.is_target = False

    def recompute_structures(self) -> None:
        """Recompute SMILES, InChI and InChIKey of compounds
        """
        mol = Chem.MolFromSmiles(self.smiles)
        self.smiles = Chem.MolToSmiles(mol)
        self.inchi = Chem.MolToInchi(mol)
        self.inchikey = Chem.MolToInchiKey(mol)

    def set_is_sink(self, is_sink: bool) -> None:
        """Set weither a compound is in sink

        Parameters
        ----------
        is_sink : bool
            True if compound is in sink
        """
        self.is_sink = is_sink

    def add_cid(self, cid: str) -> None:
        """Add an ID to the Compound

        Parameters
        ----------
        cid : str
            ID to be added
        """
        if cid not in self.cids:
            # Remove unwanted characters from compound ID
            cid = cid.replace(",", "_")
            cid = cid.replace(":", "_")
            cid = cid.replace(" ", "_")
            cid = cid.replace("[", "_")
            cid = cid.replace("]", "_")
            self.cids.append(cid)

    def get_cids(self) -> list(str):
        """Return a set of (equivalent) compound ID(s).

        Returns
        -------
        list(str)
            List of IDs associated to this compound.

        If the real compound ID is not known, then the uniq internal
        ID is returned.
        """
        if len(self.cids) != 0:
            return self.sort_cids(self.cids)  # This is a list
        return [self.uid]

    @staticmethod
    def sort_cids(cids: list(str)) -> list(str):
        """Sort compound IDs

        Parameters
        ----------
        cids : list
            list of IDs to be sorted

        Returns
        -------
        list
            sorted IDs

        A special case is handle if the compound IDs are all using
        the "MNXM" prefix (coming from MetaNetX).
        """
        # Check wether all IDs are coming from MetaNetX (MNXM prefix)
        mnx_case = True
        for cid in cids:
            if not cid.startswith('MNXM'):
                mnx_case = False
                break
        # Sort IDs
        if mnx_case:
            return sorted(cids, key=lambda x: int(x[4:]))
        else:
            return sorted(cids)

    def set_uid(self, new_uid: str) -> None:
        """Change the value of the unique ID."""
        self.uid = new_uid
    
    def set_is_target(self, value: bool) -> None:
        self.is_target = value


class Transformation(object):
    """Handle information on a one transformation."""

    # This class attribute will (i) contains all the compounds object
    #   and (ii) will be shared between all the instance objects of the class
    compounds = {}
    smiles_to_compound = {}

    @classmethod
    def set_compounds(cls, compounds: dict, smiles_to_compound: dict) -> None:
        """Store available compounds

        Parameters
        ----------
        compounds : dict
            dict of compounds {cid: Compounds}
        smiles_to_compound : dict
            dict of association {smiles: cid}
        """
        cls.compounds = compounds
        cls.smiles_to_compound = smiles_to_compound

    @classmethod
    def cmpd_to_str(cls, uid: str, coeff: int) -> str:
        """Return a string representation of a compound in a reaction.

        Parameters
        ----------
        uid : str
            compound ID
        coeff : int
            compound coefficient

        Returns
        -------
        str
            string representation
        """
        cids = cls.compounds[uid].get_cids()
        return str(coeff) + '.[' + ','.join(cids) + ']'

    @classmethod
    def __compounds_in_reaction_side(cls, side_smiles: str) -> dict:
        """Parse on side of a reaction SMILES outputted by RetroPath2.0

        Parameters
        ----------
        side_smiles : str
            reaction SMILES of one side of the reaction

        Returns
        -------
        dict
            dictionnary of involved compounds {smiles: coeff}
        """
        items = {}
        for smi in side_smiles.split('.'):
            uid = cls.smiles_to_compound[smi]
            if uid not in items:
                items[uid] = 0
            items[uid] += 1
        return items

    @staticmethod
    def __canonize_reaction_smiles(rxn_smiles: str) -> str:
        """Canonize a SMILES reaction.

        Parameters
        ----------
        rxn_smiles : str
            reaction SMILES

        Returns
        -------
        str
            reaction SMILES with canonized SMILES compound order
        
        Canonization if performed by ordering compound SMILES in 
        the alphabetical order.
        """
        left, right = rxn_smiles.split('>>')
        lsmiles = left.split('.')
        rsmiles = right.split('.')
        return '.'.join(sorted(lsmiles)) + '>>' + '.'.join(sorted(rsmiles))

    def __init__(self, row: dict, reverse: bool=False) -> Transformation:
        """Build a Transformation object

        Parameters
        ----------
        row : dict
            dictionnary of rows as outputted by RetroPath2.0
        reverse: bool, optional
            True to consider the reaction is the reverse (right
            to left) direction. Default to False.

        Returns
        -------
        Transformation
            Transformation object
        """
        self.trs_id = row['Transformation ID']
        if reverse:
            rsmiles = Transformation.__reverse_reaction(row['Reaction SMILES'])
        else:
            rsmiles = row['Reaction SMILES']
        self.rxn_smiles = Transformation.__canonize_reaction_smiles(rsmiles)
        # Get involved compounds
        left_side, right_side = self.rxn_smiles.split('>>')
        self.left_uids = Transformation.__compounds_in_reaction_side(left_side)
        self.right_uids = Transformation.__compounds_in_reaction_side(right_side)
        # ..
        self.diameter = row['Diameter']
        self.rule_ids = row['Rule ID'].lstrip('[').rstrip(']').split(', ')
        self.ec_numbers = row['EC number'].lstrip('[').rstrip(']').split(', ')
        self.rule_score = row['Score']
        self.iteration = row['Iteration']

    @staticmethod
    def __reverse_reaction(rsmiles: str) -> str:
        """Reverse the direction of a reaction SMILES

        Parameters
        ----------
        rsmiles : str
            reaction SMILES to be reversed

        Returns
        -------
        str
            reversed reaction SMILES
        """
        left, right = rsmiles.split('>>')
        return f'{right}>>{left}'


    def to_str(self, reverse=False) -> str:
        """Returns a string representation of the Transformation

        Parameters
        ----------
        reverse : bool, optional
            should the reaction considered in the reverse direction, by default False

        Returns
        -------
        str
            the string representation
        """
        # Prepare left & right
        left_side = ':'.join(sorted([Transformation.cmpd_to_str(uid, coeff) for uid, coeff in self.left_uids.items()]))
        right_side = ':'.join(sorted([Transformation.cmpd_to_str(uid, coeff) for uid, coeff in self.right_uids.items()]))
        # ..
        ls = list()
        if not reverse:
            ls += [self.trs_id]  # Transformation ID
            ls += [','.join(sorted(list(set(self.rule_ids))))]  # Rule IDs
            ls += [left_side]
            ls += ['=']
            ls += [right_side]
        else:
            ls += [self.trs_id]  # Transformation ID
            ls += [','.join(sorted(list(set(self.rule_ids))))]  # Rule IDs
            ls += [right_side]
            ls += ['=']
            ls += [left_side]
        return '\t'.join(ls)


class RP2parser:
    """Helper to parse results from RetroPath2.0
    """

    def __init__(
        self,
        infile: str,
        cmpdfile: str='compounds.csv',
        rxnfile: str='reactions.csv',
        sinkfile: str='sinks.csv',
        reverse: bool=False
        ):
        """Parse the output from RetroPath2.0

        Parameters
        ----------
        infile : str
            file path to RetroPath2.0 results
        cmpdfile : str, optional
            compound file name to be outputted, by default 'compounds.csv'
        rxnfile : str, optional
            reaction file name to be outputted, by default 'reactions.csv'
        sinkfile : str, optional
            sink file named to be outputted, by default 'sinks.csv'
        reverse : bool, optional
            should we consider the reaction in the reverse direction, by default False
        """

        # Store
        self.compounds = {}
        self.transformations = {}
        smiles_to_compound = {}

        # Some implementation trick
        Compound.init_id_handler()

        # Get content
        content = dict()
        with open(infile, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # Skip if we are in a "header"
                if row['Initial source'] == 'Initial source':
                    continue
                # Regroup by transformation ID
                tid = row['Transformation ID']
                if tid not in content.keys():
                    content[tid] = [row]
                else:
                    content[tid].append(row)

        # 1) Check consistency and 2) Populate compounds
        compounds = dict()
        for tid in sorted(content.keys()):  # Order determine CMPD IDs
            first = True
            for row in content[tid]:
                # ..
                if first:
                    first = False
                    # Parse the Reaction SMILES
                    tmp = row['Reaction SMILES'].split('>>')
                    left_cmpds_from_rxn = set(tmp[0].split('.'))
                    right_cmpds_from_rxn = set(tmp[1].split('.'))
                    # Prep for parsing Substrate and Product columns
                    left_cmpds_from_cmpd = set()
                    right_cmpds_from_cmpd = set()
                # Accumulate compounds from Substrate and Product columns
                left_cmpds_from_cmpd.add(row['Substrate SMILES'])
                right_cmpds_from_cmpd.add(row['Product SMILES'])
            # Double check that each SMILES substrate/product is also present
            #   in the description of the reaction
            try:
                assert left_cmpds_from_rxn == left_cmpds_from_cmpd
            except AssertionError:
                print('Assertion error: differences in substrates')
                print(tid)
                print(left_cmpds_from_rxn, left_cmpds_from_cmpd)
                sys.exit(0)
            try:
                assert right_cmpds_from_rxn == right_cmpds_from_cmpd
            except BaseException:
                print('Assertion error: differences in products')
                print(tid)
                print(right_cmpds_from_rxn, right_cmpds_from_cmpd)
                sys.exit(0)
            # Populate
            for smi in sorted(list(left_cmpds_from_rxn | right_cmpds_from_rxn)):
                if smi not in smiles_to_compound.keys():
                    cmpd = Compound(smi)
                    compounds[cmpd.uid] = cmpd
                    smiles_to_compound[smi] = cmpd.uid

        # 3) Annotate sink
        for tid, rows in content.items():
            for row in rows:
                if row['In Sink'] == '1':
                    cids = row['Sink name'].lstrip('[').rstrip(']').split(', ')
                    smi = row['Product SMILES']
                    uid = smiles_to_compound[smi]
                    cmpd = compounds[uid]
                    for cid in cids:
                        compounds[cmpd.uid].add_cid(cid)
                    compounds[cmpd.uid].set_is_sink(True)

        # 4) Annotate target
        target_ids_handler = IDsHandler(length=10, prefix='TARGET')
        target_visited = set()
        for tid, rows in content.items():
            if rows[0]['Iteration'] == '0':
                smi = rows[0]['Substrate SMILES']
                old_uid = smiles_to_compound[smi]
                if old_uid not in target_visited:
                    target_uid = target_ids_handler.make_new_id()
                    target_visited.add(target_uid)
                    smiles_to_compound[smi] = target_uid
                    cmpd = compounds.pop(old_uid)
                    cmpd.set_uid(target_uid)
                    cmpd.set_is_target(True)
                    compounds[target_uid] = cmpd
        
        # 5) Make accessible compounds information from Transformation objects
        Transformation.set_compounds(compounds, smiles_to_compound)

        # 6) Populate transformations
        transformations = dict()
        for tid, rows in content.items():
            trs = Transformation(rows[0])
            transformations[tid] = trs
        
        # 7) Refine structures !! Should be done after transformations
        #    because transformation is based on reaction SMILES
        for cmpd in compounds.values():
            cmpd.recompute_structures()

        # Store compounds and transformations
        self.compounds = compounds
        self.transformations = transformations


class CacheHelper:
    """Helper to use cached info
    """

    def __init__(self):
        """Helper to use cached info
        """
        self.cache = rrCache(['rr_reactions'])
    
    def get_template_reaction(self, rule_id: str) -> list(str):
        """Get template reaction IDs associated to a given reaction rule

        Parameters
        ----------
        rule_id : str
            reaction rule ID

        Returns
        -------
        list(str):
            list of reaction template IDs
            
        """
        return [rid for rid in self.cache.get_reaction_rule(rule_id)]


class RetroGraph:
    """Handling a retrosynthesis graph.
    """

    def __init__(self, compounds: dict, transformations: dict) -> RetroGraph:
        """Store a retrosynthesis graph using networkx digraph

        Parameters
        ----------
        compounds : dict
            dictionnary of Compound objects {id: Compound}
        transformations : dict
            dictionnary of Transformation objects (id: Transformation)

        Returns
        -------
        RetroGraph
            RetroGraph object
        """
        self.__network = nx.DiGraph()
        self._add_compounds(compounds)
        self._add_transformations(transformations)
        self._make_edge_ids()
    
    def keep_source_to_sink(self, to_skip: list(str)=[], target_id=[]) -> None:
        """Keep only nodes and edges linking source to sinks

        Parameters
        ----------
        to_skip : list, optional
            InChI depictions to skip, ie to filter out in the
            outputted graph. This might be typically useful to 
            filter out cofactors. By default []
        target_id : str
            Target ID to consider as the target node, by default []

        If structure structures as given then, those structure are skipped. 
        """
        sink_ids = self._get_sinks()
        cofactor_ids = self._get_nodes_matching_inchis(to_skip)
        nodes_to_keep = []
        undirected_network = self.__network.to_undirected()
        logging.info('Starting to prune network...')
        logging.info('Source to sink paths:')
        for sink_id in sink_ids:
            logging.info(f'|- Sink ID: {sink_id}')
            try:
                for path in nx.all_shortest_paths(undirected_network, sink_id, target_id):
                    logging.info(f'|  |- path: {path}')
                    if not bool(set(path) & set(cofactor_ids)):  #  bool(a & b) returns True if overlap exists
                        logging.info(f'|  |  |-> ACCEPTED')
                        for node_id in path:
                            nodes_to_keep.append(node_id)
                    else:
                        logging.info(f'|  |  |-> REJECTED')
            except nx.NetworkXNoPath:
                pass
        all_nodes = self.__network.nodes()
        nodes_to_remove = set(all_nodes) - set(nodes_to_keep)
        self.__network.remove_nodes_from(nodes_to_remove)

    def refine(self) -> None:
        """Generate SVG depictions, add template reaction IDs.
        """
        self._add_svg_depiction()
        self._add_template_rxn_ids()

    def _add_compounds(self, compounds: dict) -> None:
        """Add compounds

        Parameters
        ----------
        compounds : dict
            dictionnary of Compound objects
        """
        for compound in sorted(
                compounds.values(),
                key=lambda x: x.get_cids()
            ):
            node = {
                'id': compound.uid,
                'type': 'chemical',
                'smiles': compound.smiles,
                'inchi': compound.inchi,
                'inchikey': compound.inchikey,
                'sink_chemical': compound.is_sink,
                'target_chemical': compound.is_target
            }
            if len(compound.get_cids()) > 0:
                node['label'] = compound.get_cids()[0]
                node['all_labels'] = list(compound.get_cids())
            else:
                node['label'] = [compound.uid]
                node['all_labels'] = [compound.uid]
            self.__network.add_nodes_from([(compound.uid, node)])
    
    def _add_transformations(self, transformations: dict) -> None:
        """Add transformations

        Parameters
        ----------
        transformations : dict
            dictionnary of Transformation objects
        """
        for transform in sorted(
                transformations.values(),
                key=lambda x: x.trs_id
            ):
            # Store the reaction itself
            node = {
                'id': transform.trs_id,
                'type': 'reaction',
                'rsmiles': transform.rxn_smiles,
                'diameter': transform.diameter,
                'rule_ids': transform.rule_ids,
                'rule_score': transform.rule_score,
                'ec_numbers': transform.ec_numbers,
                'iteration': transform.iteration
            }
            if len(transform.ec_numbers) > 0:
                node['label'] = transform.ec_numbers[0]
                node['all_labels'] = transform.ec_numbers
            else:
                node['label'] = [transform.trs_id]
                node['all_labels'] = [transform.trs_id]
            self.__network.add_nodes_from([(transform.trs_id, node)])
            # Link to substrates and products
            for compound_uid, coeff in transform.left_uids.items():
                self.__network.add_edge(
                    compound_uid,
                    transform.trs_id,
                    coeff=coeff)
            for compound_uid, coeff in transform.right_uids.items():
                self.__network.add_edge(
                    transform.trs_id,
                    compound_uid,
                    coeff=coeff)

    def _make_edge_ids(self) -> None:
        """Make edge IDs
        """
        for source_id, target_id, edge_data in self.__network.edges(data=True):
            self.__network.edges[source_id, target_id]['id'] = source_id + '_=>_' + target_id

    def _add_svg_depiction(self) -> None:
        """Add SVG depiction of chemicals
        """
        for nid, node in self.__network.nodes(data=True):
            if node['type'] == 'chemical':
                try:
                    mol = Chem.MolFromInchi(node['inchi'])
                    Compute2DCoords(mol)
                    drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()
                    svg_draft = drawer.GetDrawingText().replace("svg:", "")
                    svg = 'data:image/svg+xml;charset=utf-8,' + urllib.parse.quote(svg_draft)
                    node['svg'] = svg
                except BaseException as e:
                    node['svg'] = None
                    msg = f"SVG depiction failed from inchi: {node['inchi']}"
                    logging.warning(msg)
                    raise e
    
    def _add_template_rxn_ids(self) -> None:
        """Add template reaction IDs according to rule IDs
        """
        cache_helper = CacheHelper()
        for nid, node in self.__network.nodes(data=True):
            if node['type'] == 'reaction':
                reaction_ids = []
                for rule_id in node['rule_ids']:
                    reaction_ids += cache_helper.get_template_reaction(rule_id)
                node['rxn_template_ids'] = sorted(list(set(reaction_ids)))
    
    def _get_sinks(self) -> list(str):
        """Get the list of sink compounds

        Returns
        -------
        list(str)
            list of sink given by their node / compound IDs
        """
        node_ids = []
        for nid, node in self.__network.nodes(data=True):
            if 'sink_chemical' in node and node['sink_chemical'] == 1:
                node_ids.append(nid)
        return node_ids
    
    def _get_nodes_matching_inchis(self, inchis:list(str)) -> list(str):
        """Get the list of compound matching any inchi of the list

        Parameters
        ----------
        self : list(str)
            list of InChIs

        Returns
        -------
        list(str)
            compound / node IDs
        """
        answer = []
        for nid, node in self.__network.nodes(data=True):
            if 'inchi' in node \
                and node['inchi'] in inchis \
                and nid not in answer:
                answer.append(nid)
        return answer

    def to_cytoscape_export(self) -> str:
        """Export as a cytoscape.js compliant JSON string

        Returns
        -------
        str
            JSON string representation
        """
        cyjs = nx.cytoscape_data(self.__network, name='label', ident='id')
        json_str = json.dumps({'elements': cyjs['elements']}, indent=2)
        return json_str


if __name__ == "__main__":
    pass