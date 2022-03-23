"""Convert output of the RetroPath2.0 workflow.

Copyright (C) 2016-2017 Thomas Duigou, JL Faulon's research group, INRA

Use of this source code is governed by the MIT license that can be found in the
LICENSE.txt file.

"""

from __future__ import annotations

import copy
import csv
import json
import logging
import sys
import urllib
from math import ceil

import networkx as nx
from rdkit import Chem
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
from rr_cache import rrCache
from rxn_rebuild import rebuild_rxn


def canonize_smiles(smiles: str) -> str:
    """Canonize a SMILES

    Parameters
    ----------
    smiles : str
        SMILES to be canonized

    Returns
    -------
    str
        SMILES
    """
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        return smiles


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

    def __init__(self, smiles: str = None, uid: str = None) -> Compound:
        """Build a Compound object

        Parameters
        ----------
        smiles : str
            SMILES depiction
        uid: str
            Compound unique ID

        Returns
        -------
        Compound
            a Compound object
        """

        self.cids = []

        # Handle unique ID
        if uid is None:
            self.uid = self.ids_handler.make_new_id()
        else:
            self.uid = uid
        
        # Set SMILES if any provided
        if smiles is not None:
            self.smiles = smiles
            self.original_smiles = smiles
        else:
            self.smiles = None
            self.original_smiles = None
        
        # Other attributes
        self.inchi = None
        self.inchikey = None
        self.is_sink = False
        self.is_target = False

    def compute_structures(
        self,
        smiles: bool=True,
        inchi: bool=True,
        inchikey: bool=True
        ) -> None:
        """Recompute SMILES, InChI and InChIKey of compounds

        Parameters
        ----------
        smiles : bool
            Compute smiles depiction, by default True
        inchi: bool
            Compute inchi depiction, by default True
        inchikey: bool
            Compute inchikey depiction, by default True
        """
        mol = Chem.MolFromSmiles(self.smiles)
        if smiles:
            self.smiles = Chem.MolToSmiles(mol)
        if inchi:
            self.inchi = Chem.MolToInchi(mol)
        if inchikey:
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
        # If some IDs are coming from MetaNetX (MNXM prefix)
        #   they are place in first positions
        mnx_ids = []
        other_ids = []
        for cid in cids:
            if cid.startswith('MNXM'):
                mnx_ids.append(cid)
            else:
                other_ids.append(cid)
        
        # Sort independantly each list
        mnx_ids = sorted(mnx_ids, key=lambda x: int(x[4:]))
        other_ids = sorted(other_ids)

        # MNXM IDs are placed first
        return mnx_ids + other_ids

    def set_uid(self, new_uid: str) -> None:
        """Change the value of the unique ID."""
        self.uid = new_uid
    
    def set_is_target(self, value: bool) -> None:
        self.is_target = value
    
    def get_smiles(self) -> str:
        """Return the SMILES depiction

        Returns
        -------
        str
            SMILES
        """
        return self.smiles


class Transformation(object):
    """Handle information on a one transformation."""

    # This class attribute will (i) contains all the compounds object
    #   and (ii) will be shared between all the instance objects of the class
    compounds = {}
    smiles_to_compound = {}
    cache = None

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
    def set_cache(cls) -> None:
        """Set up the cache to be used for completing the reactions"""
        cls.cache = rrCache()

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
    def complete_reactions(cls, trs: Transformation) -> list:
        """Complete a Transformation

        The Transformation is completed by adding the omitted cosubstrates
        and cofactors. From a given uncompleted Transformation, several
        completed Transformations could be outputted. This is the case if 
        several couples of cofactors are possible, eg ATP/ADP or GTP/GDP.

        Parameters
        ----------
        trs : Transformation
            Transformation to be completed.

        Returns
        -------
        list
            List of completed Transformations.
        """
        def sanitize_coeff(coeff):
            return ceil(coeff)

        cache = cls.cache
        cache_helper = CacheHelper(cls.cache)

        # Will store all the completed transformations
        completed_transformations = {}

        # Build the ID based description of the reaction
        transfo_str = ''
        side = []
        for cid, coeff in trs.left_uids.items():
            side += [cid for _ in range(coeff)]
        transfo_str = '+'.join(side)
        side = []
        for cid, coeff in trs.right_uids.items():
            side += [cid for _ in range(coeff)]
        transfo_str += '=' + '+'.join(side)

        # Iterate over possible rule IDs
        for rule_id in trs.rule_ids:

            # Get template reaction associated to current rule ID
            template_rxn_ids = cache_helper.get_template_reactions(rule_id)

            # Iterate over template reactions
            for tmpl_rxn_id in template_rxn_ids:

                # Info are stored in a new Transformation object
                trs_child = copy.deepcopy(trs)
                trs_child.rule_ids = [rule_id]
                trs_child.template_rxn_ids = [tmpl_rxn_id]

                # Make a new ID for the completed transformation
                max_i = 9999
                for i in range (max_i):
                    _ = f'{trs.trs_id}_{tmpl_rxn_id}_{i}'
                    if _ not in completed_transformations:
                        trs_child.trs_id = _
                    if i == max_i:
                        raise AssertionError(f'Maximum number of child reached from transformation {trs}.')

                # Get completion info
                rxn_info = rebuild_rxn(
                    rxn_rule_id=rule_id,
                    transfo=transfo_str,
                    tmpl_rxn_id=tmpl_rxn_id,
                    direction='reverse',
                    cache=cache)

                # Parse the list of left and right compound IDs
                trs_child.left_uids = rxn_info[tmpl_rxn_id]['full_transfo']['left']
                trs_child.right_uids = rxn_info[tmpl_rxn_id]['full_transfo']['right']
                
                # Sanitize coeff (could be float, we want integer)
                for cid in trs_child.left_uids:
                    trs_child.left_uids[cid] = sanitize_coeff(trs_child.left_uids[cid])
                for cid in trs_child.right_uids:
                    trs_child.right_uids[cid] = sanitize_coeff(trs_child.right_uids[cid])

                # DEBUG
                left_uids_debug = trs_child.left_uids
                right_uids_debug = trs_child.right_uids
                
                # Collect info on added compounds
                reaction_cmpd_info = {}
                for side in ('left', 'left_nostruct', 'right', 'right_nostruct'):
                    for _key, _dict in rxn_info[tmpl_rxn_id]['added_cmpds'][side].items():
                        reaction_cmpd_info[_dict['cid']] = _dict

                # Check if these compounds are already known according to their SMILES
                for uid, _dict_info in reaction_cmpd_info.items():
                    
                    # Skip if uid is already in the list of known compounds
                    if uid in cls.compounds:
                        continue

                    # Try to produce a canonized SMILES
                    try:
                        smi = canonize_smiles(_dict_info['smiles'])
                    except KeyError:
                        smi = None
                    
                    if smi is None:
                        # No way we got a match, we just add info on these compound
                        cls.compounds[uid] = Compound(smiles=None, uid=uid)
                    
                    elif smi in cls.smiles_to_compound:

                        # Here we have a match, we replace and update the left / right list of IDs
                        ori_uid = cls.smiles_to_compound[smi]
                        if uid in trs_child.left_uids:
                            if ori_uid not in trs_child.left_uids:
                                trs_child.left_uids[ori_uid] = 0
                            trs_child.left_uids[ori_uid] += trs_child.left_uids[uid]
                            del(trs_child.left_uids[uid])
                        if uid in trs_child.right_uids:
                            if ori_uid not in trs_child.right_uids:
                                trs_child.right_uids[ori_uid] = 0
                            trs_child.right_uids[ori_uid] += trs_child.right_uids[uid]
                            del(trs_child.right_uids[uid])
                    
                    else:
                        # Otherwise this is a new compound
                        cmpd = Compound(smiles=smi, uid=uid)
                        cmpd.compute_structures()
                        cls.compounds[uid] = cmpd
                        cls.smiles_to_compound[smi] = uid
                
                # Now let's update the reaction SMILES
                trs_child.__set_reaction_smiles_from_compound_ids()
                assert trs_child.trs_id not in completed_transformations
                completed_transformations[trs_child.trs_id] = trs_child

                # # DEBUG
                # print(f"FULL TRANSFO: {rxn_info[tmpl_rxn_id]['full_transfo']}")
                # print(f"TRS         : {trs_child.to_str()}")
                # print(f"RULE ID     : {trs_child.rule_ids}")
                # print(f"LEFT  BEFORE: {left_uids_debug}")
                # print(f"LEFT  AFTER : {trs_child.left_uids}")
                # print(f"RIGHT BEFORE: {right_uids_debug}")
                # print(f"RIGHT AFTER : {trs_child.right_uids}")

        return completed_transformations

    def __set_compound_ids_from_reaction_smiles(self, rsmiles: str) -> int:
        """Set the compound IDs from the reaction SMILES

        Parameters
        ----------
        rsmiles : str
            reaction SMILES

        Returns
        -------
        int
            the number of distinct compounds extracted
        """
        def cids_in_side(side_smiles: str) -> dict:
            items = {}
            for smi in side_smiles.split('.'):
                uid = self.smiles_to_compound[canonize_smiles(smi)]
                if uid not in items:
                    items[uid] = 0
                items[uid] += 1
            return items
        left_smiles, right_smiles = rsmiles.split('>>')
        self.left_uids = cids_in_side(left_smiles)
        self.right_uids = cids_in_side(right_smiles)
        return len(set(self.left_uids) | set(self.right_uids))

    def __set_reaction_smiles_from_compound_ids(self) -> None:
        """Build the reaction SMILES from compound IDs
        """
        def gen_smiles_side(compounds: dict) -> str:
            side = []
            for uid, coeff in compounds.items():
                if uid in self.compounds and self.compounds[uid].get_smiles() is not None:
                    side += [self.compounds[uid].get_smiles() for _ in range(coeff)]
                else:
                    pass
            return '.'.join(sorted(side))
        self.rxn_smiles = '>>'.join([
            gen_smiles_side(self.left_uids),
            gen_smiles_side(self.right_uids)
            ])

    def __init__(self, row: dict) -> Transformation:
        """Build a Transformation object

        Parameters
        ----------
        row : dict
            dictionnary of rows as outputted by RetroPath2.0

        Returns
        -------
        Transformation
            Transformation object
        """
        self.trs_id = row['Transformation ID']
        self.diameter = row['Diameter']
        self.rule_ids = row['Rule ID'].lstrip('[').rstrip(']').split(', ')
        self.ec_numbers = row['EC number'].lstrip('[').rstrip(']').split(', ')
        self.rule_score = row['Score']
        self.iteration = row['Iteration']

        # To be filled later
        self.left_uids = {}
        self.right_uids = {}
        self.rxn_smiles = ''

        # Get involved compounds from SMILES
        self.__set_compound_ids_from_reaction_smiles(row['Reaction SMILES'])

        # Re-build reaction SMILES from UIDS
        self.__set_reaction_smiles_from_compound_ids()

    def to_str(self) -> str:
        """Returns a string representation of the Transformation

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
        ls += [self.trs_id]  # Transformation ID
        ls += [','.join(sorted(list(set(self.rule_ids))))]  # Rule IDs
        ls += [left_side]
        ls += ['=']
        ls += [right_side]
        return '\t'.join(ls)


class RP2parser:
    """Helper to parse results from RetroPath2.0
    """

    def __init__(
        self,
        res_file: str,
        sink_file: str,
        cmpdfile: str='compounds.csv',
        rxnfile: str='reactions.csv',
        sinkfile: str='sinks.csv',
        reverse: bool=False
        ):
        """Parse the output from RetroPath2.0

        Parameters
        ----------
        res_file : str
            file path to RetroPath2.0 results
        sink_file : str
            file path to the sink file used by RetroPath2.0
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

        # Get results content
        content = dict()
        with open(res_file, 'r') as fh:
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
                # Get canonize version of SMILES
                can_smi = canonize_smiles(smi)
                if can_smi not in smiles_to_compound.keys():
                    cmpd = Compound(can_smi)
                    cmpd.compute_structures(smiles=False)
                    compounds[cmpd.uid] = cmpd
                    smiles_to_compound[can_smi] = cmpd.uid
        
        # 3) Annotate target
        target_ids_handler = IDsHandler(length=10, prefix='TARGET')
        target_visited = set()
        for tid, rows in content.items():
            if rows[0]['Iteration'] == '0':
                smi = rows[0]['Substrate SMILES']
                can_smi = canonize_smiles(smi)
                old_uid = smiles_to_compound[can_smi]
                if old_uid not in target_visited:
                    target_uid = target_ids_handler.make_new_id()
                    target_visited.add(target_uid)
                    smiles_to_compound[can_smi] = target_uid
                    cmpd = compounds.pop(old_uid)
                    cmpd.set_uid(target_uid)
                    cmpd.set_is_target(True)
                    compounds[target_uid] = cmpd

        # 4) Make accessible compounds and cache information from Transformation objects
        Transformation.set_compounds(compounds, smiles_to_compound)
        Transformation.set_cache()

        # 5) Populate transformations
        transformations = dict()
        for tid, rows in content.items():
            trs = Transformation(rows[0])
            # Complete transformatoins
            completed_transformations = Transformation.complete_reactions(trs)
            transformations.update(completed_transformations)

        # 6) Annotate sink
        # At first, deduce sink compounds from the annotations
        #    in the result file
        for tid, rows in content.items():
            for row in rows:
                if row['In Sink'] == '1':
                    cids = row['Sink name'].lstrip('[').rstrip(']').split(', ')
                    smi = row['Product SMILES']
                    can_smi = canonize_smiles(smi)
                    uid = smiles_to_compound[can_smi]
                    cmpd = compounds[uid]
                    for cid in cids:
                        compounds[cmpd.uid].add_cid(cid)
                    compounds[cmpd.uid].set_is_sink(True)
        # Because a result can have no "sink" annotation
        #    we also directly extract sink information from
        #    sink file
        # But, we still want to keep the part above, 
        #    because the matching criteria (SMILES vs inchi)
        #    are different. Belt AND suspenders...
        sinks = {}
        with open(sink_file) as fh:
            # We don't use dictreader because labels
            #    "name" and "inchi" can be upper, lower, 
            #    or mixed case. Index is more simple.
            #    row[0] is the compound name
            #    row[1] is the inchi
            reader = csv.reader(fh)
            for row in reader:
                # Skip the header
                if row[0].lower().startswith("name"):
                    continue
                name, inchi = row[0], row[1]
                for cid, cmpd in compounds.items():
                    if cmpd.inchi == inchi:
                        cmpd.add_cid(name)
                        cmpd.set_is_sink(True)

        # Store compounds and transformations
        self.compounds = compounds
        self.transformations = transformations


class CacheHelper:
    """Helper to use cached info
    """

    def __init__(self, cache=None) -> CacheHelper:
        """Helper to use cached info
        """
        if cache is None:
            self.cache = rrCache(['rr_reactions'])
        else:
            self.cache = cache
    
    def get_template_reactions(self, rule_id: str) -> list(str):
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
        self._merge_similar_reactions()

    def keep_source_to_sink(self, to_skip: list(str)=[], target_id=[]) -> int:
        """Keep only nodes and edges linking source to sinks

        Parameters
        ----------
        to_skip : list, optional
            InChI depictions to skip, ie to filter out in the
            outputted graph. This might be typically useful to 
            filter out cofactors. By default []
        target_id : str
            Target ID to consider as the target node, by default []
        
        Returns
        -------
            The number of kept source to sink paths

        If 'to_skip' structures as given then, those structures are skipped. 
        """
        sink_ids = self._get_sinks()
        cofactor_ids = self._get_nodes_matching_inchis(to_skip)
        
        nodes_to_keep = []
        edges_to_keep = []
        nb_paths = 0
        
        logging.info('Starting to prune network...')
        logging.info('Source to sink paths:')


        for sink_id in sink_ids:
            logging.info(f'|- Sink ID: {sink_id}')
            try:
                for path in nx.all_shortest_paths(self.__network, target_id, sink_id):
                    logging.info(f'|  |- path: {path}')
                    if not bool(set(path) & set(cofactor_ids)):  #  bool(a & b) returns True if overlap exists
                        logging.info(f'|  |  |-> ACCEPTED')
                        nb_paths += 0
                        for node_id in path:
                            nodes_to_keep.append(node_id)
                        for i in range(len(path)-1):  # Keep only edge linkig to node of interest
                            source_node = path[i]
                            target_node = path[i+1]
                            edges_to_keep.append((source_node, target_node))
                    else:
                        logging.info(f'|  |  |-> REJECTED')
            except nx.NetworkXNoPath:
                pass
        
        # Only keep nodes of interest
        all_nodes = self.__network.nodes()
        nodes_to_remove = set(all_nodes) - set(nodes_to_keep)
        self.__network.remove_nodes_from(nodes_to_remove)

        # Only keep edges of interest
        all_edges = self.__network.edges()
        edges_to_remove = set(all_edges) - set(edges_to_keep)
        self.__network.remove_edges_from(edges_to_remove)

        # Number of kept paths
        return nb_paths

    def _merge_similar_reactions(self) -> None:
        """Merge together reactions having similar information.
        
        Path sharing the following information are considered similar:
            - reaction SMILES
            - iteration
        
        The following rules are applied for the other values:
            - EC numbers: union
            - reaction template IDs: union
            - rule score: best (the lower the better)
            - rule IDs: union
            - diameter: biggest
        """
        # Collect ID of reactions
        net = self.__network  # shorcut
        rnodes_ids = []
        for nid, data in net.nodes(data=True):
            if data['type'] == 'reaction':
                rnodes_ids.append(nid)

        # Compare 2 by 2
        to_merge = {}
        for i in range(0, len(rnodes_ids)-1):
            for j in range(i+1, len(rnodes_ids)):
                inode = net.nodes[rnodes_ids[i]]
                jnode = net.nodes[rnodes_ids[j]]
                if (
                    inode['rsmiles'] == jnode['rsmiles']
                    and inode['iteration'] == jnode['iteration']
                ):
                    if inode['rsmiles'] not in to_merge:
                        to_merge[inode['rsmiles']] = set()
                    to_merge[inode['rsmiles']] |= set([inode['id'], jnode['id']])

        # Merge
        for smi, node_ids in to_merge.items():
            ref_node_id = node_ids.pop()
            for other_node_id in node_ids:
                self.__merge_two_reaction_nodes(ref_node_id, other_node_id)
                net.remove_node(other_node_id)

    def __merge_two_reaction_nodes(self, ref_node_id: str, other_node_id: str):
        """Merge information from one reaction to node into a reference node.

        Information of the "other_node" are merge into the "ref_node". Nothing
        is returned.

        Parameters
        ----------
        ref_node_id : str
            reference reaction node ID
        other_node_id : [type]
            other reaction node ID
        """
        n1 = self.__network.nodes[ref_node_id]
        n2 = self.__network.nodes[other_node_id]

        try:
            n1['diameter'] = max(int(n1['diameter']), int(n2['diameter']))
        except Exception as e:
            logging.warning(e)

        try:
            n1['rule_ids'] = list(set(n1['rule_ids']) | set(n2['rule_ids']))
        except Exception as e:
            logging.warning(e)

        try:
            n1['rule_score'] = min(float(n1['rule_score']), float(n2['rule_score']))
        except Exception as e:
            logging.warning(e)
        
        try:
            n1['ec_numbers'] = list(set(n1['ec_numbers']) | set(n2['ec_numbers']))
        except Exception as e:
            logging.warning(e)
        
        try:
            n1['rxn_template_ids'] = list(set(n1['rxn_template_ids']) | set(n2['rxn_template_ids']))
        except Exception as e:
            logging.warning(e)

    def _refine_reaction_labels(self):
        """Build the reaction labels.

        Reaction labels are made from EC numbers if any exist, otherwise
        from the node id.
        """
        for nid, node in self.__network.nodes(data=True):
            if node["type"] == "reaction":                
                if len(node.get("ec_numbers", [])) > 0:
                    node["label"] = node["ec_numbers"][0]
                    node["all_labels"] = node["ec_numbers"]
                else:
                    node["label"] = nid
                    node["all_labels"] = nid

    def refine(self) -> None:
        """Perform graph refinements.
        
        Merge similar reactions, SVG depictions, update reaction labels, ...
        """
        self._refine_svg_depictions()
        self._refine_reaction_labels()

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
                'iteration': transform.iteration,
                'rxn_template_ids': sorted(list(set(transform.template_rxn_ids))),
                "label" : "",
                "all_labels" : []
            }
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

    def _refine_svg_depictions(self) -> None:
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
            if (
                'inchi' in node
                and node["inchi"] is not None
                and any([inchi.startswith(node["inchi"]) for inchi in inchis])
                and nid not in answer
            ):
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
