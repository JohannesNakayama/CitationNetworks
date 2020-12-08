#%%

# import modules
import pandas as pd
import uuid
import re
import math
import networkx as nx
import json

#%%

class Bibliography:
    """
    a bibliography containing bibliographic data
    """

    def __init__(self, path_list, db_src):
        self.bib_df = _read_bib(path_list, db_src)
        self.bib_dict = _convert_bib_to_dict(self.bib_df, db_src)
        self.cit_rel = _extract_cit_rel(self.bib_dict)
    
    def export_bib_dict(self, path):
        """
        write bibliographic dictionary to a json file
        """
        with open(path, 'w') as json_file:
            json.dump(self.bib_dict, json_file)

#%%

class CitationNetwork:
    """
    a citation network in the form of a weighted DAG
    """

    def __init__(self, BibObject, keywords, w_method):
        self.keywords = keywords
        self.cit_net = _create_citation_network(BibObject, keywords, w_method)
        self.out_deg_dist = _extract_out_deg_dist(self.cit_net)

    def add_enclosing(self):
        """
        add sink and source to citation network
        """
        self.cit_net.add_edges_from(_enclose_cit_rel(self.cit_net))

#%%

class MainPathNetwork:
    """
    a network of main paths constructed from a citation network
    """

    def __init__(self, CitNetObject, mc=1, iterations=1):
        self.cit_net_raw = CitNetObject
        self.main_paths, self.main_net = _main_path_analysis(CitNetObject, mc, iterations)
        self.main_path_net = nx.DiGraph(kw=CitNetObject.keywords)

    def create_network(self, mw=1):
        """
        create a main path network
        contains all edges that are part of a main path
        network is pruned with some options
        """
        self.main_path_net.add_weighted_edges_from(self.main_net)
        _rm_list_e = []
        for e in self.main_path_net.edges:
            if self.main_path_net.get_edge_data(e[0], e[1])['weight'] < mw:
                _rm_list_e.append(e)
        self.main_path_net.remove_edges_from(_rm_list_e)
        _rm_list_n = []
        for n in self.main_path_net.nodes:
            if (self.main_path_net.in_degree(n) == 0 and self.main_path_net.out_degree(n) == 0):
                _rm_list_n.append(n)
        self.main_path_net.remove_nodes_from(_rm_list_n)
  
#%%

def _read_bib(path_list, db_src):
    """
    read output data from bibliographic database 
    args:
        path_list: a list of paths to read from
        db_src: database source of the bibliographic data
    returns:
        bib_df: a bibliographic dataframe
    """

    # get database specifics
    db_specs = _switch_case_db(db_src)

    # read file(s)
    if len(path_list) == 1:
        bib_df = pd.read_csv(
            path_list[0], usecols=db_specs['columns'], 
            dtype=db_specs['dtype'], sep=db_specs['sep'], 
            index_col=False
        )
    else:
        bib_df = pd.concat(
            [
                pd.read_csv(
                    path, usecols=db_specs['columns'], 
                    dtype=db_specs['dtype'], sep=db_specs['sep'], 
                    index_col=False
                ) for path in path_list
            ],
            ignore_index=True
        )  
    
    # some more formatting
    bib_df['db_src'] = db_src
    bib_df = bib_df.rename(columns=db_specs['new_cols'])
    
    return bib_df

#%%

def _convert_bib_to_dict(bib_df, db_src):
    """
    convert bibliographic dataframe into python dictionary
    args:
        bib_df: a bibliographic dataframe
        db_src: database source of the bibliographic data
    returns:
        bib_dict: a bibliographic dictionary
    """
    
    # get database specifics
    db_specs = _switch_case_db(db_src)

    # extract and reformat data 
    keys = bib_df.columns
    bib_dict = {}

    # create a dictionary entries
    for j in range(bib_df.shape[0]):
        entry = {}
        for key in keys:
            entry[key] = bib_df.loc[j, key]
        entry = _split_columns(entry, db_specs['to_split'])  # split non-atomic columns
        bib_dict[str(uuid.uuid4())] = entry

    return bib_dict

#%%

# switch scopus specifics
def _switch_scopus():
    """
    in case db_src='scopus', use these specifications
    args:
        None
    returns:
        wos: the scopus specifics
    """

    # define scopus specifics
    scopus = {
        'columns': [
            'Title', 'Authors', 'Author(s) ID', 'Source title', 
            'Year', 'Volume', 'Issue', 'Cited by', 
            'References', 'DOI', 'ISBN', 'ISSN', 
            'CODEN', 'PubMed ID', 'EID'
        ],
        'dtype': {
            'Title': 'str', 'Authors': 'str', 
            'Author(s) ID': 'str', 'Source title': 'str', 
            'Year': 'float64', 'Volume': 'str',
            'Issue': 'str', 'Cited by': 'float64', 
            'References': 'str', 'DOI': 'str', 
            'ISBN': 'str', 'ISSN': 'str', 
            'CODEN': 'str', 'PubMed ID': 'str', 
            'EID': 'str'            
        },
        'sep': ',',
        'new_cols': {
            'Title': 'title', 'Authors': 'authors', 
            'Author(s) ID': 'scopus_author_id', 'Source title': 'source', 
            'Year': 'year', 'Volume': 'vol', 
            'Issue': 'issue', 'Cited by': 'cit_count', 
            'References': 'refs', 'DOI': 'doi', 
            'ISBN': 'isbn', 'ISSN': 'issn', 
            'CODEN': 'coden', 'PubMed ID': 'pmid', 
            'EID': 'eid'
        },
        'to_split': {
            'authors': ',', 'scopus_author_id': ';', 
            'refs': ';', 'isbn': ';'
        }
    }

    return scopus

#%%

# switch web of science specifics
def _switch_wos():
    """
    in case db_src='wos', use these specifications
    args:
        None
    returns:
        wos: the wos specifics
    """

    # define web of science specifics
    wos = {
        'columns': [
            'TI', 'AU', 'AF', 'RI', 'OI', 'SO', 'PY', 'VL',
            'IS', 'TC', 'Z9', 'CR', 'U1', 'U2', 'DI', 'D2', 
            'BN', 'SN', 'PM', 'J9', 'JI', 'BP', 'EP'
        ],
        'dtype': {
            'TI': 'str', 'AU': 'str', 'AF': 'str', 'RI': 'str', 
            'OI': 'str', 'SO': 'str', 'PY': 'float64', 'VL': 'str',
            'IS': 'str', 'TC': 'float64', 'Z9': 'float64', 'CR': 'str',
            'U1': 'float64', 'U2': 'float64', 'DI': 'str', 'D2': 'str', 
            'BN': 'str', 'SN': 'str', 'PM': 'str', 'J9': 'str', 
            'JI': 'str', 'BP': 'str', 'EP': 'str'
        },
        'sep': '\t',
        'new_cols': {
            'TI': 'title', 'AU': 'authors', 'AF': 'authors_full', 
            'RI': 'researcher_id', 'OI': 'orcid', 'SO': 'source', 
            'PY': 'year', 'VL': 'vol', 'IS': 'issue', 
            'TC': 'cit_count', 'Z9': 'cit_count_z9', 'CR': 'refs',
            'U1': 'usage_count_180d', 'U2': 'usage_count_2013', 'DI': 'doi', 
            'D2': 'book_doi', 'BN': 'isbn', 'SN': 'issn',
            'PM': 'pmid', 'J9': 'src_abb_29', 'JI': 'src_abb_iso', 
            'BP': 'start_page', 'EP': 'end_page'
        },
        'to_split': {
            'authors': ';', 'authors_full': ';', 'researcher_id': ';', 
            'orcid': ';', 'refs': ';', 'isbn': ';'
        }
    }

    return wos

#%%

# switch case replacement
# adapted from: https://www.pydanny.com/why-doesnt-python-have-switch-case.html
def _switch_case_db(arg):
    """
    replacement for switch case 
    args:
        arg: the case to execute
    returns:
        func(): the switch function
    """
    
    # dictionary referring to switch statements
    switcher = {
        'scopus': _switch_scopus,
        'wos': _switch_wos
    }

    # print status if exception occurs
    if arg not in ['scopus', 'wos']:
        print(
            'Expection: Unknown db_source ' + 
            str(arg) + 
            '\nOutput might not be in proper format'
        )

    # switch function
    func = switcher.get(arg, lambda: [[], {}])

    return func() 

#%%

# split non-atomic columns
def _split_columns(entry, split_list):
    """
    split pre-defined dictionary entries along pre-defined separators
    args:
        entry: a bibliography entry
        split_list: a pre-defined list of columns to separate
    returns:
        entry: the split entry
    """

    # function to strip trailing spaces
    space_stripper = lambda x: x.rstrip().lstrip()

    # iterate over pre-defined split_list
    for label, separator in split_list.items():
        try:
            entry[label] = space_stripper(entry[label]).rstrip(';')
            entry[label] = entry[label].split(separator)
            entry[label] = list(
                map(space_stripper, entry[label])
            )
        except:
            continue
    
    return entry 

#%%

def _extract_cit_rel(bib_dict):
    """
    extract the citation relation 
    args:
        bib_dict: a bibliographic dictionary
    returns:
        cit_rel: a list of tuples (x, y) with 'x cites y'
    """

    # initialize citation relation
    cit_rel = []
    
    # iterate over all bibliography entries
    for key in bib_dict.keys():
        doi_at_key = bib_dict[key]['doi']
        # check if a doi is available, if not: continue
        if len(str(doi_at_key)) > 8:
            pass
        else:
            continue 
        refs_at_key = bib_dict[key]['refs']
        # try to extract doi and append to citation relation
        try:
            for ref_idx in range(len(refs_at_key)):
                ref_doi = _extract_doi(
                    refs_at_key[ref_idx], bib_dict[key]['db_src']
                )
                if ref_doi == 'NO_DOI':
                    continue
                if ref_doi != doi_at_key:
                    cit_rel.append((ref_doi, doi_at_key))
        except:
            continue  
    
    return cit_rel

#%%

# extract doi from reference string
# with a little help from: https://www.crossref.org/blog/dois-and-matching-regular-expressions/
def _extract_doi(ref_elem, db_src='wos'):
    """
    extract doi from reference (CAUTION: only works for db_src == 'wos' so far!)
    args:
        ref_elem: a reference
        db_src: database source of the bibliographic data
    returns:
        doi if doi is extractable
        'NO_DOI' if no doi is extractable
    """

    # currently: only works for web of science
    if db_src == 'wos':

        regex_doi = re.compile('10.\d{4,9}/[-._;()/:A-Za-z0-9]+$')  # define regex for doi
        
        # try searching for doi in reference string
        try:    
            res = regex_doi.search(ref_elem)
            doi = res.group(0)
            return doi
        # if not successful: return value for missing doi
        except:
            return 'NO_DOI'
    # <CODE FOR SCOPUS HERE> -> elif db_src == 'scopus': <CODE>
    else:
        return 'NO_DOI'

#%%

#%%

def _create_citation_network(BibObject, keywords=['citation network'], w_method='nppc'):
    """
    create a citation network
    args:
        BibObject: Bibliography object
        keywords: list of keywords as graph attributes (for later reference)
        w_method: weighting method (so far only 'nppc')
    returns:
        net: a citation network
    """
    
    # create directed graph
    net = nx.DiGraph(kw=keywords)
    net.add_edges_from(BibObject.cit_rel)
    net = _break_cycles(net)
    net = _compute_edge_weights(net, w_method)

    return net

#%%

def _compute_edge_weights(net, method='nppc'):
    """
    compute the edge weights of a citation network
    args:
        net: a citation network
        method: a weighting method (so far only 'nppc')
    returns:
        net: the net with weighted edges
    """

    if method == 'nppc':  # node pair projection count
        # extract all pairs of connected nodes
        con_nodes = _find_connected_nodes(net)
        # compute nppc weights
        for sub in [nx.all_simple_paths(
                net, source=pair[0], target=pair[1]
            ) for pair in con_nodes]:
            tmp = net.subgraph({node for path in sub for node in path})
            for edge in tmp.edges:
                try:
                    net[edge[0]][edge[1]]['weight'] += 1
                except:
                    net[edge[0]][edge[1]]['weight'] = 1
    else:
        print('This does not seem to be a valid weighting method.')
    
    return net

#%%

def _find_connected_nodes(net):
    """
    find all connected nodes (a, b) where there is a path from a to b
    args:
        net: a citation network
    returns:
        con_nodes: a list of tuples with all connected nodes
    """

    con_nodes = [
        (struc[0], i) 
        for node in net.nodes 
        for struc in nx.bfs_successors(net, node) 
        for i in struc[1]
    ]

    return(con_nodes)

#%%

def _extract_out_deg_dist(net):
    """
    extract the distribution of out-degrees in the citation network for later pruning
    args:
        net: a citation graph
    returns:
        a pandas Dataframe with the frequency distribution of out-degrees
    """

    out_deg_list = [net.out_degree(node) for node in net.nodes]
    freq = [0] * (max(out_deg_list) + 1)
    for i in out_deg_list:
        freq[i] += 1
    data = {
        'out_degree': list(range(max(out_deg_list) + 1)),
        'freq': freq
    }

    return(pd.DataFrame(data))

#%%

def _enclose_cit_rel(net):
    """
    add sink and source to a citation network
    args:
        net: an unweighted citation network
    return:
        enclosing: list of edges to add sink and source
    """

    try:
        source_edges = [('source', node, {'weight': 1}) for node in net.nodes if net.in_degree(node) == 0]
        sink_edges = [(node, 'sink', {'weight': 1}) for node in net.nodes if net.out_degree(node) == 0]
        return(source_edges + sink_edges)
    except:
        return(False)

#%%

def _break_cycles(net):
    """
    breaks potential cycles in a citation net
    args:
        net: a citation net
    returns:
        net: the acyclic citation net
    """

    flag = True
    counter = 0
    while(flag):
        try:
            counter += 1
            cycles = nx.find_cycle(net)
            net.remove_edge(cycles[-1][0], cycles[-1][1])
        except:
            if counter == 1:
                print('NOTE: {} edge was removed to break cycles'.format(counter))
            else:
                print('NOTE: {} edges were removed to break cycles'.format(counter))
            flag = False
        if counter >= 100:
            print('NOTE: There are oddly many cycles. Please check your data to avoid problems in the further process.')
            flag = False

    return(net)



#%% 

def _main_path_analysis(CitNetObject, mc, iterations):
    """
    conduct main path analysis on a citation network
    args:
        CitNetObject: a CitationNetwork object
        mc: minimum citations for start nodes
        iterations: number of iterations to conduct
    returns:
        main_paths: a list of all mined main paths
        main_net: the edges of all main paths in one list
    """

    # setup
    CitNetObject.cit_net.remove_nodes_from(['source', 'sink'])  
    print('NOTE: source and sink have been removed from your citation network')
    init_net = CitNetObject.cit_net

    # iterations
    for i in range(iterations):

        # initialize
        start_nodes = _retrieve_start_nodes(init_net, mc)
        main_paths = []
        main_net = []

        for node in start_nodes:

            # loop setup
            counter = 0  # if paths cross lengths of 100, the while loop is interrupted
            flag = True
            error = False
            mp = []
            cur_nodes = [node]

            # find main path
            while(flag):

                # create list containing all lists of out_edges of the current nodes 
                candidates = [
                    [
                        (e[0], e[1], init_net.edges[e[0], e[1]]['weight']) 
                        for e in init_net.out_edges(cur_node)
                    ] for cur_node in cur_nodes
                ]

                # extract weights from candidate edges
                weights = [[t[2] for t in l] for l in candidates]
                
                # determine maximum and prepare next iteration
                mw = [max(w) for w in weights]
                max_idx = [
                    [
                        i for i, j in enumerate(weights[m]) if j == mw[m]
                    ] for m in range(len(mw))
                ]

                # update current nodes and extend current main path
                cur_nodes.clear()
                for i, mi in enumerate(max_idx):
                    next_edges = [candidates[i][j] for j in mi]
                    mp.extend(next_edges)
                    cur_nodes.extend([e[1] for e in next_edges])
                cur_nodes = list(dict.fromkeys(cur_nodes))

                # remove node from current nodes if end of path is reached
                rm_idx = []
                for i in range(len(cur_nodes)):
                    if init_net.out_degree(cur_nodes[i]) == 0:
                        rm_idx.append(i)
                for idx in sorted(rm_idx, reverse=True):
                    del cur_nodes[idx]

                counter += 1

                # stop criteria
                if not cur_nodes:
                    flag = False
                if counter >= 100:
                    print('This takes oddly long. Something must have gone wrong.')
                    error = True
                    flag = False
            
            # append extracted main path to main path collection and extend main path network
            mp = list(dict.fromkeys(mp))
            main_paths.append(mp)
            main_net.extend(mp)

            # report potential error
            if error:
                print('An error occurred.')
                break
            
        init_net = nx.DiGraph()
        init_net.add_weighted_edges_from(main_net)

    return(main_paths, main_net)

#%%

def _retrieve_start_nodes(net, mc):
    """
    retrieve nodes with in_degree == 0 as starting points for main path analysis
    args:
        net: a CitationNetwork object
        mc: minimum citations
    returns:
        list of start nodes in net
    """
    return [node for node in net.nodes if (net.in_degree(node) == 0 and net.out_degree(node) > mc)]
