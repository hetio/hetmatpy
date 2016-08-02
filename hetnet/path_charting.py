
# coding: utf-8

# In[1]:

import numpy as np
import scipy as sp
import networkx as nx
import sys
import os
import scipy.sparse

# SETUP GRAPH
fname = './adjacency/data-big/adjacency.tsv'
with open(fname, 'rb') as graphfile:
    next(graphfile) # skip header row
    G=nx.read_weighted_edgelist(graphfile, delimiter='\t')
print('Done reading in graph')
A = nx.adjacency_matrix(G, nodelist=None, weight=None)
print('Done converting to adjacency matrix')
nodelist = G.nodes()
Gene2Index = {}
Index2Gene = {}
GeneIndexList = []
Compound2Index = {}
Index2Compound = {}
CompoundIndexList = []
for index, item in enumerate(nodelist):
    #item =
    if "Gene::" in item:
        Gene2Index[item] = index
        Index2Gene[index] = item
        GeneIndexList.append(index)
    if "Compound::" in item:
        Compound2Index[item] = index
        Index2Compound[index] = item
        CompoundIndexList.append(index)
# Check for the expected number of compounds
assert len(Compound2Index) == 1538
assert len(Gene2Index) == 18875
print('Graph loaded and checked.')


# In[9]:


# Used to locate all nodes with a path length between lb and ub
def find_indices( array, lb, ub ):
	indices = []
	for j, value in enumerate(array):
		if value >= lb and value <= ub:
			indices.append(j)
	return indices

# To count number of node-pairs with number of paths inside specified range
def count_paths( P2, P3, path_lb, path_ub ):
	path2_lb, path3_lb = path_lb
	path2_ub, path3_ub = path_ub
	rows2, cols2, vals2 = scipy.sparse.find(P2)
	rows3, cols3, vals3 = scipy.sparse.find(P3)
	indices_of_min = find_indices( vals2, path2_lb, path2_ub )
	rows_min = [ rows2[j] for j in indices_of_min ]
	cols_min = [ cols2[j] for j in indices_of_min ]
	pairs_path2 = zip( rows_min, cols_min )
	indices_of_min = find_indices( vals3, path3_lb, path3_ub )
	rows_min = [ rows3[j] for j in indices_of_min ]
	cols_min = [ cols3[j] for j in indices_of_min ]
	pairs_path3 = zip( rows_min, cols_min )
	pairs_of_interest = set(pairs_path2).intersection(pairs_path3)
	print('There are ' + str( len(pairs_of_interest) ) + ' pairs of interest.\n' )
	return pairs_of_interest


A2 = A*A
print("Done with A2 = A*A.")

# next subtract diagonal of A2
N = len( G.nodes() )
Dsp = sp.sparse.spdiags( A2.diagonal(), 0, N, N )
A2 = A2 - Dsp
del(Dsp)
print("Done subtracting the diag of A2.")

P2 = A2[:,CompoundIndexList]
del(A2)
P3 = A*P2
P2 = P2[GeneIndexList,:]
P3 = P3[GeneIndexList,:]
print("Done computing P2, P3.")



# In[ ]:

# Extract just the gene-compound pairs that have the smallest number of 2- and 3-paths connecting them.
path_LB = [10, 20]
path_UB = [100, 100]
POI = count_paths( P2, P3, path_LB, path_UB )
print("If you're satisfied with the number of node-pairs of interest,\n then call print_paths() to write the node-pairs to file,\n and call chart_paths() to write the actual paths to file.\n")

# convert from local node indices contained in POI to global indices used in G
pairs_of_interest = []
for i,j in POI:
    pairs_of_interest.append( (GeneIndexList[i], CompoundIndexList[j]) )
    
# To output list of paths
# and write list of node-pairs to file
def chart_paths(pairs_of_interest):
    path2_lb, path3_lb = path_LB
    path2_ub, path3_ub = path_UB
    fname = str(path2_lb) + '-' + str(path2_ub) + '-' + str(path3_lb) + '-' + str(path3_ub)
    with open('./path-lists/list-small-paths' + fname + '.txt','w') as outputf:
        for snode,tnode in pairs_of_interest:
            #sname, tname = ( nodelist[])
            sname, tname = nodelist[snode], nodelist[tnode]
            paths = nx.all_simple_paths(G, source=sname, target=tname, cutoff=3)
            pathlist = list(paths)
            for pathitem in pathlist:
                for node in pathitem:
                    #outputf.write( nodelist[node] + '\t' )
                    outputf.write( node + '\t' )
                outputf.write( '\n')
            outputf.write( '\n\n' )
    with open('./path-lists/list-nodepairs' + fname + '.tsv','w') as pathfile:
        # write header row
        pathfile.write( 'gene_name' + '\t' + 'compound_name' + '\n' )
        for i,j in pairs_of_interest:
            which_gene = nodelist[i]
            which_compound = nodelist[j]
            pathfile.write( which_gene + '\t' + which_compound + '\n' )


# In[7]:

chart_paths(pairs_of_interest)

