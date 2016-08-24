
# coding: utf-8

# # Proof-of-concept queries for Project Hetmech
# 
# Based on a highly customized _CbGiG_ metapath (_Compound-binds-Gene-interacts-Gene_).

# In[1]:

from time import perf_counter

import pandas
from sklearn.metrics import roc_auc_score
from neo4j.v1 import GraphDatabase


# ## Connect to the [Hetionet v1.0 Neo4j instance](https://neo4j.het.io)

# In[2]:

driver = GraphDatabase.driver("bolt://neo4j.het.io")
assert driver.encrypted


# In[3]:

def run_query(query, parameters={}):
    """Execute a Cypher query and return results as a pandas.DataFrame"""
    start_time = perf_counter()
    with driver.session() as session:
        result = session.run(query, parameters)
        result_df = pandas.DataFrame((x.values() for x in result), columns=result.keys())
    runtime = perf_counter() - start_time
    return result_df, runtime


# ## Identify candidate treatments
# 
# The following query was prototyped on [GitHub Issues](https://github.com/greenelab/hetmech/issues/6#issuecomment-236287256). It looks for compound-disease pairs where:
# 
# + the compound treats the disease
# + the disease has at least one GWAS-associated gene
# + the compound binds 2 or more genes
# + the compound binds a gene that interacts with a gene that is disease-associated
# + the compound does not bind a gene that is disease-associated (filter the obvious)

# In[4]:

query = """
MATCH path = (compound:Compound)-[TREATS_CtD]-(disease:Disease)
WHERE
  exists((disease)-[:ASSOCIATES_DaG {unbiased: true}]-())
  AND size((compound)-[:BINDS_CbG]-()) > 1
  AND exists((compound)-[:BINDS_CbG]-()-[:INTERACTS_GiG]-()-[:ASSOCIATES_DaG {unbiased: true}]-(disease))
  AND NOT exists((compound)-[:BINDS_CbG]-()-[:ASSOCIATES_DaG {unbiased: true}]-(disease))
RETURN compound.name AS compound, disease.name AS disease
ORDER BY compound, disease
"""
treatment_df, query_time = run_query(query)
print('Neo4j query took {:.2f} seconds'.format(query_time))
treatment_df.tail(3)


# ## Infer the mechanistic targets for each treatment
# 
# For each treatment (compound-disease pair), we evaluate each target (gene that the compound binds) for whether it's the therapeutic mechanism. We base our inferrence of therapeutic mechanism by looking for targets which interact with GWAS-associated genes of the treated disease. We rank targets based on a precision assumption: the target with the greatest percentage of associations in its set of interacting genes (`precision`) is ranked highest.
# 
# Here are some definitions:
# 
# + positives: the set of GWAS-associated genes for the disease
# + `TP + FP`: the set of interacting genes for the target that is bound by the compound
# + `TP` (true positives): the set of GWAS-associated genes that interact with the target 
# + `FP` (false positives): the set of GWAS-unassociated genes that interact with the target
# + `precision`: the percent of genes interacting with the target that are disease-associated
# 
# The output table contains the following columns for whether the potential mechanistic target has a relationship with the disease:
# 
# + associated_target: the target gene is associated with the disease
# + upregulated_target: the target gene is upregulated by the disease
# + downregulated_target: the target gene is downregulated by the disease

# In[5]:

query = """
MATCH path = (compound:Compound)-[:BINDS_CbG]-(gene_1)-[:INTERACTS_GiG]-(gene_2)
  WHERE compound.name = { compound }
OPTIONAL MATCH (gene_2)-[association:ASSOCIATES_DaG {unbiased: true}]-(disease:Disease)
  WHERE disease.name = { disease }
WITH
  gene_1, gene_2, association
  ORDER BY gene_2.name
WITH
  gene_1,
  extract(x in collect(gene_2) | x.name) AS interactors,
  extract(x in collect(association) | endNode(x).name) AS associated_interactors
WITH
  gene_1,
  interactors, associated_interactors,
  size(associated_interactors) AS TP,
  size(interactors) AS `TP + FP`
RETURN
  { compound } AS compound,
  { disease } AS disease,
  gene_1.name AS target,
  associated_interactors,
  interactors,
  TP, `TP + FP`,
  toFloat(TP) / `TP + FP` AS precision,
  exists((gene_1)-[:ASSOCIATES_DaG]-(:Disease {name: { disease }})) AS associated_target,
  exists((gene_1)-[:UPREGULATES_DuG]-(:Disease {name: { disease }})) AS upregulated_target,
  exists((gene_1)-[:DOWNREGULATES_DdG]-(:Disease {name: { disease }})) AS downregulated_target
ORDER BY precision DESC, `TP + FP`, target
"""

dfs = list()
query_time = 0
for i, parameters in treatment_df.iterrows():
    parameters = dict(parameters)
    df, runtime = run_query(query, parameters)
    dfs.append(df)
    query_time += runtime

print('Neo4j queries took {:.2f} seconds'.format(query_time))

mechanism_df = pandas.concat(dfs)
for column in 'interactors', 'associated_interactors':
    mechanism_df[column] = mechanism_df[column].str.join(', ')

outcomes = 'associated_target', 'downregulated_target', 'upregulated_target'
for outcome in outcomes:
    mechanism_df[outcome] = mechanism_df[outcome].astype(int)


# In[6]:

# Number of treatment-target pairs
len(mechanism_df)


# In[7]:

mechanism_df.iloc[: , :8].head()


# In[8]:

mechanism_df.to_csv('CbGiG-candidates.tsv', index=False, sep='\t', float_format='%.3g')


# ## Do mechanisms correspond to known disease-genes?
# 
# This section computes an average AUROC across all compound-disease pairs. The AUROC measures the ability of the `precision` score to identify disease-related targets.

# In[9]:

def get_auroc(df):
    series = pandas.Series()
    for outcome in outcomes:
        y_true = df[outcome]
        if y_true.nunique() != 2:
            series[outcome] = 0.5
        else:
            series[outcome] = roc_auc_score(y_true, df.precision)
    return series
    
auroc_df = mechanism_df.groupby(['compound', 'disease']).apply(get_auroc).reset_index()


# In[10]:

auroc_df.mean()

