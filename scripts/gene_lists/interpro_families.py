# %%
import pandas as pd
import json 

# %%
swissprot_labels = pd.read_csv('../data/labels/gene_families/swissprot_interpro.tsv',sep='\t',usecols=['Entry','InterPro'])
swissprot_labels['InterPro'] = swissprot_labels.InterPro.str.split(';')
swissprot_labels = swissprot_labels.explode('InterPro')

# %%
interpro_children = pd.read_csv('../data/labels/gene_families/interpro_children.txt',sep='::',header=None)
interpro_children = interpro_children[[0,1]]
interpro_parents = interpro_children[~interpro_children[0].str.startswith('--')]
interpro_parents.columns = ['InterPro','InterPro_name']

# %%
import re
interpro_childs = interpro_children[interpro_children[0].str.startswith('--')][0]
interpro_childs = interpro_childs.apply(lambda x: re.sub('--','',x))
interpro_parents = interpro_parents[~interpro_parents.InterPro.isin(interpro_childs.values)]

# %%
interpro_entries = pd.read_csv('../data/labels/gene_families/entry.list',sep='\t')
interpro_entries.columns  = ['InterPro','entry_type','entry_name']
interpro_domains = interpro_entries[interpro_entries.entry_type=='Domain']
interpro_domains_toplevel = interpro_parents.merge(interpro_domains[['InterPro']])

# %%
interpro_domains_toplevel

# %%
swissprot_labels_toplevel = swissprot_labels.merge(interpro_domains_toplevel).groupby('Entry').aggregate(list).reset_index()
swissprot_labels_toplevel['n_toplevel_domains'] = swissprot_labels_toplevel.InterPro.map(len)
swissprot_labels_toplevel['interpro_domain_ids'] = swissprot_labels_toplevel.InterPro.map(json.dumps)
swissprot_labels_toplevel['interpro_domain_names'] = swissprot_labels_toplevel.InterPro_name.map(json.dumps)
swissprot_labels_toplevel = swissprot_labels_toplevel[['Entry','n_toplevel_domains',
       'interpro_domain_ids', 'interpro_domain_names']]

# %%
swissprot_labels_toplevel

# %%
# add genes
swissprot_labels_genes = pd.read_csv('../data/labels/gene_families/swissprot_interpro.tsv',sep='\t',usecols=['Entry','Gene Names','Length'])
swissprot_labels_toplevel = swissprot_labels_toplevel.merge(swissprot_labels_genes)
swissprot_labels_toplevel = swissprot_labels_toplevel.rename(columns = {
    'Entry':'swissprot_acc',
    'Gene Names':'gene_names',
    'Length':'length'
})

swissprot_labels_toplevel.to_csv('../data/labels/gene_families/swissprot_interpro_toplevel.tsv',sep='\t')

# %%
swissprot_labels_toplevel.n_toplevel_domains.value_counts()

# %%
gene_params = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_gnomad.tsv',sep='\t',usecols=['gene','cds_length','num_coding_exons','exp_lof','exp_mis_pphen'])

# %%
all_swissprot_genes = swissprot_labels_toplevel.gene_names.str.split(' ').explode().values
gene_params.gene.isin(all_swissprot_genes).mean()

# %%
n_domains_gene = swissprot_labels_toplevel[['gene_names','n_toplevel_domains','length']].copy()
n_domains_gene['gene_names'] = n_domains_gene.gene_names.str.split(' ')
n_domains_gene = n_domains_gene.explode('gene_names')
gene_params_n_domains = gene_params.merge(n_domains_gene, left_on='gene',right_on='gene_names',how='left')

# %%
gene_params_n_domains[~gene_params_n_domains.n_toplevel_domains.isna()]

# %%
sns.boxplot(
    data = gene_params_n_domains,
    x = 'n_toplevel_domains',
    y = 'length'
)

# %%
sns.violinplot(
    data = gene_params_n_domains,
    x = 'n_toplevel_domains',
    y = 'num_coding_exons', cut=0
)

# %%
import seaborn as sns

sns.boxplot(
    data = gene_params_n_domains,
    x = 'n_toplevel_domains',
    y = 'exp_lof'
)

# %%
sns.boxplot(
    data = gene_params_n_domains,
    x = 'n_toplevel_domains',
    y = 'exp_mis_pphen'
)

# %%
gene_params_n_domains['exp_lof_norm'] = gene_params_n_domains.exp_lof / gene_params_n_domains.length
sns.boxplot(
    data = gene_params_n_domains,
    x = 'n_toplevel_domains',
    y = 'exp_lof_norm'
)

# %%
gene_params_n_domains['exp_mis_pphen_norm'] = gene_params_n_domains.exp_mis_pphen / gene_params_n_domains.length
sns.boxplot(
    data = gene_params_n_domains,
    x = 'n_toplevel_domains',
    y = 'exp_mis_pphen_norm'
)

# %%
import matplotlib.pyplot as plt
gene_params_n_domains['num_coding_exons_norm'] = gene_params_n_domains.num_coding_exons / gene_params_n_domains.length
sns.violinplot(
    data = gene_params_n_domains,
    x = 'n_toplevel_domains',
    y = 'num_coding_exons_norm', cut =0
)
plt.xlabel('Number of top-level domains (InterPro)')
plt.ylabel('Number of coding exons / Coding sequence length')

# %%
from scipy import stats
(rows, cols), tab = stats.contingency.crosstab(
    gene_params_n_domains.n_toplevel_domains==1,
    gene_params_n_domains.num_coding_exons==1
)
print(tab)
print(tab/ tab.sum(axis=0))
print(stats.chi2_contingency(tab))

# %%
gpcrs = pd.read_csv('../data/labels/gene_families/gpcr_genes_human_gpcrdb.tsv',sep='\t')
gene_params_n_domains['is_gpcr'] = gene_params_n_domains.gene.isin(gpcrs.gene_gnomad)
gene_params_n_domains['gpcr_n_domains'] = \
    gene_params_n_domains['is_gpcr'].astype(str) + '_' +\
    (gene_params_n_domains.n_toplevel_domains == 1).astype(str) 

(rows, cols), tab = stats.contingency.crosstab(
    gene_params_n_domains[~gene_params_n_domains.n_toplevel_domains.isna()].num_coding_exons==1,
    gene_params_n_domains[~gene_params_n_domains.n_toplevel_domains.isna()].gpcr_n_domains
)
print(cols)
print(tab)
print( tab.sum(axis=0))
print(tab/ tab.sum(axis=0))

# %%
gpcrs = pd.read_csv('../data/labels/gene_families/gpcr_genes_human_gpcrdb.tsv',sep='\t')
gene_params_n_domains['is_gpcr'] = gene_params_n_domains.gene.isin(gpcrs.gene_gnomad)
gene_params_n_domains['gpcr_n_domains'] = \
    gene_params_n_domains['is_gpcr'].astype(str) + '_' +\
    (gene_params_n_domains.n_toplevel_domains == 1).astype(str) + '_' +\
    (~gene_params_n_domains.n_toplevel_domains.isna()).astype(str)

(rows, cols), tab = stats.contingency.crosstab(
    gene_params_n_domains.num_coding_exons==1,
    gene_params_n_domains.gpcr_n_domains
)
print(cols)
print(tab)
print( tab.sum(axis=0))
print(tab/ tab.sum(axis=0))


