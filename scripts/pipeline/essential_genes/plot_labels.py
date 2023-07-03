# %% [markdown]
# # Setup environment

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
import seaborn as sns
from utils.roc_functions import resample, empirical_ci, plot_comparison, plot_precision_recall, plot_roc
from gpcr_mapper.plot_gpcr_mapper import plot_gpcr_mapper

from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind

# %% [markdown]
# # Check labels

# %%

gnomad_mouse_essentials = pd.read_csv('../data/labels/mouse_essential_genes/gnomad_mouse_essential_genes_hom.tsv',sep='\t', header=None)[0]
print(gnomad_mouse_essentials.shape)
gpcr_phenotypes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_phenotypes.csv')
print(gpcr_phenotypes.shape[0])
print(gpcr_phenotypes.mouse_knockout_phenotype_level.isin(['Lethal','Developmental']).sum())
gpcr_phenotypes_curated = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_genes_mgi_essential_curated.tsv',sep='\t')
print((gpcr_phenotypes_curated.level=='Lethal').sum())
all_gpcrs_phenotypes = pd.read_csv('../data/labels/mouse_essential_genes/gnomad_constraint_gpcrs_w_phenotypes.tsv',sep='\t')
print(all_gpcrs_phenotypes.shape[0])
all_gpcrs_phenotypes.mouse_knockout_phenotype_level.value_counts()

# %%
l1 = gpcr_phenotypes[gpcr_phenotypes.mouse_knockout_phenotype_level=='Lethal'].symbol
l2 = all_gpcrs_phenotypes[all_gpcrs_phenotypes.mouse_knockout_phenotype_level=='Lethal'].symbol
l1[~l1.isin(l2)]

l1 = gpcr_phenotypes.symbol
l2 = all_gpcrs_phenotypes.symbol
l1[~l1.isin(l2)]

# %%
l1 = gpcr_phenotypes_curated.symbol
l2 = gpcr_phenotypes.symbol
print(l1[~l1.isin(l2)].shape)
print(l2[~l2.isin(l1)].shape)

l1 = gpcr_phenotypes_curated[gpcr_phenotypes_curated.level=='Lethal'].symbol
l2 = gpcr_phenotypes[gpcr_phenotypes.mouse_knockout_phenotype_level.isin(['Lethal','Developmental'])].symbol
print(l1[~l1.isin(l2)].shape)
print(l2[~l2.isin(l1)].shape)

# %%
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv',sep='\t')
for i in gpcrs[~gpcrs.gene.isin(all_gpcrs_phenotypes.symbol)].gene: print(i)
print('n')
for i in all_gpcrs_phenotypes[~all_gpcrs_phenotypes.symbol.isin(gpcrs.gene)].symbol: print(i)

# %%
df = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
df[~df.gene.isin(gpcrs.gene)]
df

# %% [markdown]
# ## Convert to mouse genes and match to mouse lethal genes (Mousemine)

# %%
# library(biomaRt)
# library(data.table)




# convertMouseGeneList <- function(x){
#    human <- useMart("ensembl", dataset = "hsapiens_gene_ensembl",host= "https://dec2021.archive.ensembl.org")
#    mouse <- useMart("ensembl", dataset = "mmusculus_gene_ensembl",host= "https://dec2021.archive.ensembl.org")

#    genesV2 <- getLDS(attributes = c("mgi_symbol"), filters = "mgi_symbol", 
#                  values = x , mart = mouse, attributesL = c("hgnc_symbol"), martL = human, uniqueRows=T)

#    humanx <- unique(genesV2[, 2])

#    return(humanx)
# }

# het_lethal_mouse  <- fread('./data/gene_lists/list_mouse_het_lethal_genes.tsv',header = F)$V1
# het_lethal_human <- convertMouseGeneList(het_lethal_mouse)
# fwrite(data.table(het_lethal_human),"./data/gene_lists/mouse_het_lethal_human_genes.tsv",sep="\t",col.names = F)


# hom_lethal_mouse  <- fread('./data/gene_lists/list_mouse_lethal_genes.tsv',header = F)$V1
# hom_lethal_human <- convertMouseGeneList(hom_lethal_mouse)
# fwrite(data.table(hom_lethal_human),"./data/gene_lists/mouse_hom_lethal_human_genes.tsv",sep="\t",col.names = F)

# %% [markdown]
# ## GWAS and eQTLs

# %%
genes = pd.read_csv('../data/gene_lists/all_coding_genes.txt',sep='\t')
eQTLs= pd.read_csv('../data/eQTLs/eqtls.txt',sep='\t')


genes_with_eqtls = genes.merge(eQTLs, left_on='Ensembl_ID',right_on='eGene_Symbol')

genes_with_eqtls.to_csv('../data/eQTLs/eqtl_genes.txt',sep='\t')


gwas = pd.read_csv('../data/eQTLs/gwas.txt',sep='\t')
gwas[['Chr','Pos','Wt','Mut']]= gwas.Variant.str.split(pat=':',expand=True)
gwas['Chr'] = 'chr' + gwas.Chr
gwas['Pos'] = gwas.Pos.astype(int)
gwas.merge(genes, on='Chr')

gwas['gene'] = ''
gwas['dist'] = 0
for chr in gwas.Chr.unique():
    genes_chr = genes[genes.Chr == chr]
    gwas_chr = gwas[gwas.Chr == chr]

    distmat = cdist(
        genes_chr.TSS.values.reshape(-1,1),
        gwas_chr.Pos.values.reshape(-1,1)
    )
    gwas.loc[gwas.Chr == chr, 'dist'] = \
        distmat.min(axis=0)
    gwas.loc[gwas.Chr == chr, 'gene'] = \
        genes_chr.Name.iloc[distmat.argmin(axis=0)].values
gwas.to_csv('../data/eQTLs/gwas_genes.txt',sep='\t')

plt.hist(gwas.Trait.value_counts())

plt.hist(genes_with_eqtls.Tissue.value_counts())

# %%
constraint = pd.read_csv('../data/constraint/precalc_obs_exp.tsv',sep='\t')
genes_with_eqtls = pd.read_csv('../data/eQTLs/eqtl_genes.txt',sep='\t')
gwas = pd.read_csv('../data/eQTLs/gwas_genes.txt',sep='\t')
gpcrs = pd.read_csv('../data/gene_lists/gpcr_gene_symbols.tsv',sep='\t')

gwas_genes = gwas.gene.unique()
print("GWAS genes:", len(gwas_genes))
eqtl_genes = genes_with_eqtls.Name.unique()
print("eQTL genes:",len(eqtl_genes))

constraint['is_gwas'] = constraint.gene.isin(gwas_genes)
constraint['is_eqtl'] = constraint.gene.isin(eqtl_genes)
constraint['associations'] = np.select(
    [
        constraint.is_gwas & constraint.is_eqtl,
        constraint.is_gwas,
        constraint.is_eqtl
    ],
    ['GWAS and eQTL','GWAS only','eQTL only'],
    'Neither'
)
print('Total genes:',len(constraint))
print(constraint[constraint.exp_lof >5].associations.value_counts())
print('GPCR genes:',len(constraint[constraint.gene.isin(gpcrs.Grch37_symbol)]))
print(constraint[(constraint.exp_lof >5) & constraint.gene.isin(gpcrs.Grch37_symbol)].associations.value_counts())

# %% [markdown]
# # Load constraint metrics and annotate with labels

# %%
constraint = pd.read_csv("../data/constraint/gnomad/all_genes_constraint_exact_a0.1.tsv", sep = '\t', index_col=0)

rankscale_oeuf = lambda x: 1-x.rank(pct=True).fillna(1)
constraint['oeuf_lof_rank'] = rankscale_oeuf(constraint['oeuf_lof'])
constraint['oeuf_mis_pphen_rank'] = rankscale_oeuf(constraint['oeuf_mis_pphen'])
constraint['oeuf_mean_rank'] = (constraint.oeuf_lof_rank + constraint.oeuf_mis_pphen_rank) / 2

gnomad_mouse_essentials = pd.read_csv('../data/labels/mouse_essential_genes/gnomad_mouse_essential_genes_hom.tsv',sep='\t', header=None)[0]
constraint['mouse_essential'] = constraint.gene.isin(gnomad_mouse_essentials)

print(constraint.mouse_essential.mean())


gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
constraint_ = constraint.rename(columns = {'gene':'gene_gnomad'})
constraint_gpcrs = constraint_.merge(gpcrs, on='gene_gnomad')
print(constraint_gpcrs.mouse_essential.mean())

constraint_gpcrs_curated = constraint_gpcrs.copy()
gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_genes_mgi_essential_curated.tsv',sep='\t',index_col=0)
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.level.isin(['Lethal','Developmental'])]
constraint_gpcrs_curated['mouse_essential'] = constraint_gpcrs_curated.gene.isin(gpcr_curated_essential_genes.symbol)
print(constraint_gpcrs_curated.mouse_essential.mean())

gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.combined_phenotype.isin(['Lethal','Developmental'])]
constraint_gpcrs_curated['mouse_essential_'] = constraint_gpcrs_curated.gene.isin(gpcr_curated_essential_genes.gene)
print(constraint_gpcrs_curated.mouse_essential_.mean())

print((constraint_gpcrs_curated.mouse_essential_).sum())

# %% [markdown]
# # GPCR tree plots

# %%
constraint = pd.read_csv("../data/constraint/gnomad/all_genes_constraint_zscores.tsv", sep = '\t', index_col=0)
gnomad_mouse_essentials = pd.read_csv('../data/labels/mouse_essential_genes/gnomad_mouse_essential_genes_hom.tsv',sep='\t', header=None)[0]
constraint['mouse_essential'] = constraint.gene.isin(gnomad_mouse_essentials)

gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
constraint_gpcrs = constraint.rename(columns = {'gene':'gene_gnomad'}).merge(gpcrs)

mappings = pd.read_csv('gpcr_mapper/gpcr_treemapper_coords.txt',sep='\t')
df = constraint_gpcrs.merge(mappings)

df = df[df.mouse_essential]
df['label']  = df.mouse_essential


ax = plot_gpcr_mapper(df.x, df.y, df['z_lof'], df.label, 
    'pLoF constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.savefig('../plots/Fig3A_z_lof_essentials_treemapper_uncurated.png',dpi=450)

df['label'] = 'Essential in mice'
ax = plot_gpcr_mapper(df.x, df.y, df['z_mis_pphen'], df.label, 
    'pPM constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.savefig('../plots/Fig3B_z_mis_pphen_essentials_treemapper_uncurated.png',dpi=450)



# %%
eqtl_genes = pd.read_csv('../data/eQTLs/gpcrs_gwas_and_eqtls.txt')
associated_genes = eqtl_genes[eqtl_genes.associations=='GWAS and eQTL'].gene
phenotyped_genes['GWAS_and_eQTL'] = phenotyped_genes.symbol.isin(associated_genes)

phenotyped_genes['genetic_associations'] = np.select(
    (
        (phenotyped_genes.mouse_knockout_phenotype_level == 'Lethal') |
        (phenotyped_genes.mouse_knockout_phenotype_level == 'Developmental'),
        phenotyped_genes['GWAS_and_eQTL']
    ),
    (
        'Essential in model organisms','Disease-associated in humans'
    ),
    default='Other'
)

sns.set_context('paper')
df = phenotyped_genes[phenotyped_genes.genetic_associations!='Other'].copy()
ax = plot_gpcr_mapper(
    df.x, df.y, 
    df['oeuf|mis_pphen'], 
    df.genetic_associations, 
    'damaging missense \n obs/exp upper bound',
    markers=['s','X'],cmap='RdBu',cbar_shrink=0.5)
fig = plt.gcf()
plt.savefig('../plots/GPCR_mapper_genetic_associations.png',dpi=500)

# %%
constraint = pd.read_csv("../data/constraint/gnomad/all_genes_constraint_zscores.tsv", sep = '\t', index_col=0).drop_duplicates(subset ='gene')


gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
constraint_gpcrs = constraint.rename(columns = {'gene':'gene_gnomad'}).merge(gpcrs)

gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.combined_phenotype.isin(['Lethal','Developmental'])]
constraint_gpcrs['mouse_essential_'] = constraint_gpcrs.gene.isin(gpcr_curated_essential_genes.gene)
print(constraint_gpcrs.mouse_essential_.mean())

print(constraint_gpcrs[(constraint_gpcrs.receptor_class=='Class A (Rhodopsin)') & (constraint_gpcrs.z_mis_pphen>5)&constraint_gpcrs.mouse_essential_].shape[0])

mappings = pd.read_csv('gpcr_mapper/gpcr_treemapper_coords.txt',sep='\t')
df = constraint_gpcrs.merge(mappings)

df = df[df.mouse_essential_]
df['label']  = df.mouse_essential_


ax = plot_gpcr_mapper(df.x, df.y, df['z_lof'], df.label, 
    'pLoF constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.savefig('../plots/Fig3A_z_lof_essentials_treemapper.png',dpi=450)

df['label'] = 'Essential in mice'
ax = plot_gpcr_mapper(df.x, df.y, df['z_mis_pphen'], df.label, 
    'pPM constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.savefig('../plots/Fig3B_z_mis_pphen_essentials_treemapper.png',dpi=450)



# %%


# %% [markdown]
# # Predictive performance

# %% [markdown]
# ## All genes

# %%
y = constraint.mouse_essential
print('P =', y.mean())


for metric, label in zip(['oeuf_lof_rank', 'oeuf_mis_pphen_rank'], ['pLoF OEUF','pPM OEUF']):
    x = constraint[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,2))

plt.legend()

plt.subplots()
for metric, label in zip(['oeuf_lof_rank', 'oeuf_mis_pphen_rank'], ['pLoF OEUF','pPM OEUF']):
    x = constraint[metric].values
    ap = plot_precision_recall(y, x)
    print(round(ap,2))

plt.legend()
plt.savefig('../plots/prc_all_genes.png')

g = sns.jointplot(
    data = constraint,
    x = 'oeuf_lof',
    y = 'oeuf_mis_pphen',
    hue = y,
    xlim=(0, 3),
    ylim=(0, 3),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5},
    marginal_kws={'common_norm':False}
)

# %% [markdown]
# ## GPCRs

# %%
g = sns.jointplot(
    data = constraint_gpcrs,
    x = 'oeuf_lof',
    y = 'oeuf_mis_pphen',
    hue = 'mouse_essential_2',
    xlim=(0, 3),
    ylim=(0, 3),
    kind='scatter',
    joint_kws={'alpha':0.5},
    marginal_kws={'common_norm':False}
)

# %%
y = constraint_gpcrs.mouse_essential_2
print('P =', y.mean())


for metric, label in zip(['oeuf_lof_rank', 'oeuf_mis_pphen_rank'], ['pLoF OEUF','pPM OEUF']):
    x = constraint_gpcrs[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,2))

plt.legend()

plt.subplots()
for metric, label in zip(['oeuf_lof_rank', 'oeuf_mis_pphen_rank'], ['pLoF OEUF','pPM OEUF']):
    x = constraint_gpcrs[metric].values
    ap = plot_precision_recall(y, x)
    print(round(ap,2))

plt.legend()
plt.savefig('../plots/prc_all_genes.png')

# %% [markdown]
# # Other labels

# %% [markdown]
# ## GPCRs with curated labels

# %%
g = sns.jointplot(
    data = constraint_gpcrs_curated,
    x = 'oeuf_lof',
    y = 'oeuf_mis_pphen',
    hue = 'mouse_essential',
    xlim=(0, 3),
    ylim=(0, 3),
    kind='scatter',
    joint_kws={'alpha':0.5},
    marginal_kws={'common_norm':False}
)

# %%
y = constraint_gpcrs_curated.mouse_essential
print('P =', y.mean())

fig, ax = plt.subplots(1,2,figsize=(7,3))

for metric in ['oeuf_lof_rank', 'oeuf_mis_pphen_rank']:
    x = constraint_gpcrs_curated[metric].values
    ap = plot_precision_recall(y, x, ax= ax[0])
    auroc  = plot_roc(y, x, ax = ax[1])
    print(round(auroc,2), round(ap,2))

ax[0].legend()
ax[1].legend()
plt.savefig('../plots/prc_all_genes.png')

# %% [markdown]
# ## All genes, GWAS and eQTLs

# %%

df = constraint[(constraint.exp_lof >5) & ~constraint.oe_lof_upper.isna()]

sns.countplot(df.associations,
    order= ['GWAS and eQTL','GWAS only','eQTL only','Neither'])

plt.subplots()
sns.violinplot(
    data = df,
    x = 'associations',
    y = 'oe_lof_upper',
    cut=0,
    order= ['GWAS and eQTL','GWAS only','eQTL only','Neither']
)

plt.subplots()
precision, recall, _ = precision_recall_curve(df.associations.isin(('GWAS and eQTL','GWAS only')), -df.oe_lof_upper)
plt.plot(recall, precision)
print(average_precision_score(df.associations.isin(('GWAS and eQTL','GWAS only')), -df.oe_lof_upper))

precision, recall, _ = precision_recall_curve(df.associations.isin(('GWAS and eQTL','GWAS only')), -df.oe_lof_upper.sample(frac=1))
plt.plot(recall, precision)
print(average_precision_score(df.associations.isin(('GWAS and eQTL','GWAS only')), -df.oe_lof_upper.sample(frac=1)))

# %%
sns.countplot(df.associations,
    order= ['GWAS and eQTL','GWAS only','eQTL only','Neither'])
plt.subplots()
sns.violinplot(
    data = constraint[constraint.gene.isin(gpcrs.symbol)],
    x = 'associations',
    y = 'oe_lof_upper',
    cut=0,
    order= ['GWAS and eQTL','GWAS only','eQTL only','Neither']
)
plt.subplots()

precision, recall, _ = precision_recall_curve(df.associations == 'GWAS and eQTL', -df.oe_lof_upper)
plt.plot(recall, precision, color='#1b9e77')
precision, recall, _ = precision_recall_curve(df.associations == 'GWAS and eQTL', -df.oe_mis_pphen)
plt.plot(recall, precision,color='#d95f02')
precision, recall, _ = precision_recall_curve(df.associations == 'GWAS and eQTL', -df[['oe_mis_pphen','oe_lof_upper']].rank().median(axis=1))
plt.plot(recall, precision,color='k')
print(average_precision_score(df.associations == 'GWAS and eQTL', -df.oe_mis_pphen.sample(frac=1)))

# %% [markdown]
# ## GPCR essential genes + GWAS/eQTL genes

# %%
df.columns

# %%
df = pd.read_csv('../data/labels/eQTLs/gpcrs_gwas_and_eqtls.csv')
df = df[~df.gene.str.startswith('OR')]
df = df.drop(columns=['oe_lof_upper','oe_mis_pphen']).merge(constraint)
df['target'] = (df.gene.isin(gnomad_mouse_essentials) | (df.associations == 'GWAS and eQTL'))

fpr, tpr,_ = roc_curve(df.target, 1-df.oeuf_lof.rank(pct=True).fillna(1))
plt.plot(fpr, tpr)
fpr, tpr,_ = roc_curve(df.target, 1-df.oeuf_mis_pphen.rank(pct=True).fillna(1))
plt.plot(fpr, tpr)

plt.subplots()
precision, recall, _ = precision_recall_curve(df.target, 1-df.oeuf_lof.rank(pct=True).fillna(1))
plt.plot(recall, precision)
precision, recall, _ = precision_recall_curve(df.target, -df.oeuf_mis_pphen)
plt.plot(recall, precision)
df.shape
plt.ylim((0,1))

# %%
constraint_gpcrs.merge(df, on)

# %%
constraint_gpcrs

# %%
eqtl_genes = pd.read_csv('../data/eQTLs/gpcrs_gwas_and_eqtls.txt',index_col=0)
associated_genes = eqtl_genes[eqtl_genes.associations=='GWAS and eQTL'].gene

phenotyped_genes = pd.read_csv('../data/constraint/gpcrs_with_constraint_and_phenotypes.tsv',index_col=0,sep='\t')
phenotyped_genes['GWAS_and_eQTL'] = phenotyped_genes.symbol.isin(associated_genes)

phenotyped_genes['genetic_associations'] = np.select(
    (
        phenotyped_genes.mouse_knockout_phenotype_level == 'Lethal',
        phenotyped_genes.mouse_knockout_phenotype_level == 'Developmental',
        phenotyped_genes['GWAS_and_eQTL']
    ),
    (
        'Lethal','Developmental','GWAS'
    ),
    default='Other'
)
y_true = phenotyped_genes.genetic_associations != 'Other'

metrics = ['oeuf|lof','oeuf|mis_pphen','oeuf|mis_non_pphen']
labels = ['pLoF', 'Damaging\nmissense','Benign\nmissense']
colors = [plt.get_cmap("Dark2")(i) for i in range(3)]

sns.set_context('talk')
plt.subplots(figsize = (7.5,5))
for metric, label, color in zip(metrics,labels,colors):
    plot_precision_recall(y_true, -gpcrs_constraint[metric].rank(pct=True).fillna(1),label=label, color = color)

plt.legend(title= 'obs/exp upper bound',fontsize = 15,title_fontsize=15)
plt.ylabel('Precision (essential/associated)')
plt.xlabel('Recall (essential/associated)')
#plt.legend(loc='upper left',bbox_to_anchor=(1.05,1))
plt.savefig('../plots/prc_all_gpcrs_associated.png',bbox_inches='tight')

# %%

constraint = pd.read_csv("../data/constraint/all_genes_constraint_and_phenotypes_gnomad.csv")
constraint[constraint.is_hom_lethal | constraint.is_cell_line_nonviable].symbol

# %% [markdown]
# # Other genesets

# %% [markdown]
# ## All genes hom lethals

# %% [markdown]
# ## All plof underpowered genes

# %%
gnomad_constraint = pd.read_csv('../data/all_genes_constraint_gnomad_w_cis.tsv',sep='\t',index_col=0)
gnomad_constraint = gnomad_constraint[['gene','exp_lof','exp_mis_pphen','oeuf_lof','oeuf_mis_pphen']].dropna()
gnomad_constraint['oeuf_lof_rank'] = gnomad_constraint.oeuf_lof.rank(pct=True)
gnomad_constraint['oeuf_mis_pphen_rank'] = gnomad_constraint.oeuf_mis_pphen.rank(pct=True)



with open('../data/gene_lists/lists/mgi_essential.tsv','r') as fid:
    mgi_essential_genes = [x.rstrip() for x in fid.readlines()]

gnomad_constraint['mouse_essential'] = gnomad_constraint.gene.isin(mgi_essential_genes)
gnomad_constraint['lof_underpowered'] = (gnomad_constraint.exp_lof < 10) & (gnomad_constraint.exp_mis_pphen > 30)

g = sns.jointplot(
    data = gnomad_constraint[gnomad_constraint.lof_underpowered],
    x = 'oeuf_lof_rank',
    y = 'oeuf_mis_pphen_rank',
    hue = 'mouse_essential',
    xlim=(0, 1),
    ylim=(0, 1),
    kind='kde',
    marginal_kws={'common_norm':False}
)

# %%
select_constraint = constraint[(constraint['exp|lof'] < 10)]# & (constraint['exp|mis_pphen'] > 25)]
y_true = select_constraint.is_hom_lethal | select_constraint['is_cell_line_nonviable']

plot_precision_recall(y_true, -select_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -select_constraint[metric].rank(pct=True).fillna(1), metric)

print(np.mean(y_true))
print(y_true.shape)
plt.legend()
plt.savefig('../plots/prc_all_underpowered_genes.png')

# %%
y_true = select_constraint.is_hom_lethal | select_constraint.is_cell_line_nonviable
y_pred_1 = select_constraint.logreg_mouse


y_pred_2 = -select_constraint['oeuf|lof'].rank(pct=True).fillna(1)
print(average_precision_bootstrap_ci_contrast(y_true,y_pred_1,y_pred_2,N_iter=1000))

y_pred_2 = -select_constraint['oeuf|mis_pphen'].rank(pct=True).fillna(1)
print(average_precision_bootstrap_ci_contrast(y_true,y_pred_1,y_pred_2,N_iter=1000))

# %% [markdown]
# ## EF hand

# %%
ef_hand_genes = pd.read_csv('../data/gene_lists/ef_hand_proteins.txt',sep='\t')
print(ef_hand_genes.shape)
ef_hands_constraint = constraint[constraint.symbol.isin(ef_hand_genes['Approved symbol'].values)]

print(ef_hands_constraint.shape)
print(ef_hands_constraint['is_hom_lethal'].sum()+ ef_hands_constraint['is_cell_line_nonviable'].sum())
print(ef_hands_constraint.is_underpowered.sum())
y_true = ef_hands_constraint.is_hom_lethal | ef_hands_constraint.is_cell_line_nonviable


plot_precision_recall(y_true, -ef_hands_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -ef_hands_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %% [markdown]
# ## Zinc fingers

# %%
znf_genes = pd.read_csv('../data/gene_lists/zinc_fingers.txt',sep='\t')
print(znf_genes.shape)
znfs_constraint = constraint[constraint.symbol.isin(znf_genes['Approved symbol'].values)]

print(znfs_constraint.shape)
print(znfs_constraint['is_hom_lethal'].sum()+ znfs_constraint['is_cell_line_nonviable'].sum())
print(znfs_constraint.is_underpowered.sum())
y_true = znfs_constraint.is_hom_lethal | znfs_constraint.is_cell_line_nonviable


plot_precision_recall(y_true, -znfs_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -znfs_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %%
znf_genes = pd.read_csv('../data/gene_lists/zinc_finger_C2H2_genes.txt',sep='\t')
print(znf_genes.shape)
znfs_constraint = constraint[constraint.symbol.isin(znf_genes['Approved symbol'].values)]

print(znfs_constraint.shape)
print(znfs_constraint['is_hom_lethal'].sum()+ znfs_constraint['is_cell_line_nonviable'].sum())
print(znfs_constraint.is_underpowered.sum())
y_true = znfs_constraint.is_hom_lethal | znfs_constraint.is_cell_line_nonviable


plot_precision_recall(y_true, -znfs_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -znfs_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %% [markdown]
# ## Receptor ligands

# %%
ligands_constraint.sort_values('oeuf|mis_pphen')[['symbol','exp|lof','exp|mis_pphen','oeuf|lof','oeuf|mis_pphen','is_hom_lethal']].head(20)

# %%
ligand_genes = pd.read_csv('../data/gene_lists/receptor_ligand_genes.txt',sep='\t')
ligands_constraint = constraint[constraint.symbol.isin(ligand_genes['Approved symbol'].values)]

print(ligands_constraint.shape)
print(ligands_constraint['is_hom_lethal'].sum())
print(ligands_constraint['is_cell_line_nonviable'].sum())
y_true = ligands_constraint.is_hom_lethal | ligands_constraint.is_cell_line_nonviable

print(ligands_constraint.is_underpowered.sum())

print(ligands_constraint['exp|mis_pphen'].mean())
plot_precision_recall(y_true, -ligands_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -ligands_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %%
ligands_constraint[['exp|lof','exp|mis_pphen','oeuf|lof','oeuf|mis_pphen']].corr(method='spearman')

# %% [markdown]
# ## G proteins

# %%
g_protein_genes = pd.read_csv('../data/gene_lists/g_proteins.txt',sep='\t')
g_proteins_constraint = constraint[constraint.symbol.isin(g_protein_genes['Approved symbol'].values)]

print(g_proteins_constraint.shape)
print(g_proteins_constraint['is_hom_lethal'].sum())
print(g_proteins_constraint['is_cell_line_nonviable'].sum())
y_true = g_proteins_constraint.is_hom_lethal | g_proteins_constraint.is_cell_line_nonviable
print(g_proteins_constraint.is_underpowered.sum())

plot_precision_recall(y_true, -g_proteins_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -g_proteins_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %%
g_protein_genes = pd.read_csv('../data/gene_lists/ras_like_g_proteins.txt',sep='\t')
g_proteins_constraint = constraint[constraint.symbol.isin(g_protein_genes['Approved symbol'].values)]

print(g_proteins_constraint.shape)
print(g_proteins_constraint['is_hom_lethal'].sum())
print(g_proteins_constraint['is_cell_line_nonviable'].sum())
y_true = g_proteins_constraint.is_hom_lethal | g_proteins_constraint.is_cell_line_nonviable
print(g_proteins_constraint.is_underpowered.sum())

plot_precision_recall(y_true, -g_proteins_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -g_proteins_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %%
g_proteins_constraint[['exp|lof','exp|mis_pphen','oeuf|lof','oeuf|mis_pphen']].corr(method='spearman')

# %% [markdown]
# ## Box transcription factors

# %%
boxtf_genes = pd.read_csv('../data/gene_lists/box_transcription_factors.txt',sep='\t')
boxtfs_constraint = constraint[constraint.symbol.isin(boxtf_genes['Approved symbol'].str.split(', ').explode().values)]
print(boxtfs_constraint['exp|mis_pphen'].mean())


y_true = boxtfs_constraint.is_hom_lethal | boxtfs_constraint.is_cell_line_nonviable
plot_precision_recall(y_true, -boxtfs_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -boxtfs_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %%
hlh_genes = pd.read_csv('../data/gene_lists/hlh_proteins.txt',sep='\t')
print(hlh_genes.shape)
hlh_constraint = constraint[constraint.symbol.isin(hlh_genes['Approved symbol'].str.split(', ').explode().values)]
print(hlh_constraint.shape)
print(hlh_constraint.is_hom_lethal.sum() / hlh_genes.shape[0])

y_true = hlh_constraint.is_hom_lethal | hlh_constraint.is_cell_line_nonviable

print(hlh_constraint.is_underpowered.sum())

print(hlh_constraint['exp|mis_pphen'].mean())
plot_precision_recall(y_true, -hlh_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -hlh_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %%
boxtf_genes[~boxtf_genes['Approved symbol'].isin(boxtfs_constraint.symbol.values)]['Locus type'].value_counts()

# %% [markdown]
# ## Homeobox genes

# %%
hbox_genes = pd.read_csv('../data/gene_lists/hbox_genes.txt',sep='\t')
hboxes_constraint = constraint[
    constraint.symbol.isin(hbox_genes['Approved symbol'].values) |
    constraint.symbol.isin(hbox_genes['Previous symbols'].str.split(', ').explode().values)
    ]

print(hboxes_constraint.shape)
print(hboxes_constraint['is_hom_lethal'].sum())
print(hboxes_constraint['is_cell_line_nonviable'].sum())
y_true = hboxes_constraint.is_hom_lethal | hboxes_constraint.is_cell_line_nonviable


plot_precision_recall(y_true, -hboxes_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -hboxes_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %% [markdown]
# ## Ub-converting enzymes

# %%
ube2_enzyme_genes = pd.read_csv('../data/gene_lists/ube2_enzymes.txt',sep='\t')
ube2_enzymes_constraint = constraint[constraint.symbol.isin(ube2_enzyme_genes['Approved symbol'].values)]

print(ube2_enzymes_constraint.shape)
print(ube2_enzymes_constraint['is_hom_lethal'].sum())
print(ube2_enzymes_constraint['is_cell_line_nonviable'].sum())
y_true = ube2_enzymes_constraint.is_hom_lethal | ube2_enzymes_constraint.is_cell_line_nonviable


plot_precision_recall(y_true, -ube2_enzymes_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -ube2_enzymes_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %% [markdown]
# ## Histones

# %%
histone_genes = pd.read_csv('../data/gene_lists/histones.txt',sep='\t')
histones_constraint = constraint[constraint.symbol.isin(histone_genes['Previous symbols'].str.split(', ').explode().values)]

print(histones_constraint.shape)
print(histones_constraint['is_hom_lethal'].sum())
print(histones_constraint['is_cell_line_nonviable'].sum())
y_true = histones_constraint.is_hom_lethal | histones_constraint.is_cell_line_nonviable


plot_precision_recall(y_true, -histones_constraint[['oeuf|lof', 'oeuf|mis_pphen']].rank(pct=True).fillna(1).mean(axis=1), 'Mean_rank')
print(np.mean(y_true))
print(y_true.shape)

for metric in ['oeuf|lof', 'oeuf|mis_pphen','oeuf|mis_non_pphen']:
    plot_precision_recall(y_true, -histones_constraint[metric].rank(pct=True).fillna(1),metric)

plt.legend()

# %% [markdown]
# ## Performance barplot

# %%
df = pd.read_csv('../results/average_precision_other_families.txt',sep='\t')
df=df.melt(id_vars=['HGNC family','No. genes'],value_name='Average precision',var_name='Predictor')
df.Predictor = df.Predictor.apply(lambda x: x[3:])
palette = sns.color_palette([plt.get_cmap('Dark2')(i) for i in range(3)] + ['black'])
fig, ax = plt.subplots(figsize=(6,3))
sns.barplot(data = df,x='HGNC family',y='Average precision',hue='Predictor',palette=palette,ax=ax)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 0.7), loc=2, borderaxespad=0.)
plt.savefig('../plots/other_families_barplot.png',bbox_inches='tight',)

# %% [markdown]
# # Other metrics

# %% [markdown]
# ## Logistic regression across all genes

# %%
gnomad_constraint = pd.read_csv('../data/all_genes_constraint_gnomad_w_cis.tsv',sep='\t',index_col=0)
gnomad_constraint = gnomad_constraint[['gene','oeuf_lof','oeuf_mis_pphen','exp_lof','exp_mis_pphen']].dropna()
gnomad_constraint['oeuf_lof_rank'] = gnomad_constraint.oeuf_lof.rank(pct=True)
gnomad_constraint['oeuf_mis_pphen_rank'] = gnomad_constraint.oeuf_mis_pphen.rank(pct=True)
gnomad_constraint['oeuf_mean_rank'] = (gnomad_constraint.oeuf_lof_rank + gnomad_constraint.oeuf_mis_pphen_rank) / 2
gnomad_constraint['exp_oeuf_lof'] = gnomad_constraint.oeuf_lof_rank * gnomad_constraint.exp_lof
gnomad_constraint['exp_oeuf_mis_pphen'] = gnomad_constraint.oeuf_mis_pphen_rank * gnomad_constraint.exp_lof
gnomad_constraint['intercept'] = 1
with open('../data/gene_lists/lists/mgi_essential.tsv','r') as fid:
    mgi_essential_genes = [x.rstrip() for x in fid.readlines()]

gnomad_constraint['mouse_essential'] = gnomad_constraint.gene.isin(mgi_essential_genes)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(gnomad_constraint[[
    'intercept','oeuf_lof_rank','oeuf_mis_pphen_rank',
    'exp_lof','exp_mis_pphen'
    ]].values, gnomad_constraint.mouse_essential.astype(int))
gnomad_constraint['oeuf_logit_rank'] = model.predict_proba(
    gnomad_constraint[[
        'intercept','oeuf_lof_rank','oeuf_mis_pphen_rank','exp_lof','exp_mis_pphen'
        ]].values
    )[:,0]
print(model.coef_)

model = LogisticRegression()
model.fit(gnomad_constraint[['intercept','oeuf_lof_rank','oeuf_mis_pphen_rank','exp_lof','exp_mis_pphen','exp_oeuf_lof','exp_oeuf_mis_pphen']].values, gnomad_constraint.mouse_essential.astype(int))
gnomad_constraint['oeuf_logit_inter_rank'] = model.predict_proba(
    gnomad_constraint[[
        'intercept','oeuf_lof_rank','oeuf_mis_pphen_rank','exp_lof','exp_mis_pphen','exp_oeuf_lof','exp_oeuf_mis_pphen'
        ]].values
    )[:,0]
print(model.coef_)


from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

for metric in ('lof','mis_pphen','mean','logit'):#,'logit_inter'):
    fpr, tpr, _ = roc_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[f'oeuf_{metric}_rank'])
    plt.plot(fpr, tpr, label=metric)
plt.legend()

plt.subplots()
for metric, label in zip(('lof','mis_pphen','mean','logit'), ('pLoF','pPM','Mean','Logit')):
    precision, recall, _ = precision_recall_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[f'oeuf_{metric}_rank'],
        )
    plt.plot(recall, precision, label=label)
plt.legend()
plt.xlabel('Recall (mouse essential genes)')
plt.ylabel('Precision (mouse essential genes)')

metric = 'oeuf_lof_rank'

precision, recall, t = precision_recall_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[metric])

df = gnomad_constraint[gnomad_constraint[metric] < 1 - min(t[precision[1:] > 0.6])].sort_values(metric)
print(df.shape[0],df.mouse_essential.mean())

metric = 'oeuf_logit_rank'

precision, recall, t = precision_recall_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[metric])

df = gnomad_constraint[gnomad_constraint[metric] < 1 - min(t[precision[1:] > 0.6])].sort_values(metric)
print(df.shape[0],df.mouse_essential.mean())
print(df.head())

# %% [markdown]
# ## logistic regression applied to pLoF underpowered genes

# %%
gnomad_constraint = pd.read_csv('../data/all_genes_constraint_gnomad_w_cis.tsv',sep='\t',index_col=0)
gnomad_constraint = gnomad_constraint[['gene','exp_lof','exp_mis_pphen','oeuf_lof','oeuf_mis_pphen']].dropna()
gnomad_constraint['oeuf_lof_rank'] = gnomad_constraint.oeuf_lof.rank(pct=True)
gnomad_constraint['oeuf_mis_pphen_rank'] = gnomad_constraint.oeuf_mis_pphen.rank(pct=True)
gnomad_constraint['oeuf_mean_rank'] = (gnomad_constraint.oeuf_lof_rank + gnomad_constraint.oeuf_mis_pphen_rank) / 2



with open('../data/gene_lists/lists/mgi_essential.tsv','r') as fid:
    mgi_essential_genes = [x.rstrip() for x in fid.readlines()]

gnomad_constraint['mouse_essential'] = gnomad_constraint.gene.isin(mgi_essential_genes)



gnomad_constraint['lof_underpowered'] = (gnomad_constraint.exp_lof < 8) #& (gnomad_constraint.exp_mis_pphen > 30)

gnomad_constraint = gnomad_constraint[gnomad_constraint.lof_underpowered]


print(gnomad_constraint.mouse_essential.sum())
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
gnomad_constraint['intercept'] = 1
model.fit(gnomad_constraint[[
    'intercept','oeuf_lof_rank','oeuf_mis_pphen_rank',
    'exp_lof','exp_mis_pphen'
    ]].values, gnomad_constraint.mouse_essential.astype(int))
gnomad_constraint['oeuf_logit_rank'] = model.predict_proba(
    gnomad_constraint[[
        'intercept','oeuf_lof_rank','oeuf_mis_pphen_rank','exp_lof','exp_mis_pphen'
        ]].values
    )[:,0]
print(model.coef_)
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

for metric in ('lof','mis_pphen','mean','logit'):
    fpr, tpr, _ = roc_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[f'oeuf_{metric}_rank'])
    plt.plot(fpr, tpr, label=metric)
plt.legend()

plt.subplots()
for metric in ('lof','mis_pphen','mean','logit'):
    precision, recall, _ = precision_recall_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[f'oeuf_{metric}_rank'])
    plt.plot(recall, precision, label=metric)
plt.legend()

# %%
gnomad_constraint = pd.read_csv('../data/all_genes_constraint_gnomad_w_cis_both.tsv',sep='\t',index_col=0)
gnomad_constraint = gnomad_constraint[['gene','oeuf_lof','oeuf_mis_pphen','oeuf_both','oeuf_both_weighted']].dropna()
plt.scatter(
    x = gnomad_constraint.oeuf_lof,
    y = gnomad_constraint.oeuf_both,
    marker='o',alpha=0.01,color='k'
)
plt.xlim((0,2))
plt.ylim((0,2))
plt.subplots()
plt.scatter(
    x = gnomad_constraint.oeuf_lof,
    y = gnomad_constraint.oeuf_both_weighted,
    marker='o',alpha=0.01,color='k'
)

plt.xlim((0,2))
plt.ylim((0,2))
plt.xlabel('pLoF OEUF')
plt.ylabel('Combined OEUF')
print(stats.spearmanr(gnomad_constraint.oeuf_mis_pphen,gnomad_constraint.oeuf_both,))
print(stats.spearmanr(gnomad_constraint.oeuf_lof,gnomad_constraint.oeuf_both,))
print(stats.spearmanr(gnomad_constraint.oeuf_mis_pphen,gnomad_constraint.oeuf_both_weighted,))
print(stats.spearmanr(gnomad_constraint.oeuf_lof,gnomad_constraint.oeuf_both_weighted,))
plt.subplots()
with open('../data/gene_lists/lists/mgi_essential.tsv','r') as fid:
    mgi_essential_genes = [x.rstrip() for x in fid.readlines()]

gnomad_constraint['mouse_essential'] = gnomad_constraint.gene.isin(mgi_essential_genes)


def roc_analysis(y, x):
    fpr, tpr, _ = roc_curve(y, x)
    score = roc_auc_score(y, x)
    return fpr, tpr, score
fpr, tpr, score = roc_analysis(
    gnomad_constraint.mouse_essential,
    1-gnomad_constraint.oeuf_lof
)
print(score)
plt.plot(fpr, tpr)
fpr, tpr, score = roc_analysis(
    gnomad_constraint.mouse_essential,
    1-gnomad_constraint.oeuf_both
)
print(score)
plt.plot(fpr, tpr)
fpr, tpr, score = roc_analysis(
    gnomad_constraint.mouse_essential,
    1-gnomad_constraint.oeuf_mis_pphen
)
print(score)
plt.plot(fpr, tpr)

fpr, tpr, score = roc_analysis(
    gnomad_constraint.mouse_essential,
    1-gnomad_constraint.oeuf_both_weighted
)
print(score)
plt.plot(fpr, tpr)

# %% [markdown]
# ## Logistic regression applied to GPCRs

# %%
gnomad_constraint = pd.read_csv('../data/all_genes_constraint_gnomad_w_cis.tsv',sep='\t',index_col=0)
gnomad_constraint['oeuf_lof_rank'] = gnomad_constraint.oeuf_lof.rank(pct=True).fillna(1)
gnomad_constraint['oeuf_mis_pphen_rank'] = gnomad_constraint.oeuf_mis_pphen.rank(pct=True).fillna(1)
gnomad_constraint['oeuf_mean_rank'] = (gnomad_constraint.oeuf_lof_rank + gnomad_constraint.oeuf_mis_pphen_rank) / 2

gpcr_constraint = pd.read_csv('../data/gpcr_genes_constraint_gnomad.tsv',sep='\t',index_col=0)
gnomad_constraint['is_gpcr'] = gnomad_constraint.gene.isin(gpcr_constraint.gene_gnomad)
gnomad_constraint = gnomad_constraint[gnomad_constraint.is_gpcr]

print(gnomad_constraint.is_gpcr.sum())
with open('../data/gene_lists/lists/mgi_essential.tsv','r') as fid:
    mgi_essential_genes = [x.rstrip() for x in fid.readlines()]

gnomad_constraint['mouse_essential'] = gnomad_constraint.gene.isin(mgi_essential_genes)

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

y = gnomad_constraint.mouse_essential
print(np.mean(y))

for metric in ('lof','mis_pphen','mean'):
    x = 1-gnomad_constraint[f'oeuf_{metric}_rank']
    print(roc_auc_score(y,x))
    fpr, tpr, _ = roc_curve(y, x)
    plt.plot(fpr, tpr, label=metric)
plt.legend()

plt.subplots()
for metric in ('lof','mis_pphen','mean'):
    x = 1-gnomad_constraint[f'oeuf_{metric}_rank']
    print(average_precision_score(y, x))
    precision, recall, _ = precision_recall_curve(y, x)        
    plt.plot(recall, precision, label=metric)
plt.legend()

# %%


# %%
gnomad_constraint = pd.read_csv('../data/all_genes_constraint_gnomad_w_cis.tsv',sep='\t',index_col=0)
gnomad_constraint['oeuf_lof_rank'] = gnomad_constraint.oeuf_lof.rank(pct=True).fillna(1)
gnomad_constraint['oeuf_mis_pphen_rank'] = gnomad_constraint.oeuf_mis_pphen.rank(pct=True).fillna(1)
gnomad_constraint['oeuf_mean_rank'] = (gnomad_constraint.oeuf_lof_rank + gnomad_constraint.oeuf_mis_pphen_rank) / 2

gpcr_constraint = pd.read_csv('../data/gpcr_genes_constraint_gnomad.tsv',sep='\t',index_col=0)
gnomad_constraint['is_gpcr'] = gnomad_constraint.gene.isin(gpcr_constraint.gene_gnomad)
gnomad_constraint = gnomad_constraint[gnomad_constraint.is_gpcr]

print(gnomad_constraint.is_gpcr.sum())


essential_gpcrs = pd.read_csv('../data/gpcr_genes_mgi_essential_curated.tsv',sep='\t', index_col=0)
essential_gpcrs = essential_gpcrs[essential_gpcrs.level.isin(['Lethal','Developmental'])].symbol.values
gnomad_constraint['mouse_essential'] = gnomad_constraint.gene.isin(essential_gpcrs)

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

y = gnomad_constraint.mouse_essential
print(np.mean(y))

for metric in ('lof','mis_pphen','mean'):
    x = 1-gnomad_constraint[f'oeuf_{metric}_rank']
    print(roc_auc_score(y,x))
    fpr, tpr, _ = roc_curve(y, x)
    plt.plot(fpr, tpr, label=metric)
plt.legend()

plt.subplots()
for metric in ('lof','mis_pphen','mean'):
    x = 1-gnomad_constraint[f'oeuf_{metric}_rank']
    print(average_precision_score(y, x))
    precision, recall, _ = precision_recall_curve(y, x)        
    plt.plot(recall, precision, label=metric)
plt.legend()

# %% [markdown]
# ## Weighted constraint combination on all genes

# %%
gnomad_constraint = pd.read_csv('../data/all_genes_constraint_gnomad_w_cis_both.tsv',sep='\t',index_col=0)
gnomad_constraint = gnomad_constraint[['gene','oeuf_lof','oeuf_mis_pphen','exp_lof','exp_mis_pphen','oeuf_both','oeuf_both_weighted']].dropna()
gnomad_constraint['oeuf_lof_rank'] = gnomad_constraint.oeuf_lof.rank(pct=True)
gnomad_constraint['oeuf_mis_pphen_rank'] = gnomad_constraint.oeuf_mis_pphen.rank(pct=True)
gnomad_constraint['oeuf_mean_rank'] = (gnomad_constraint.oeuf_lof_rank + gnomad_constraint.oeuf_mis_pphen_rank) / 2
gnomad_constraint['oeuf_both_rank'] = gnomad_constraint.oeuf_both.rank(pct=True)
gnomad_constraint['oeuf_both_weighted_rank'] = gnomad_constraint.oeuf_both_weighted.rank(pct=True)
# gnomad_constraint['exp_oeuf_lof'] = gnomad_constraint.oeuf_lof_rank * gnomad_constraint.exp_lof
# gnomad_constraint['exp_oeuf_mis_pphen'] = gnomad_constraint.oeuf_mis_pphen_rank * gnomad_constraint.exp_lof
# gnomad_constraint['intercept'] = 1
with open('../data/gene_lists/lists/mgi_essential.tsv','r') as fid:
    mgi_essential_genes = [x.rstrip() for x in fid.readlines()]

gnomad_constraint['mouse_essential'] = gnomad_constraint.gene.isin(mgi_essential_genes)

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

for metric in ('lof','mis_pphen','mean','both','both_weighted'):
    fpr, tpr, _ = roc_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[f'oeuf_{metric}_rank'])
    plt.plot(fpr, tpr, label=metric)
plt.legend()

plt.subplots()
for metric in ('lof','mis_pphen','mean','both','both_weighted'):
    precision, recall, _ = precision_recall_curve(
        gnomad_constraint.mouse_essential,
        1-gnomad_constraint[f'oeuf_{metric}_rank'])
    plt.plot(recall, precision, label=metric)
plt.legend()

# %% [markdown]
# # Bootstrap significance test for gene sets

# %% [markdown]
# ## pLoF, sample size = 19,704, p = 0.12

# %%
# test for AUROC and average precision scores
y = constraint.mouse_essential_2
x = constraint.oeuf_lof_rank

auroc_samples  = resample(
    lambda x, y: roc_auc_score(y, x), 
    x, y, iter=10000)
ap_samples  = resample(
    lambda x, y: average_precision_score(y, x), 
    x, y, iter=10000)

# %%
alpha = 0.05
print(roc_auc_score(y, x))
print(empirical_ci(auroc_samples, alpha))

print(average_precision_score(y, x))
print(empirical_ci(ap_samples, alpha))

fig, ax = plt.subplots(1,2,figsize = (6,2.5), sharey = True)
ax[0].hist(auroc_samples,bins=20)
ax[0].set_xlabel('AUROC')
ax[1].hist(ap_samples,bins=20)
ax[1].set_xlabel('Average precision')

# %% [markdown]
# ## pLoF, sample size = 392, p = 0.12

# %%
y = constraint.mouse_essential_2
x = constraint.oeuf_lof_rank


auroc_samples = resample(
    lambda x, y: roc_auc_score(y, x), 
    x, y, iter=10000, n_sample = 392)

ap_samples = resample(
    lambda x, y: average_precision_score(y, x), 
    x, y, iter=10000, n_sample = 392)

# %%
print(roc_auc_score(y, x))
print(empirical_ci(auroc_samples, alpha))
print(average_precision_score(y, x))
print(empirical_ci(ap_samples, alpha))

fig, ax = plt.subplots(1,2,figsize = (6,2.5), sharey = True)
ax[0].hist(auroc_samples,bins=20)
ax[0].set_xlabel('AUROC')
ax[1].hist(ap_samples,bins=20)
ax[1].set_xlabel('Average precision')

# %% [markdown]
# ## Comparison sanity check (pLoF vs. pLoF + noise)

# %%
y = constraint.mouse_essential
x = constraint.oeuf_lof_rank
noise = np.random.normal(scale = 0.3, size=len(x))

auroc_samples  = resample(
    lambda x, y: roc_auc_score(y, x), 
    [x, x + noise], y, iter=10000, n_sample = 392, multiple=True)  


ap_samples  = resample(
    lambda x, y: average_precision_score(y, x), 
    [x, x + noise], y, iter=10000, n_sample = 392, multiple=True)

plot_comparison(auroc_samples, sample_labels=['pLoF OEUF','pLoF OEUF\n + noise'],metric_label='AUROC',figsize=(9, 4), legend_loc='upper left')
plot_comparison(ap_samples, sample_labels=['pLoF OEUF','pLoF OEUF\n + noise'],metric_label='Average precision',figsize=(9, 4), legend_loc='upper right')

# %% [markdown]
# ## pLoF vs. pPM, n = N

# %%
y = constraint.mouse_essential_2
x = constraint.oeuf_lof_rank
x_2 = constraint.oeuf_mis_pphen_rank

auroc_samples  = resample(
    lambda x, y: roc_auc_score(y, x), 
    [x, x_2], y, iter=10000, multiple=True)  
ap_samples  = resample(
    lambda x, y: average_precision_score(y, x), 
    [x, x_2], y, iter=10000, multiple=True)

print(roc_auc_score(y, x))
print(roc_auc_score(y, x_2))
plot_comparison(
    auroc_samples, 
    sample_labels=['pLoF OEUF','pPM OEUF'],
    metric_label='AUROC',
    figsize=(9, 4), 
    legend_loc='upper left'
    )

print(average_precision_score(y, x))
print(average_precision_score(y, x_2))
plot_comparison(
    ap_samples, 
    sample_labels=['pLoF OEUF','pPM OEUF'],
    metric_label='Average precision',
    figsize=(9, 4), 
    legend_loc='upper right'
    )

# %% [markdown]
# ## pLoF vs. pPM, n=392

# %%
y = constraint.mouse_essential_2
x = constraint.oeuf_lof_rank
x_2 = constraint.oeuf_mis_pphen_rank

auroc_samples  = resample(
    lambda x, y: roc_auc_score(y, x), 
    [x, x_2], y, iter=10000, n_sample = 392, multiple=True)  
ap_samples  = resample(
    lambda x, y: average_precision_score(y, x), 
    [x, x_2], y, iter=10000, n_sample = 392, multiple=True)

plot_comparison(auroc_samples, sample_labels=['pLoF OEUF','pPM OEUF'],metric_label='AUROC',figsize=(9, 4), legend_loc='upper left')
plot_comparison(ap_samples, sample_labels=['pLoF OEUF','pPM OEUF'],metric_label='Average precision',figsize=(9, 4), legend_loc='upper right')

# %% [markdown]
# ## Bootstrapping from GPCR subset for comparison

# %%
y = constraint_gpcrs.mouse_essential
x = constraint_gpcrs.oeuf_lof_rank
x_2 = constraint_gpcrs.oeuf_mis_pphen_rank

auroc_samples  = resample(
    lambda x, y: roc_auc_score(y, x), 
    [x, x_2], y, iter=10000, multiple=True)  
ap_samples  = resample(
    lambda x, y: average_precision_score(y, x), 
    [x, x_2], y, iter=10000, multiple=True)

plot_comparison(auroc_samples, sample_labels=['pLoF OEUF','pPM OEUF'],metric_label='AUROC',figsize=(9, 4), legend_loc='upper left')
plot_comparison(ap_samples, sample_labels=['pLoF OEUF','pPM OEUF'],metric_label='Average precision',figsize=(9, 4), legend_loc='upper right')

# %%
auroc_samples  = roc_functions.resample(
    lambda x, y: roc_auc_score(y, x), 
    [x, x_2], y, iter=10000, n_sample = 392, multiple=True)  
ap_samples  = roc_functions.resample(
    lambda x, y: average_precision_score(y, x), 
    [x, x_2], y, iter=10000, n_sample = 392, multiple=True)

# %%
fig, ax = plt.subplots(1, 2, figsize = (6, 2.5), sharey=True)

_ = ax[0].hist(auroc_samples[:,0],alpha=0.5,bins=20,label='pLoF')
_ = ax[0].hist(auroc_samples[:,1],alpha=0.5,bins=20,label='pPM')
ax[0].set_xlabel('AUROC')
ax[0].set_ylabel('Frequency')
ax[0].legend(loc= 'upper left')

d = auroc_samples[:,0] - auroc_samples[:,1]
_ = ax[1].hist(d,alpha=0.5,bins=20,label='pLoF OEUF', color='k')
ax[1].set_xlabel('AUROC difference')
ax[1].set_ylabel('Frequency')

roc_functions.empirical_ci(d, 0.05)

# %%
fig, ax = plt.subplots(1, 2, figsize = (6, 2.5), sharey=True)

_ = ax[0].hist(ap_samples[:,0],alpha=0.5,bins=20,label='pLoF')
_ = ax[0].hist(ap_samples[:,1],alpha=0.5,bins=20,label='pPM')
ax[0].set_xlabel('Average precision')
ax[0].set_ylabel('Frequency')
ax[0].legend(loc= 'upper right')

d = ap_samples[:,0] - ap_samples[:,1]
_ = ax[1].hist(d,alpha=0.5,bins=20,label='pLoF OEUF', color='k')
ax[1].set_xlabel('Average precision difference')
ax[1].set_ylabel('Frequency')

roc_functions.empirical_ci(d, 0.05)

# %%
roc_functions.normal_ci(auroc_samples[:,0] - auroc_samples[:,1], 0.05)

# %%
roc_functions.empirical_ci(auroc_samples[:,0] - auroc_samples[:,1], 0.05)

# %%
d = auroc_samples[:,0] - auroc_samples[:,1]
d = np.sort(d)
n = len(d)
(
    d[int(n * alpha/2 )],
    d[int(n * (1 - alpha/2))]
)

# %%
_ = plt.hist(ap_samples[:,0],alpha=0.5,bins=20,label='pLoF OEUF')
_ = plt.hist(ap_samples[:,1],alpha=0.5,bins=20,label='pPM OEUF')
plt.xlabel('Average precision')
plt.ylabel('Frequency')
plt.legend()

roc_functions.empirical_ci(ap_samples[:,0] - ap_samples[:,1], 0.05)

# %%
f = lambda x, y: average_precision_score(y, x)
ap_samples  = roc_functions.resample(f, x, y, iter=1000)
plt.hist(ap_samples, bins=50)
# Set confidence level
alpha = 0.05
# Normal CIs
print(roc_functions.normal_ci(ap_samples, alpha))


# %% [markdown]
# # Sensitivity analysis

# %%
impacts = ['lof','mis_pphen']
threshold = 10
binspace = np.linspace(0,3,50)

fig, ax = plt.subplots(2,2)
for i in range(2):
    impact = impacts[i]
    for j in range(2):
        if j == 0:
            df = constraint[
                (constraint[f'exp|lof'] > threshold) &
                (constraint[f'oeuf|{impact}'] <3)
            ]
        else:
            df = constraint[
                (constraint[f'exp|lof'] < threshold) &
                (constraint[f'oeuf|{impact}'] <3)
            ]
        ax[i, j].hist(df[f'oeuf|{impact}'],bins=binspace)
        ax[i, j].hist(df[df.is_hom_lethal==1][f'oeuf|{impact}'],bins=binspace)
        ax[i, j].set_xlim((0,3))

# %%
impacts = ['lof','mis_pphen']
threshold = 10
binspace = np.linspace(0,100,100)

fig, ax = plt.subplots(2,2,sharey=True)
for i in range(2):
    impact = impacts[i]
    for j in range(2):
        if j == 1:
            df = constraint[
                (constraint[f'exp|lof'] > threshold) &
                (constraint[f'exp|{impact}'] <100)
            ]
        else:
            df = constraint[
                (constraint[f'exp|lof'] < threshold) &
                (constraint[f'exp|{impact}'] <100)
            ]
        n, _, _ = ax[i, j].hist(df[f'exp|{impact}'],bins=binspace)
        ax[i, j].hist(df[df.is_hom_lethal==1][f'exp|{impact}'],bins=binspace)
        print(df.shape[0], df.is_hom_lethal.sum())
        ax[i, j].set_xlim((0,100))
plt.savefig('../plots/histogram_exp_underpowered.png')

# %%
impacts = ['lof','mis_pphen']
threshold = 10
binspace = np.linspace(0,100,20)

gpcrs_constraint = pd.read_csv('../data/constraint/constraint_and_phenotypes_all.tsv',sep='\t',index_col=0)

fig, ax = plt.subplots(2,2,sharey=True)
for i in range(2):
    impact = impacts[i]
    for j in range(2):
        if j == 1:
            df = gpcrs_constraint[
                (gpcrs_constraint[f'exp|lof'] > threshold) &
                (gpcrs_constraint[f'exp|{impact}'] <100)
            ]
        else:
            df = gpcrs_constraint[
                (gpcrs_constraint[f'exp|lof'] < threshold) &
                (gpcrs_constraint[f'exp|{impact}'] <100)
            ]
        n, _, _ = ax[i, j].hist(df[f'exp|{impact}'],bins=binspace)
        ax[i, j].hist(df[gpcrs_constraint['mouse_knockout_phenotype_level'].isin(['Lethal','Developmental'])][f'exp|{impact}'],bins=binspace)
        ax[i, j].set_xlim((0,100))


plt.savefig('../plots/histogram_exp_underpowered_gpcrs.png')

# %% [markdown]
# # Z scores

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/essential_genes/gnomad_mouse_essential_genes_hom.txt',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gnomad_zscores = gnomad_zscores.fillna(0)
gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

y = gnomad_zscores.mouse_essential
print('P =', y.mean())


for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,2))

plt.legend()

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label)
    print(round(ap,2))
plt.legend()


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='kde',
    #joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %% [markdown]
# # mouse het lethals

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/essential_genes/mouse_het_lethals.txt',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gnomad_zscores = gnomad_zscores.fillna(0)
gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

y = gnomad_zscores.mouse_essential
print('P =', y.mean())


for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,3))

plt.legend()

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label)
    print(round(ap,3))
plt.legend()


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='kde',
    #joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %% [markdown]
# # cell line essential

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/essential_genes/cell_essentials.txt',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gnomad_zscores = gnomad_zscores.fillna(0)
gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

y = gnomad_zscores.mouse_essential
print('P =', y.mean())


for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,2))

plt.legend()

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label)
    print(round(ap,2))
plt.legend()


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='kde',
    #joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %% [markdown]
# # Mouse het lethal or cell line essential

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/essential_genes/ce_mouse_het_lethals.txt',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gnomad_zscores = gnomad_zscores.fillna(0)
gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

y = gnomad_zscores.mouse_essential
print('P =', y.mean())


for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,2))

plt.legend()

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label)
    print(round(ap,2))
plt.legend()


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='kde',
    #joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %% [markdown]
# ## Mouse lethal or cell line essential

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/essential_genes/ce_mouse_lethals.txt',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gnomad_zscores = gnomad_zscores.fillna(0)
gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

y = gnomad_zscores.mouse_essential
print('P =', y.mean())


for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,2))

plt.legend()

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_mean','z_max'], ['pLoF z-score','pPM z-score','Mean z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label)
    print(round(ap,2))
plt.legend()


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='kde',
    #joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %% [markdown]
# # Uncurated GPCR labels

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/mouse_essential_genes/gnomad_mouse_essential_genes_hom.tsv',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
gnomad_zscores = gnomad_zscores.merge(gpcrs, left_on='gene',right_on='gene_gnomad')
gnomad_zscores = gnomad_zscores.fillna(0)

gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

y = gnomad_zscores.mouse_essential
print('P =', y.mean())

sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label = label)
    print(round(auroc,2))

plt.legend()
plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential GPCR genes)')
plt.ylabel('True positive rate (mouse essential GPCR genes)')

plt.legend()
plt.savefig('../plots/Fig3C_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label)
    print(round(ap,2))
plt.hlines(y.mean(), 0, 1, color='k',linestyle='dotted',label='Baseline')
plt.ylim((0,1))
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')

plt.savefig('../plots/Fig3D_gpcr_essential_prc.png', dpi=450)
g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


# %% [markdown]
# # Z-scores with all genes from genome

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/mouse_essential_genes/gnomad_mouse_essential_genes_hom.tsv',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')['gene_gnomad']
gnomad_zscores['is_gpcr'] = gnomad_zscores.gene.isin(gpcrs)
gnomad_zscores = gnomad_zscores.fillna(0)

gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential
print('N =', y.sum())
print('P =', y.mean())


sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential genes)')
plt.ylabel('True positive rate (mouse essential genes)')

plt.legend()
plt.savefig('../plots/SuppFig2B_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label)
    print(round(ap,5))
plt.hlines(y.mean(), 0, 1, color='k',linestyle='dotted',label='Baseline')
plt.ylim((0,1))

plt.legend()
plt.xlabel('Recall (mouse essential genes)')
plt.ylabel('Precision (mouse essential genes)')
#plt.yticks(np.linspace(0, 0.02, 5))

plt.savefig('../plots/Fig3D_essential_prc.png', dpi=450)

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gnomad_mouse_essentials = pd.read_csv('../data/labels/mouse_essential_genes/gnomad_mouse_essential_genes_hom.tsv',sep='\t', header=None)[0]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gnomad_mouse_essentials)
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')['gene_gnomad']
gnomad_zscores['is_gpcr'] = gnomad_zscores.gene.isin(gpcrs)
gnomad_zscores = gnomad_zscores.fillna(0)

gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential & gnomad_zscores.is_gpcr
print('N =', y.sum())
print('P =', y.mean())


sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential GPCR genes)')
plt.ylabel('True positive rate (mouse essential GPCR genes)')

plt.legend()
plt.savefig('../plots/SuppFig1A_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label,dp=4)
    print(round(ap,5))
plt.ylim((0,0.02))
plt.hlines(y.mean(), 0, 1, color='k', linestyle='dotted', label = 'Baseline')
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')
plt.yticks(np.linspace(0, 0.02, 5))

plt.savefig('../plots/Fig3C_gpcr_essential_prc.png', dpi=450,layout='tight')

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


# %% [markdown]
# # Z scores classification of curated GPCR labels with all genes from genome

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_genes_mgi_essential_curated.tsv',sep='\t',index_col=0)
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.level.isin(['Lethal','Cardiovascular'])]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gpcr_curated_essential_genes.symbol)
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')['gene_gnomad']
gnomad_zscores['is_gpcr'] = gnomad_zscores.gene.isin(gpcrs)
gnomad_zscores = gnomad_zscores.fillna(0)

gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential & gnomad_zscores.is_gpcr
print('N =', y.sum())
print('P =', y.mean())


sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential GPCR genes)')
plt.ylabel('True positive rate (mouse essential GPCR genes)')

plt.legend()
plt.savefig('../plots/SuppFig1A_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label,dp=4)
    print(round(ap,5))
plt.ylim((0,0.02))
plt.hlines(y.mean(), 0, 1, color='k', linestyle='dotted', label = 'Baseline')
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')
plt.yticks(np.linspace(0, 0.02, 5))

plt.savefig('../plots/Fig3C_gpcr_essential_prc.png', dpi=450,layout='tight')

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t').drop_duplicates(subset='gene').fillna(0)

gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
gnomad_zscores = gnomad_zscores.rename(columns = {'gene':'gene_gnomad'})
gnomad_zscores = gnomad_zscores.merge(gpcrs, on='gene_gnomad',how='left')

gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.combined_phenotype.isin(['Lethal','Developmental'])]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gpcr_curated_essential_genes.gene)


print(gnomad_zscores.mouse_essential.mean())


gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential
print('N =', y.sum())
print('P =', y.mean())

sns.set_context('notebook')
sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate \n(mouse essential GPCR genes)')
plt.ylabel('True positive rate \n(mouse essential GPCR genes)')

plt.legend()
plt.tight_layout()
plt.savefig('../plots/SuppFig2A_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label,dp=4)
    print(round(ap,5))
plt.ylim((0,0.02))
plt.hlines(y.mean(), 0, 1, color='k', linestyle='dotted', label = 'Baseline')
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')
plt.yticks(np.linspace(0, 0.02, 5))
plt.tight_layout()
plt.savefig('../plots/Fig3C_gpcr_essential_prc.png', dpi=450)

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t').drop_duplicates(subset='gene').fillna(0)

gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
gnomad_zscores = gnomad_zscores.rename(columns = {'gene':'gene_gnomad'})
gnomad_zscores = gnomad_zscores.merge(gpcrs, on='gene_gnomad')

gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.combined_phenotype.isin(['Lethal'])]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gpcr_curated_essential_genes.gene)


print(gnomad_zscores.mouse_essential.mean())


gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential
print('N =', y.sum())
print('P =', y.mean())


sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential GPCR genes)')
plt.ylabel('True positive rate (mouse essential GPCR genes)')

plt.legend()
plt.savefig('../plots/SuppFig1A_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label,dp=4)
    print(round(ap,5))
plt.ylim((0,1))
plt.hlines(y.mean(), 0, 1, color='k', linestyle='dotted', label = 'Baseline')
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')
plt.yticks(np.linspace(0, 1, 5))

plt.savefig('../plots/Fig3C_gpcr_essential_prc.png', dpi=450,layout='tight')

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t').drop_duplicates(subset='gene').fillna(0)

gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
gnomad_zscores = gnomad_zscores.rename(columns = {'gene':'gene_gnomad'})
gnomad_zscores = gnomad_zscores.merge(gpcrs, on='gene_gnomad')
gnomad_zscores = gnomad_zscores[gnomad_zscores.receptor_class == 'Class A (Rhodopsin)']
gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.combined_phenotype.isin(['Lethal','Developmental'])]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gpcr_curated_essential_genes.gene)


print(gnomad_zscores.mouse_essential.mean())


gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential
print('N =', y.sum())
print('P =', y.mean())


sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential GPCR genes)')
plt.ylabel('True positive rate (mouse essential GPCR genes)')

plt.legend()
plt.savefig('../plots/SuppFig1A_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label,dp=4)
    print(round(ap,5))
plt.ylim((0,1))
plt.hlines(y.mean(), 0, 1, color='k', linestyle='dotted', label = 'Baseline')
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')
plt.yticks(np.linspace(0, 1, 5))

plt.savefig('../plots/Fig3C_gpcr_essential_prc.png', dpi=450,layout='tight')

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t').drop_duplicates()

gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')['gene_gnomad']
gnomad_zscores['is_gpcr'] = gnomad_zscores.gene.isin(gpcrs)
gnomad_zscores = gnomad_zscores.fillna(0)
gnomad_zscores = gnomad_zscores[gnomad_zscores.is_gpcr]

gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)


gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.combined_phenotype.isin(['Lethal','Developmental'])]
print(gpcr_curated_essential_genes.shape[0])
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gpcr_curated_essential_genes.gene)
print(gnomad_zscores.mouse_essential.mean())

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential 
print('N =', y.sum())
print('P =', y.mean())


sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential GPCR genes)')
plt.ylabel('True positive rate (mouse essential GPCR genes)')

plt.legend()
plt.savefig('../plots/SuppFig1A_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label,dp=2)
    print(round(ap,5))
plt.ylim((0,1))
plt.hlines(y.mean(), 0, 1, color='k', linestyle='dotted', label = 'Baseline')
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')
#plt.yticks(np.linspace(0, 0.02, 5))

plt.savefig('../plots/Fig3C_gpcr_essential_prc.png', dpi=450,layout='tight')

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)

# %%
gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
gpcr_curated_essential_genes = pd.read_csv('../data/labels/mouse_essential_genes/gpcr_genes_mgi_essential_curated.tsv',sep='\t',index_col=0)
gpcr_curated_essential_genes = gpcr_curated_essential_genes[gpcr_curated_essential_genes.level.isin(['Lethal','Cardiovascular'])]
gnomad_zscores['mouse_essential'] = gnomad_zscores.gene.isin(gpcr_curated_essential_genes.symbol)
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')['gene_gnomad']
gnomad_zscores['is_gpcr'] = gnomad_zscores.gene.isin(gpcrs)
gnomad_zscores = gnomad_zscores.fillna(0)
gnomad_zscores = gnomad_zscores[gnomad_zscores.is_gpcr]

gnomad_zscores['z_mean'] = (gnomad_zscores.z_lof + gnomad_zscores.z_mis_pphen)/ 2
gnomad_zscores['z_max'] = np.maximum(gnomad_zscores.z_lof,gnomad_zscores.z_mis_pphen)

#gnomad_zscores = gnomad_zscores[~(gnomad_zscores.mouse_essential & ~gnomad_zscores.is_gpcr)]
y = gnomad_zscores.mouse_essential & gnomad_zscores.is_gpcr
print('N =', y.sum())
print('P =', y.mean())


sns.set_palette('Dark2')
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None, label=label)
    print(round(auroc,2))

plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline')
plt.xlabel('False positive rate (mouse essential GPCR genes)')
plt.ylabel('True positive rate (mouse essential GPCR genes)')

plt.legend()
plt.savefig('../plots/SuppFig1A_gpcr_essential_roc.png', dpi=450)

plt.subplots()
for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    ap = plot_precision_recall(y, x, label = label,dp=2)
    print(round(ap,5))
plt.ylim((0,1))
plt.hlines(y.mean(), 0, 1, color='k', linestyle='dotted', label = 'Baseline')
plt.legend()
plt.xlabel('Recall (mouse essential GPCR genes)')
plt.ylabel('Precision (mouse essential GPCR genes)')
#plt.yticks(np.linspace(0, 0.02, 5))

plt.savefig('../plots/Fig3C_gpcr_essential_prc.png', dpi=450,layout='tight')

g = sns.jointplot(
    data = gnomad_zscores,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = y,
    xlim=(-10, 20),
    ylim=(-10, 20),
    kind='scatter',
    joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
    marginal_kws={'common_norm':False}
)


# %% [markdown]
# # Comparison of zscores with OEUF

# %%
y = constraint.mouse_essential

plt.subplots()
for metric, label in zip(['oeuf_lof_rank', 'oeuf_mis_pphen_rank'], ['pLoF OEUF','pPM OEUF']):
    x = constraint[metric].values
    ap = plot_roc(y, x)
    print(round(ap,2))

y = gnomad_zscores.mouse_essential
print('P =', y.mean())

for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_roc(y, x, ax= None)
    print(round(auroc,2))
plt.plot((0, 1), color='k',linestyle='dotted')
plt.legend()

y = constraint.mouse_essential
plt.subplots()
for metric, label in zip(['oeuf_lof_rank', 'oeuf_mis_pphen_rank'], ['pLoF OEUF','pPM OEUF']):
    x = constraint[metric].values
    ap = plot_precision_recall(y, x)
    print(round(ap,2))

y = gnomad_zscores.mouse_essential
print('P =', y.mean())

for metric, label in zip(['z_lof','z_mis_pphen','z_max'], ['pLoF z-score','pPM z-score','Max z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_precision_recall(y, x, ax= None)
    print(round(auroc,2))
plt.hlines(y.mean(), 0, 1, color='k',linestyle='dotted')
plt.legend()

# %%
y = constraint.mouse_essential
plt.subplots()
for metric, label in zip(['oeuf_lof_rank'], ['pLoF OEUF']):
    x = constraint[metric].values
    ap = plot_precision_recall(y, x)
    print(round(ap,2))

y = gnomad_zscores.mouse_essential
print('P =', y.mean())

for metric, label in zip(['z_lof'], ['pLoF z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_precision_recall(y, x, ax= None)
    print(round(auroc,2))
plt.hlines(y.mean(), 0, 1, color='k',linestyle='dotted')
plt.legend()

# %%
y = constraint.mouse_essential
plt.subplots()
for metric, label in zip(['oeuf_mis_pphen_rank'], ['pLoF OEUF']):
    x = constraint[metric].values
    ap = plot_precision_recall(y, x)
    print(round(ap,2))

y = gnomad_zscores.mouse_essential
print('P =', y.mean())

for metric, label in zip(['z_mis_pphen'], ['pLoF z-score']):
    x = gnomad_zscores[metric].values
    auroc = plot_precision_recall(y, x, ax= None)
    print(round(auroc,2))
plt.hlines(y.mean(), 0, 1, color='k',linestyle='dotted')
plt.legend()

# %% [markdown]
# # GWAS scores

# %%
otg_disease_associations = pd.read_csv('../data/labels/opentargets-genetics/otg_disease_trait_count_by_gene_with_eqtl.tsv', sep='\t')
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
constraint_gpcrs = constraint.merge(gpcrs, left_on='gene',right_on='gene_gnomad')
constraint_gpcrs = constraint_gpcrs.merge(otg_disease_associations, left_on = 'gene_y', right_on='gene', how='left')
constraint_gpcrs['efos'] = ~constraint_gpcrs.efos.isna()

g = sns.jointplot(
    data = constraint_gpcrs,
    x = 'oeuf_lof',
    y = 'oeuf_mis_pphen',
    hue = 'efos',
    xlim=(0, 3),
    ylim=(0, 3),
    kind='scatter',
    joint_kws={'alpha':0.5},
    marginal_kws={'common_norm':False}
)
plt.subplots()
plot_roc(
    constraint_gpcrs.efos,
    constraint_gpcrs.oeuf_lof_rank,
    label = 'pLoF'
)
plot_roc(
    constraint_gpcrs.efos,
    constraint_gpcrs.oeuf_mis_pphen_rank,
    label = 'pPM'
)
plt.legend()

plt.subplots()
plot_precision_recall(
    constraint_gpcrs.efos,
    constraint_gpcrs.oeuf_lof_rank,
    label = 'pLoF'
)
plot_precision_recall(
    constraint_gpcrs.efos,
    constraint_gpcrs.oeuf_mis_pphen_rank,
    label = 'pPM'
)
plt.legend()

# %%
otg_disease_associations = pd.read_csv('../data/labels/opentargets-genetics/otg_disease_trait_count_by_gene_with_eqtl.tsv', sep='\t')

gnomad_zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t').fillna(0)
gpcrs = pd.read_csv('../data/labels/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
constraint_gpcrs = gnomad_zscores.merge(gpcrs, left_on='gene',right_on='gene_gnomad')
constraint_gpcrs = constraint_gpcrs.merge(otg_disease_associations, left_on = 'gene_y', right_on='gene', how='left')
constraint_gpcrs['efos'] = ~constraint_gpcrs.efos.isna()

g = sns.jointplot(
    data = constraint_gpcrs,
    x = 'z_lof',
    y = 'z_mis_pphen',
    hue = 'efos',
    xlim=(0, 3),
    ylim=(0, 3),
    kind='scatter',
    joint_kws={'alpha':0.5},
    marginal_kws={'common_norm':False}
)
plt.subplots()
plot_roc(
    constraint_gpcrs.efos,
    constraint_gpcrs.z_lof,
    label = 'pLoF'
)
plot_roc(
    constraint_gpcrs.efos,
    constraint_gpcrs.z_mis_pphen,
    label = 'pPM'
)
plt.legend()

plt.subplots()
plot_precision_recall(
    constraint_gpcrs.efos,
    constraint_gpcrs.z_lof,
    label = 'pLoF'
)
plot_precision_recall(
    constraint_gpcrs.efos,
    constraint_gpcrs.z_mis_pphen,
    label = 'pPM'
)
plt.legend()

# %%
constraint_gpcrs.columns


