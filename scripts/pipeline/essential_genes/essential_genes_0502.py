# %%
import pandas as pd
from tqdm import tqdm
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
# # Count essential genes by family

# %%
gpcr_genes = pd.read_csv('../data/labels/gene_families/gpcr_genes_human_gpcrdb.tsv',sep='\t')
gpcr_mouse_phenotypes_curated = pd.read_csv('../data/labels/curated_essential_genes/gpcr_mouse_human_curated_genes.txt',sep='\t')
gpcr_genes_with_essentiality = gpcr_genes.merge(gpcr_mouse_phenotypes_curated)
essential_gpcr_genes = gpcr_genes_with_essentiality[gpcr_genes_with_essentiality.combined_phenotype.isin(['Lethal','Developmental'])]

print(gpcr_genes_with_essentiality.receptor_class.value_counts())
print(essential_gpcr_genes.receptor_class.value_counts())

# print(gpcr_genes_with_essentiality.receptor_family.value_counts())
# print(essential_gpcr_genes.receptor_family.value_counts())


# %% [markdown]
# # Load LOEUF, zscore and label GPCR genes

# %%
constraint = pd.read_csv("../data/constraint/gnomad/all_genes_constraint_exact_a0.1.tsv", sep = '\t', index_col=0)
constraint = constraint.drop_duplicates('gene')

rankscale_oeuf = lambda x: 1-x.rank(pct=True).fillna(1)
constraint['oeuf_lof_rank'] = rankscale_oeuf(constraint['oeuf_lof'])
constraint['oeuf_mis_pphen_rank'] = rankscale_oeuf(constraint['oeuf_mis_pphen'])
constraint['oeuf_mean_rank'] = (constraint.oeuf_lof_rank + constraint.oeuf_mis_pphen_rank) / 2

zscores = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')
zscores = zscores.fillna(0).drop_duplicates('gene')
constraint = constraint.merge(zscores[['gene','z_lof','z_mis_pphen']])
constraint['z_mean'] = (constraint.z_lof + constraint.z_mis_pphen)/ 2
constraint['z_max'] = np.maximum(constraint.z_lof,constraint.z_mis_pphen)

gpcrs = pd.read_csv('../data/labels/gene_families/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
constraint = constraint.rename(columns = {'gene':'gene_gnomad'})
constraint = constraint.merge(gpcrs[['gene','gene_gnomad']], how= 'left')
constraint['is_gpcr'] = ~constraint.gene.isna()

constraint

# %% [markdown]
# # Label essential genes

# %%
gnomad_ce_mouse_het_essentials = pd.read_csv('../data/labels/gnomad_essential_genes/ce_mouse_het_lethals.txt',sep='\t', header=None)[0]
constraint['ce_het_lethal'] = constraint.gene_gnomad.isin(gnomad_ce_mouse_het_essentials)

gnomad_mouse_lethals= pd.read_csv('../data/labels/gnomad_essential_genes/mouse_lethals.txt',sep='\t', header=None)[0]
constraint['mouse_lethal'] = constraint.gene_gnomad.isin(gnomad_mouse_lethals)
constraint['gpcr_mouse_lethal'] = constraint.mouse_lethal & constraint.is_gpcr

curated_mouse_essentials = pd.read_csv('../data/labels/curated_essential_genes/combined_essential_genes.txt',sep='\t',header=None)[0]
constraint['gpcr_mouse_essential'] = constraint.gene.isin(curated_mouse_essentials) & constraint.is_gpcr

constraint_gpcrs = constraint[constraint.is_gpcr]

# %% [markdown]
# # Plotting functions

# %%
def roc_plot(data, predictor_cols, predictor_labels, output_col, output_label):
    for predictor_col, predictor_label in zip(predictor_cols, predictor_labels):
        x = data[predictor_col].values
        y = data[output_col]
        auroc = plot_roc(y, x, ax= None, label = predictor_label)
        print(predictor_label, 'AUROC:', round(auroc,2))

    plt.plot((0, 1), color='k',linestyle='dotted',label='Baseline: AUROC=0.50')
    plt.xlabel(f'False positive rate\n({output_label})')
    plt.ylabel(f'True positive rate\n({output_label})')
    plt.legend()

gnomad_mouse_essentials = pd.read_csv('../data/labels/gnomad_essential_genes/ce_mouse_lethals.txt',sep='\t', header=None)[0]
zscores['essential'] = zscores.gene.isin(gnomad_mouse_essentials)
roc_plot(
    zscores, 
    ['z_lof'],
    ['pLoF z-score'], 
    'essential', 
    'Cell line essential or mouse hom lethal'
)

# %%
def prc_plot(data, predictor_cols, predictor_labels, output_col, output_label, dp = 2):
    for predictor_col, predictor_label in zip(predictor_cols, predictor_labels):
        x = data[predictor_col].values
        y = data[output_col]
        ap = plot_precision_recall(y, x, ax= None, label = predictor_label, dp=dp)
        print(predictor_label, 'AP:', round(ap,dp))

    print(round(ap,dp))
    plt.ylim((0,1))
    plt.hlines(y.mean(), 0, 1, color='k',linestyle='dotted',label=f'Baseline: AP= {round(y.mean(),dp)}')
    plt.xlabel(f'False positive rate\n({output_label})')
    plt.ylabel(f'True positive rate\n({output_label})')
    plt.legend()

gnomad_mouse_essentials = pd.read_csv('../data/labels/gnomad_essential_genes/ce_mouse_lethals.txt',sep='\t', header=None)[0]
zscores['essential'] = zscores.gene.isin(gnomad_mouse_essentials)
prc_plot(
    zscores, 
    ['z_lof'],
    ['pLoF z-score'], 
    'essential', 
    'Cell line essential or mouse hom lethal'
)

# %%
def jointplot_scatter(data):
    return sns.jointplot(
        data = zscores,
        x = 'z_lof',
        y = 'z_mis_pphen',
        hue = 'essential',
        xlim=(-10, 20),
        ylim=(-10, 20),
        kind='scatter',
        joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
        marginal_kws={'common_norm':False}
    )
jointplot_scatter(zscores)

# %%
def jointplot_kde(data):
    return sns.jointplot(
        data = data,
        x = 'z_lof',
        y = 'z_mis_pphen',
        hue = 'essential',
        xlim=(-10, 20),
        ylim=(-10, 20),
        kind='kde',
        #joint_kws={'size':5,'alpha':0.5, 'edgecolor':None},
        marginal_kws={'common_norm':False}
    )
jointplot_kde(zscores)

# %% [markdown]
# # Bootstrapping analysis

# %%
def resample(target, predictors, fns, iter=100, n_sample = None, multiple=False):
    # Indices of positive and negative examples
    plus = np.where(target==1)[0]
    minus = np.where(target==0)[0]
    # Count positive and negative examples    
    if n_sample is None:
        n_plus = len(plus)
        n_minus = len(minus)
    # if number of samples specified, keep them in same ratio
    else:
        n_plus = int(n_sample * len(plus) / len(target))
        n_minus = int(n_sample * len(minus) / len(target))
        
    
    if not multiple:
        results = [[] for f in fns]
        for i in tqdm(range(iter)):
            # select same number of positive and negative examples with replacement
            plus_select = np.random.choice(plus, size = n_plus)
            minus_select = np.random.choice(minus, size = n_minus)

            x_sample = np.concatenate((predictors[plus_select], predictors[minus_select]),axis=0)
            y_sample = np.concatenate((target[plus_select], target[minus_select]),axis=0)
            for j, f in enumerate(fns):
                results[j].append(f(y_sample, x_sample))
    else: 
        results = [[[] for p in predictors] for f in fns]
        for i in tqdm(range(iter)):
            # select same number of positive and negative examples with replacement
            plus_select = np.random.choice(plus, size = n_plus)
            minus_select = np.random.choice(minus, size = n_minus)
            
            for k, x_ in enumerate(predictors):
                x_sample = np.concatenate((x_[plus_select], x_[minus_select]),axis=0)
                y_sample = np.concatenate((target[plus_select], target[minus_select]),axis=0)
                for j, f in enumerate(fns):
                    results[j][k].append(f(y_sample, x_sample))

    return np.array(results)

def plot_comparison(samples, sample_labels, metric_label, bins=20, figsize=(6,2.5), legend_loc = 'upper right'):
    x1 = samples[0]
    x2 = samples[1]
    d = x1 - x2

    print(empirical_ci(x1, 0.05))
    print(empirical_ci(x2, 0.05))
    print(empirical_ci(d, 0.05))
    
    fig, ax = plt.subplots(1, 2, figsize = figsize, sharey=True)

    _ = ax[0].hist(x1,alpha=0.5,bins=bins,label=sample_labels[0])
    _ = ax[0].hist(x2,alpha=0.5,bins=bins,label=sample_labels[1])
    ax[0].set_xlabel(metric_label)
    ax[0].set_ylabel('Frequency')
    ax[0].legend(loc=legend_loc)

    _ = ax[1].hist(d,alpha=0.5,bins=bins, color='k')
    ax[1].set_xlabel(f'{metric_label} difference \n({sample_labels[0]} - {sample_labels[1]})')
    ax[1].set_ylabel('Frequency')
    return ax

def bootstrap_analysis(data, x1_col, x2_col, y_col, x1_label, x2_label, y_label, iter = 10000, figsize = (9,4)):
    x1 = data[x1_col].values
    x2 = data[x2_col].values
    y = data[y_col].values

    print(roc_auc_score(y, x1))
    print(roc_auc_score(y, x2))
    print(average_precision_score(y, x1))
    print(average_precision_score(y, x2))

    auroc_samples, ap_samples = resample(
        target = y,
        predictors = [x1, x2],
        fns = [roc_auc_score, average_precision_score],
        iter = iter, multiple=True
    )
    plot_comparison(
        auroc_samples, 
        sample_labels=[x1_label, x2_label],
        metric_label = f'AUROC ({y_label})',
        figsize=figsize, 
        legend_loc='upper left'
    )

    plot_comparison(
        ap_samples, 
        sample_labels=[x1_label, x2_label],
        metric_label = f'Average precision ({y_label})',
        figsize=figsize,
        legend_loc='upper right'
    )

bootstrap_analysis(zscores, 'z_lof','z_mis_pphen','essential','pPM z-score','pLoF z-score','Essential', iter=100)

# %% [markdown]
# # Mouse het lethal or cell line essential with LOEUF
# 
# Replicate analysis from Karczewski et al

# %%
print(constraint.ce_het_lethal.sum())

roc_plot(constraint, ['oeuf_lof_rank','oeuf_mis_pphen_rank'],['pLoF OEUF','pPM OEUF'],'ce_het_lethal','Mouse het lethal or cell line essential')
plt.subplots()
prc_plot(constraint, ['oeuf_lof_rank','oeuf_mis_pphen_rank'],['pLoF OEUF','pPM OEUF'],'ce_het_lethal','Mouse het lethal or cell line essential')

# %%
print(constraint.ce_het_lethal.sum())
sns.set_context('notebook')
sns.set_palette('Dark2')
roc_plot(constraint, 
         ['z_lof','z_mis_pphen','z_max'],
         ['pLoF z-score','pPM z-score','Max z-score'],
         'ce_het_lethal',
         'Mouse het lethal or cell line essential')
plt.tight_layout()
plt.savefig('../plots/SuppFig3B_ce_het_lethal_roc.png',dpi=450)
plt.subplots()
prc_plot(constraint,          
         ['z_lof','z_mis_pphen','z_max'],
         ['pLoF z-score','pPM z-score','Max z-score'],
         'ce_het_lethal',
         'Mouse het lethal or cell line essential')
plt.tight_layout()
plt.savefig('../plots/SuppFig3C_ce_het_lethal_prc.png',dpi=450)

# %%
bootstrap_analysis(constraint,'oeuf_lof_rank','oeuf_mis_pphen_rank','ce_het_lethal','pLoF OEUF','pPM OEUF','Mouse het lethal or cell line essential',iter=10000)
bootstrap_analysis(constraint,'z_lof','z_mis_pphen','ce_het_lethal','pLoF z-score','pPM z-score','Mouse het lethal or cell line essential',iter=10000)
bootstrap_analysis(constraint,'z_lof','z_max','ce_het_lethal','pLoF z-score','pPM z-score','Mouse het lethal or cell line essential',iter=10000)

# %% [markdown]
# # Mouse hom lethals

# %%
roc_plot(constraint, ['oeuf_lof_rank','oeuf_mis_pphen_rank'],['pLoF OEUF','pPM OEUF'],'mouse_lethal','Mouse knockout lethal genes')
plt.subplots()
prc_plot(constraint, ['oeuf_lof_rank','oeuf_mis_pphen_rank'],['pLoF OEUF','pPM OEUF'],'mouse_lethal','Mouse knockout lethal genes')


plt.subplots()
roc_plot(constraint, 
         ['oeuf_lof_rank','z_lof'],
         ['pLoF OEUF','pLoF z-score'],
         'mouse_lethal','Mouse lethal')
plt.subplots()
prc_plot(constraint, 
         ['oeuf_lof_rank','z_lof'],
         ['pLoF OEUF','pLoF z-score'],
         'mouse_lethal','Mouse lethal')


sns.set_palette('Dark2')
sns.set_context('notebook')
plt.subplots()
roc_plot(constraint, 
         ['z_lof','z_mis_pphen','z_max'],
         ['pLoF z-score','pPM z-score','Max z-score'],
         'mouse_lethal','Mouse knockout lethal genes')
plt.tight_layout()
plt.savefig('../plots/SuppFig2B_essential_roc.png',dpi=450)
plt.subplots()
prc_plot(constraint, 
         ['z_lof','z_mis_pphen','z_max'],
         ['pLoF z-score','pPM z-score','Max z-score'],
         'mouse_lethal','Mouse knockout lethal genes')
plt.tight_layout()
plt.savefig('../plots/Fig3D_essential_prc.png',dpi=450)

# %%
bootstrap_analysis(constraint,'oeuf_lof_rank','oeuf_mis_pphen_rank','mouse_lethal','pLoF OEUF','pPM OEUF','Mouse lethal', iter=10000)
bootstrap_analysis(constraint,'oeuf_lof_rank','z_lof','mouse_lethal','pLoF OEUF','pLoF z-score','Mouse knockout lethal',iter=10000)
bootstrap_analysis(constraint,'z_lof','z_mis_pphen','mouse_lethal','pLoF OEUF','pLoF z-score','Mouse knockout lethal',iter=10000)
bootstrap_analysis(constraint,'z_lof','z_max','mouse_lethal','pLoF OEUF','pLoF z-score','Mouse knockout lethal',iter=10000)

# %% [markdown]
# # Z scores for GPCRs from whole genome (uncurated)

# %%
sns.set_context('notebook')
sns.set_palette('Dark2')
roc_plot(constraint, 
         ['oeuf_lof_rank','z_mis_pphen','z_max'],
         ['LOEUF','pPM z-score','Max z-score'],
         'gpcr_mouse_lethal','Mouse lethal GPCR genes')
plt.tight_layout()
plt.savefig('../plots/SuppFig2B_gpcr_lethal_roc.png',dpi=450)
plt.subplots()


# %%
prc_plot(constraint, 
         ['oeuf_lof_rank','z_mis_pphen','z_max'],
         ['LOEUF','pPM z-score','Max z-score'],
         'gpcr_mouse_lethal','Mouse lethal GPCR genes',dp=4)
plt.ylim(0, 0.015)
plt.tight_layout()
plt.savefig('../plots/SuppFig2C_gpcr_lethal_prc.png',dpi=450)

# %%
bootstrap_analysis(constraint,'z_lof','z_max','gpcr_mouse_lethal','LOEUF','pPM z-score','Max z-score',iter=10000)
bootstrap_analysis(constraint,'z_lof','z_mis_pphen','gpcr_mouse_lethal','LOEUF','pPM z-score','Max z-score',iter=10000)

# %% [markdown]
# # Z scores for GPCRs from whole genome (curated)

# %%
sns.set_palette('Dark2')
sns.set_context('notebook')
roc_plot(constraint, 
         ['z_lof','z_mis_pphen','z_max'],
         ['pLoF z-score','pPM z-score','Max z-score'],
         'gpcr_mouse_essential','Mouse essential GPCR genes')
plt.tight_layout()
plt.savefig('../plots/SuppFig2A_gpcr_essential_roc.png',dpi=450)
plt.subplots()
prc_plot(constraint, 
         ['z_lof','z_mis_pphen','z_max'],
         ['pLoF z-score','pPM z-score','Max z-score'],
         'gpcr_mouse_essential','Mouse essential GPCR genes',dp=4)
plt.ylim(0, 0.015)
plt.tight_layout()
plt.savefig('../plots/Fig3C_gpcr_essential_prc.png',dpi=450)


# %%
from sklearn.metrics import confusion_matrix

# %%
print(confusion_matrix(constraint.mouse_lethal & ~constraint.gpcr_mouse_essential, constraint.z_max > 5))
print(confusion_matrix(constraint.mouse_lethal & ~constraint.gpcr_mouse_essential, constraint.z_max > 5, normalize='true'))

# %%
print(confusion_matrix(constraint.gpcr_mouse_essential, constraint.z_max > 5))
print(confusion_matrix(constraint.gpcr_mouse_essential, constraint.z_max > 5, normalize='true'))

# %%
bootstrap_analysis(constraint,'z_lof','z_max','gpcr_mouse_essential','pLoF z-score','pPM z-score','Max z-score',iter=10000)
bootstrap_analysis(constraint,'z_lof','z_mis_pphen','gpcr_mouse_essential','pLoF z-score','pPM z-score','Max z-score',iter=10000)

# %% [markdown]
# # GPCRs only (uncurated)

# %%
sns.set_palette('Dark2')
roc_plot(constraint_gpcrs, 
         ['oeuf_lof_rank','z_mis_pphen','z_max'],
         ['LOEUF','pPM z-score','Max z-score'],
         'mouse_lethal','Mouse lethal')
plt.subplots()
prc_plot(constraint_gpcrs, 
         ['oeuf_lof_rank','z_mis_pphen','z_max'],
         ['LOEUF','pPM z-score','Max z-score'],
         'mouse_lethal','Mouse lethal')


# %%
print(confusion_matrix(constraint_gpcrs.mouse_lethal, constraint_gpcrs.z_max > 5))
print(confusion_matrix(constraint_gpcrs.mouse_lethal, constraint_gpcrs.z_max > 5,normalize='true'))
print(confusion_matrix(constraint_gpcrs.mouse_lethal, constraint_gpcrs.z_mis_pphen > 5,normalize='pred'))

# %%
bootstrap_analysis(constraint_gpcrs,'z_lof','z_max','gpcr_mouse_lethal','pLoF z-score','Max z-score','Mouse knockout lethal GPCRs',iter=10000)
bootstrap_analysis(constraint_gpcrs,'z_lof','z_mis_pphen','gpcr_mouse_lethal','pLoF z-score','Max z-score','Mouse knockout lethal GPCRs',iter=10000)

# %% [markdown]
# # GPCRs only (curated)

# %%
sns.set_palette('Dark2')
roc_plot(constraint_gpcrs, 
         ['oeuf_lof_rank','z_mis_pphen','z_max'],
         ['LOEUF','pPM z-score','Max z-score'],
         'gpcr_mouse_essential','Mouse essential GPCRs')
plt.subplots()
prc_plot(constraint_gpcrs, 
         ['oeuf_lof_rank','z_mis_pphen','z_max'],
         ['LOEUF','pPM z-score','Max z-score'],
         'gpcr_mouse_essential','Mouse essential GPCRs')


# %%
bootstrap_analysis(constraint_gpcrs,'z_lof','z_max','gpcr_mouse_essential','pLoF z-score','Max z-score','Mouse knockout lethal GPCRs',iter=10000)
bootstrap_analysis(constraint_gpcrs,'z_lof','z_mis_pphen','gpcr_mouse_essential','pLoF z-score','Max z-score','Mouse knockout lethal GPCRs',iter=10000)

# %% [markdown]
# # Z score GPCR tree plots

# %%
constraint = pd.read_csv("../data/constraint/gnomad/all_genes_constraint_zscores.tsv", sep = '\t', index_col=0)
gnomad_mouse_essentials = pd.read_csv('../data/labels/gnomad_essential_genes/gnomad_mouse_essential_genes_hom.txt',sep='\t', header=None)[0]
constraint['mouse_essential'] = constraint.gene.isin(gnomad_mouse_essentials)

gpcrs = pd.read_csv('../data/labels/gene_families/gpcr_genes_human_gpcrdb.tsv', sep = '\t')
constraint_gpcrs = constraint.rename(columns = {'gene':'gene_gnomad'}).merge(gpcrs)

mappings = pd.read_csv('gpcr_mapper/gpcr_treemapper_coords.txt',sep='\t')
df = constraint_gpcrs.merge(mappings)

df = df[df.mouse_essential]
df['label']  = df.mouse_essential


ax = plot_gpcr_mapper(df.x, df.y, df['z_lof'], df.label, 
    'pLoF constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.tight_layout()
plt.savefig('../plots/Fig3A_z_lof_essentials_treemapper_uncurated.png',dpi=450)

df['label'] = 'Essential in mice'
ax = plot_gpcr_mapper(df.x, df.y, df['z_mis_pphen'], df.label, 
    'pPM constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.tight_layout()
plt.savefig('../plots/Fig3B_z_mis_pphen_essentials_treemapper_uncurated.png',dpi=450)



# %%
mappings = pd.read_csv('gpcr_mapper/gpcr_treemapper_coords.txt',sep='\t')
df = constraint_gpcrs.merge(mappings)

df = df[df.gpcr_mouse_essential]
df['label']  = df.gpcr_mouse_essential


ax = plot_gpcr_mapper(df.x, df.y, df['z_lof'], df.label, 
    'pLoF constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.tight_layout()
plt.savefig('../plots/Fig3A_z_lof_essentials_treemapper_curated.png',dpi=450)

df['label'] = 'Essential in mice'
ax = plot_gpcr_mapper(df.x, df.y, df['z_mis_pphen'], df.label, 
    'pPM constraint z-score',
    marker_size=30,cmap='RdBu_r',cscale=(-5,5))
ax.get_legend().remove()
plt.tight_layout()
plt.savefig('../plots/Fig3B_z_mis_pphen_essentials_treemapper_curated.png',dpi=450)



# %% [markdown]
# # Mouse het lethal GPCR constraint

# %%
gnomad_ce_mouse_het_essentials = pd.read_csv('../data/labels/gnomad_essential_genes/ce_mouse_het_lethals.txt',sep='\t', header=None)[0]
cols = ['gene','oeuf_lof', 'z_lof','z_mis_pphen']
constraint[constraint.is_gpcr & constraint.gene.isin(gnomad_ce_mouse_het_essentials)][cols]

# %% [markdown]
# # Different families

# %%
df_iuphar = pd.read_csv('../data/labels/gene_families/targets_and_families.csv', skiprows=1)
df_iuphar = df_iuphar[~df_iuphar['HGNC symbol'].isna()] # remove multi-unit targets
df_iuphar = df_iuphar[~df_iuphar['Human SwissProt'].isna()] # remove pseudogenes

# Rename columns
df_iuphar = df_iuphar[['Type', 'Family name', 'Target name', 'HGNC id', 'HGNC symbol', 'HGNC name', 'Human SwissProt']]
df_iuphar.columns = ['target_class', 'target_family', 'target_name', 'hgnc_id', 'hgnc_symbol', 'hgnc_name', 'human_swissprot']

# Check no NAs present
print(df_iuphar.isna().sum())

print('Total targets', df_iuphar.shape[0])
print('Breakdown by class:\n',df_iuphar.value_counts('target_class'))
df_iuphar.to_csv('../data/labels/gene_families/iuphar_targets_clean.csv')

# %%
gencode_transcripts = pd.read_csv('../data/labels/gene_families/gencode_grch37_transcripts_by_HGNC.txt',sep='\t',header=None)
gencode_transcripts.columns = ['ensembl_transcript_version','hgnc_symbol','hgnc_id']
gencode_transcripts = gencode_transcripts.drop(columns=['hgnc_id'])
df_iuphar = df_iuphar.merge(gencode_transcripts)
df_iuphar['ensembl_transcript'] = df_iuphar['ensembl_transcript_version'].map(lambda x: x.split('.')[0])
df_iuphar.to_csv('../data/labels/gene_families/iuphar_targets_with_transcripts.tsv',sep='\t')

# %%
df_iuphar = pd.read_csv('../data/labels/gene_families/iuphar_targets_with_transcripts.tsv',sep='\t')
df_iuphar_w_constraint = df_iuphar.merge(constraint, left_on=['ensembl_transcript'],right_on = ['transcript'], how='right')
df_iuphar_w_constraint['target_class'] = df_iuphar_w_constraint.target_class.fillna('None')

# %%


order = ['None', 'gpcr','transporter', 'catalytic_receptor', 'enzyme',  'lgic','vgic',
       'nhr',   'other_ic','other_protein']
fig, ax = plt.subplots(2, 1, figsize = (8, 7))

sns.boxplot(data = df_iuphar_w_constraint, x = 'target_class',y='z_lof',hue='mouse_lethal',order = order,ax=ax[0])
# sns.stripplot(data = df_iuphar_w_constraint, x = 'target_class',y='z_mis_pphen',order = order,ax=ax[0], s=2, color='k')
#sns.stripplot(data = df_iuphar_w_constraint, x = 'target_class',y='z_mis_pphen',s=2,color='k')
locs = ax[0].get_xticks()
labels = ax[0].get_xticklabels()
ax[0].set_xticks(locs,[x.get_text().replace('_','\n') for x in labels], rotation=45)
ax[0].set_ylim((-5, 15))


sns.boxplot(data = df_iuphar_w_constraint, x = 'target_class',y='z_mis_pphen',hue='mouse_lethal',order = order,ax=ax[1])
# sns.stripplot(data = df_iuphar_w_constraint, x = 'target_class',y='z_lof',order = order,ax=ax[1], s=2, color='k')
#sns.stripplot(data = df_iuphar_w_constraint, x = 'target_class',y='z_mis_pphen',s=2,color='k')
locs = ax[1].get_xticks()
labels = ax[1].get_xticklabels()
ax[1].set_xticks(locs,[x.get_text().replace('_','\n') for x in labels], rotation=45)
ax[1].set_ylim((-5, 15))

plt.tight_layout()

# %%
fig, ax = plt.subplots(1, 2, figsize = (8, 4))

sns.boxplot(
    data = df_iuphar_w_constraint, 
    x = 'is_gpcr',
    y = 'z_lof',
    hue = 'mouse_lethal',
    ax = ax[0]
)
ax[0].set_ylim((-5, 10))
ax[0].hlines(df_iuphar_w_constraint.z_lof.median(), -0.5, 1.5,color='r',linestyle = 'dashed')

sns.boxplot(
    data = df_iuphar_w_constraint, 
    x = 'is_gpcr',
    y = 'z_mis_pphen',
    hue = 'mouse_lethal',
    ax = ax[1]
)

ax[1].set_ylim((-5, 12))
ax[1].hlines(df_iuphar_w_constraint.z_mis_pphen.median(), -0.5, 1.5,color='r',linestyle = 'dashed')
plt.tight_layout()

# %%
fig, ax = plt.subplots(1, 2, figsize = (8, 4))

sns.violinplot(
    data = df_iuphar_w_constraint, 
    x = 'is_gpcr',
    y = 'z_lof',
    hue = 'mouse_lethal',
    ax = ax[0]
)
ax[0].set_ylim((-5, 10))
ax[0].hlines(df_iuphar_w_constraint.z_lof.median(), -0.5, 1.5,color='r',linestyle = 'dashed')

sns.violinplot(
    data = df_iuphar_w_constraint, 
    x = 'is_gpcr',
    y = 'z_mis_pphen',
    hue = 'mouse_lethal',
    ax = ax[1]
)

ax[1].set_ylim((-5, 10))
ax[1].hlines(df_iuphar_w_constraint.z_mis_pphen.median(), -0.5, 1.5,color='r',linestyle = 'dashed')
plt.tight_layout()

# %%
from patsy import dmatrix

# %%
X = dmatrix('z_lof + z_mis_pphen ', df_iuphar_w_constraint)
X = pd.DataFrame(X, columns=X.design_info.column_names)
y = df_iuphar_w_constraint.mouse_lethal.astype(int).values
mod = Logit(y, X)
res = mod.fit()
res.summary()

# %%
X = dmatrix('z_lof + z_mis_pphen ', df_iuphar_w_constraint[df_iuphar_w_constraint.is_gpcr])
X = pd.DataFrame(X, columns=X.design_info.column_names)
y = df_iuphar_w_constraint[df_iuphar_w_constraint.is_gpcr].mouse_lethal.astype(int).values
mod = Logit(y, X)
res = mod.fit()
print(res.summary())

X = dmatrix('z_lof', df_iuphar_w_constraint[df_iuphar_w_constraint.is_gpcr])
X = pd.DataFrame(X, columns=X.design_info.column_names)
y = df_iuphar_w_constraint[df_iuphar_w_constraint.is_gpcr].mouse_lethal.astype(int).values
mod = Logit(y, X)
res = mod.fit()
print(res.summary())

X = dmatrix('z_mis_pphen ', df_iuphar_w_constraint[df_iuphar_w_constraint.is_gpcr])
X = pd.DataFrame(X, columns=X.design_info.column_names)
y = df_iuphar_w_constraint[df_iuphar_w_constraint.is_gpcr].mouse_lethal.astype(int).values
mod = Logit(y, X)
res = mod.fit()
print(res.summary())

# %%
X_ = dmatrix('C(target_class) * z_lof + C(target_class) * z_mis_pphen ', df_iuphar_w_constraint)
X = pd.DataFrame(X_, columns=X_.design_info.column_names)
y = df_iuphar_w_constraint.mouse_lethal.astype(int).values
mod = Logit(y, X)
res = mod.fit()
res.summary()

# %%
df = pd.DataFrame(res.tvalues)

#df = df.reset_index()
df = df[df.index.str.startswith('C(target_class)')]
df = df.rename(columns={0:'t'})
df['class'] = df.index.map(lambda x: x.split(':')[0].split('T.')[1][:-1])
df['predictor'] = np.repeat(['Class','Z_lof','Z_mis_pphen'], 9)
df = df[df.predictor.isin(['Z_lof','Z_mis_pphen'])]

df_wide = df.pivot(index = 'class',columns='predictor')
# df_wide['diff'] = df_wide[('t','Z_lof')] - df_wide[('t','Z_mis_pphen')]
order = df_wide.sort_values(('t','Z_mis_pphen'),ascending=False).index
sns.set_palette('Dark2')
sns.barplot(data = df, x = 'class', y = 't', hue = 'predictor', order = order)
plt.xticks(rotation=45)


