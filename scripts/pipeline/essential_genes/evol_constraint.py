# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# %%
df_evol = pd.read_csv('../data/constraint/evolutionary/evol_constraint.txt', sep='\t', skiprows=1)
df_evol = df_evol[~df_evol.gene_name.isna()].copy()
df_evol['gene_name'] = df_evol.gene_name.str.slice(start=1, stop=-1)
df_evol

# %%
df_gnomad = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv', sep='\t')

# %%
gencode_grch37_transcripts = pd.read_csv('../data/labels/gene_families/gencode_grch37_transcripts_by_HGNC.txt',sep='\t',header = None)

# %%
gencode_grch37_transcripts['transcript'] = gencode_grch37_transcripts[0].str.split('.').str[0]
gencode_grch37_transcripts['gene'] = gencode_grch37_transcripts[1]

# %%
print(df_evol.shape[0])
print(df_evol.gene_name.nunique())
print(df_evol.gene_name.isin(gencode_grch37_transcripts.gene).sum())

# %%
print(df_gnomad.shape[0])
print(df_gnomad.transcript.isin(gencode_grch37_transcripts.transcript).sum())

# %%
df_evol = df_evol.merge(gencode_grch37_transcripts, left_on='gene_name', right_on='gene', how='inner')

# %%
df_evol_gnomad = df_evol.merge(df_gnomad, left_on='transcript', right_on='transcript', how='inner')

# %%
gnomad_mouse_lethals= pd.read_csv('../data/labels/gnomad_essential_genes/mouse_lethals.txt',sep='\t', header=None)[0]
df_evol_gnomad['mouse_lethal'] = df_evol_gnomad.gene_y.isin(gnomad_mouse_lethals)

# %%
df_evol_gnomad.shape

# %%
sns.histplot(
    data = df_evol_gnomad, x = 'z_lof', y = 'z_mis_pphen'
)
plt.xlim(-5, 15)
plt.ylim(-5, 15)

# %%
# Mammalian conservation and mutational constraint
sns.histplot(
    data = df_evol_gnomad, x = 'z_mis_pphen', y = 'fracCdsCons'
)
print(spearmanr(df_evol_gnomad.z_lof.fillna(0), df_evol_gnomad.fracCdsCons))
print(spearmanr(df_evol_gnomad.z_mis_pphen.fillna(0), df_evol_gnomad.fracCdsCons))

# %%
# Primate conservation and mutational constraint
sns.histplot(
    data = df_evol_gnomad, x = 'z_lof', y = 'fracConsPr'
)
print(spearmanr(df_evol_gnomad.z_lof.fillna(0), df_evol_gnomad.fracConsPr))
print(spearmanr(df_evol_gnomad.z_mis_pphen.fillna(0), df_evol_gnomad.fracConsPr))

# %%
sns.histplot(data = df_evol_gnomad, x = 'z_mis_pphen', y = 'fracCdsCons', hue = 'mouse_lethal', multiple = 'stack')


# %%
from sklearn.metrics import roc_auc_score, roc_curve

for metric in ['z_mis_pphen','z_lof','z_max','fracCdsCons','fracConsPr']:
    print(metric, roc_auc_score(df_evol_gnomad.mouse_lethal, df_evol_gnomad[metric].fillna(0)))
    fpr, tpr, _ = roc_curve(df_evol_gnomad.mouse_lethal, df_evol_gnomad[metric].fillna(0))
    plt.plot(fpr, tpr, label = metric)

# %%
df_evol_gnomad['z_max'] = df_evol_gnomad[['z_mis_pphen','z_lof']].max(axis=1)
df_evol_gnomad[['z_lof','z_mis_pphen','z_max', 'fracCdsCons','fracConsPr']].corr(method='spearman')

# %%
from patsy import dmatrix
from statsmodels.discrete.discrete_model import Logit
df_evol_gnomad['mouse_lethal'] = df_evol_gnomad.mouse_lethal.astype(int)
form = 'mouse_lethal ~ z_mis_pphen + z_lof + fracCdsCons + fracConsPr'
mod= Logit.from_formula(form, data = df_evol_gnomad.fillna(0))
res = mod.fit()
print(res.summary())

print(metric, roc_auc_score(df_evol_gnomad.mouse_lethal, res.predict()))


