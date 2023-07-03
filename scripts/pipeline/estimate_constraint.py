# %%
import pandas as pd
import numpy as np

from statsmodels.stats.rates import test_poisson, confint_poisson
from statsmodels.stats.multitest import fdrcorrection

def chisq_zscore(count, exposure):
    if exposure == 0:
        return None
    else:
        chisq = np.sqrt((count - exposure)**2 / exposure)
        if count < exposure:
            return chisq
        else:
            return -chisq

def poisson_upper_bound(obs, exp, alpha=0.1):
    if exp == 0:
        return (None, None, None, None)
    else:
        ci = confint_poisson(
            count = obs, 
            exposure=exp,
            method="exact-c",
            alpha = alpha
        )
    return ci[1]
    
# # Exact Poisson confidence intervals, alpha = 0.1

gnomad_constraint = pd.read_csv('data/constraint/oeuf_hgnc.tsv',sep='\t')

# Check input columns
impacts = ['syn','mis','lof','mis_pphen']
input_columns = sum([[f'obs_{impact}', f'exp_{impact}'] for impact in impacts], ['cds_length','num_coding_exons'])
gnomad_constraint[input_columns] = gnomad_constraint[input_columns].fillna(0)


alpha = 0.1

output_columns = []
for impact in impacts:
    print(impact)
    gnomad_constraint[f'z_{impact}'] = gnomad_constraint.apply(
        lambda x: chisq_zscore(x[f'obs_{impact}'], x[f'exp_{impact}']),
        axis=1
        )
    gnomad_constraint[f'oeuf_{impact}'] = gnomad_constraint.apply(
        lambda row: poisson_upper_bound(row[f'obs_{impact}'], row[f'exp_{impact}'], alpha),
        axis=1
        )
    gnomad_constraint[f'pval_{impact}'] = np.where(
                gnomad_constraint[f'z_{impact}'] > 0,
                np.exp(-(gnomad_constraint[f'z_{impact}']) **2),
                1
        )
    fdr_sig, fdr_pval_corr = fdrcorrection(gnomad_constraint[f'pval_{impact}'], alpha=0.01)
    gnomad_constraint[f'fdr_sig_{impact}'] = fdr_sig
    gnomad_constraint[f'fdr_pval_corr_{impact}'] = fdr_pval_corr
    print(impact)
    print('threshold: ', gnomad_constraint[gnomad_constraint[f'fdr_sig_{impact}']][f'z_{impact}'].min())
    print('number significant: ', gnomad_constraint[f'fdr_sig_{impact}'].sum())

    output_columns.append(f'z_{impact}')
    output_columns.append(f'oeuf_{impact}')
    
gnomad_constraint['z_max'] = gnomad_constraint[['z_lof','z_mis_pphen']].max(axis=1)

rvis = pd.read_csv('data/constraint/rvis_hgnc.tsv',sep='\t')
rvis = rvis[['hgnc_symbol','rvis_score']]
gnomad_constraint = gnomad_constraint.merge(rvis, on = 'hgnc_symbol', how = 'left')

phylop = pd.read_csv('data/constraint/phylop_hgnc.tsv',sep='\t')
phylop = phylop[['hgnc_symbol','phylop_score', 'phylop_score_primate']]
gnomad_constraint = gnomad_constraint.merge(phylop, on = 'hgnc_symbol', how = 'left')

columns = ['hgnc_symbol','hgnc_name','transcript'] + \
    input_columns + output_columns + \
    ['rvis_score','phylop_score','phylop_score_primate']

gnomad_constraint.to_csv('data/constraint/zscores_hgnc.tsv',sep='\t')

# # %% [markdown]
# # Show that their method gives slightly different values to Poisson CIs

# # %%
# n_obs = 33
# n_exp = 22
# alpha = 0.1
# print('CDF method, threshold=2:', estimate_constraint.poisson_ecdf_interval(n_obs, n_exp, cdf_threshold=2))
# print('CDF method, threshold=200:',estimate_constraint.poisson_ecdf_interval(n_obs, n_exp, cdf_threshold=200))
# print('statsmodels:',estimate_constraint.poisson_exact_interval(n_obs, n_exp, alpha=alpha))

# # %% [markdown]
# # Replicate their values

# # %%
# # Poisson ecdf confidence intervals, threshold = 2 
# output_columns = []
# for impact in impacts:
#     est = estimate_constraint.estimate_constraint(gnomad_constraint,f'obs_{impact}',f'exp_{impact}',method='ecdf', kw={'cdf_threshold':2})
#     output_columns_ = [x + f'_{impact}' for x in est.columns]
#     gnomad_constraint[output_columns_] = est
#     output_columns.append(output_columns_)

# output_columns = sum(output_columns, [])
# columns = ['gene','transcript'] + input_columns + output_columns

# gnomad_constraint[columns].to_csv('../data/constraint/gnomad/all_genes_constraint_ecdf_t2.tsv',sep='\t')

# # %%
# gnomad_constraint_precalc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_gnomad.tsv',sep='\t')
# gnomad_constraint_calc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_ecdf_t2.tsv',sep='\t')

# gnomad_constraint_calc.shape, gnomad_constraint_precalc.shape
# gnomad_constraint_compare = pd.merge(gnomad_constraint_precalc, gnomad_constraint_calc, on= ['gene','transcript'], suffixes=('_precalc','_calc'))



# fig, ax = plt.subplots(1, 3)
# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oe_lof_upper',
#     y = 'oeuf_lof',
#     ax = ax[0]
# )

# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oe_mis_upper',
#     y = 'oeuf_mis',
#     ax = ax[1]
# )
# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oe_syn_upper',
#     y = 'oeuf_syn',
#     ax = ax[2]
# )
# plt.tight_layout()

# # %% [markdown]
# # Adjust threshold

# # %%
# # Poisson ecdf confidence intervals, threshold = 200
# output_columns = []
# for impact in impacts:
#     est = estimate_constraint.estimate_constraint(gnomad_constraint,f'obs_{impact}',f'exp_{impact}',method='ecdf', kw={'cdf_threshold':200})
#     output_columns_ = [x + f'_{impact}' for x in est.columns]
#     gnomad_constraint[output_columns_] = est
#     output_columns.append(output_columns_)

# output_columns = sum(output_columns, [])
# columns = ['gene','transcript'] + input_columns + output_columns

# gnomad_constraint[columns].to_csv('../data/constraint/gnomad/all_genes_constraint_ecdf_t200.tsv',sep='\t')

# # %%
# gnomad_constraint_precalc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_gnomad.tsv',sep='\t')
# gnomad_constraint_calc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_ecdf_t200.tsv',sep='\t')

# gnomad_constraint_calc.shape, gnomad_constraint_precalc.shape
# gnomad_constraint_compare = pd.merge(gnomad_constraint_precalc, gnomad_constraint_calc, on= ['gene','transcript'], suffixes=('_precalc','_calc'))



# fig, ax = plt.subplots(1, 3)
# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oe_lof_upper',
#     y = 'oeuf_lof',
#     ax = ax[0]
# )

# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oe_mis_upper',
#     y = 'oeuf_mis',
#     ax = ax[1]
# )
# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oe_syn_upper',
#     y = 'oeuf_syn',
#     ax = ax[2]
# )
# plt.tight_layout()

# # %% [markdown]
# # Compare to exact values

# # %%
# # Exact Poisson confidence intervals, alpha = 0.1
# output_columns = []
# for impact in impacts:
#     est = estimate_constraint.estimate_constraint(gnomad_constraint,f'obs_{impact}',f'exp_{impact}', method='exact')
#     output_columns_ = [x + f'_{impact}' for x in est.columns]
#     gnomad_constraint[output_columns_] = est
#     output_columns.append(output_columns_)

# output_columns = sum(output_columns, [])
# columns = ['gene','transcript'] + input_columns + output_columns

# gnomad_constraint[columns].to_csv('../data/constraint/gnomad/all_genes_constraint_exact_a0.1.tsv',sep='\t')

# # %%
# gnomad_constraint_precalc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_gnomad.tsv',sep='\t')
# gnomad_constraint_calc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_exact_a0.1.tsv',sep='\t')

# gnomad_constraint_calc.shape, gnomad_constraint_precalc.shape
# gnomad_constraint_compare = pd.merge(gnomad_constraint_precalc, gnomad_constraint_calc, on= ['gene','transcript'], suffixes=('_precalc','_calc'))
# fig, ax = plt.subplots(figsize = (6, 4))
# plt.scatter(
#     x = gnomad_constraint_compare.oe_lof_upper,
#     y = gnomad_constraint_compare.oeuf_lof,
#     alpha=0.5,s=1, label = 'pLoF'
# )

# plt.scatter(
#     x = gnomad_constraint_compare.oe_mis_upper,
#     y = gnomad_constraint_compare.oeuf_mis,
#     alpha=0.5,s=1, label = 'Missense'
# )
# plt.scatter(
#     x = gnomad_constraint_compare.oe_syn_upper,
#     y = gnomad_constraint_compare.oeuf_syn,
#     alpha=0.5,s=1, label = 'Synonymous'
# )
# plt.vlines(2, 0.01, 100, color='k',linestyle='dotted', label='ECDF threshold=2')
# plt.tight_layout()
# plt.xlabel('obs/exp upper bound\n(gnomAD eCDF method)')
# plt.ylabel('obs/exp upper bound\n(exact method)')
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()

# # %%
# gnomad_constraint_precalc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_gnomad.tsv',sep='\t')
# gnomad_constraint_calc = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_exact_a0.1.tsv',sep='\t')

# gnomad_constraint_calc.shape, gnomad_constraint_precalc.shape
# gnomad_constraint_compare = pd.merge(gnomad_constraint_precalc, gnomad_constraint_calc, on= ['gene','transcript'], suffixes=('_precalc','_calc'))
# fig, ax = plt.subplots(figsize = (6, 4))
# plt.scatter(
#     x = gnomad_constraint_compare.oe_lof_lower,
#     y = gnomad_constraint_compare.oelf_lof,
#     alpha=0.5,s=1, label = 'pLoF'
# )

# plt.scatter(
#     x = gnomad_constraint_compare.oe_mis_lower,
#     y = gnomad_constraint_compare.oelf_mis,
#     alpha=0.5,s=1, label = 'Missense'
# )
# plt.scatter(
#     x = gnomad_constraint_compare.oe_syn_lower,
#     y = gnomad_constraint_compare.oelf_syn,
#     alpha=0.5,s=1, label = 'Synonymous'
# )
# plt.vlines(2, 0.01, 100, color='k',linestyle='dotted', label='ECDF threshold=2')
# plt.tight_layout()
# plt.xlabel('obs/exp lower bound\n (gnomAD eCDF method)')
# plt.ylabel('obs/exp lower bound\n (exact method)')
# plt.ylim((0,3))
# plt.legend()

# # %%
# gnomad_constraint_exact = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_exact_a0.1.tsv',sep='\t')
# gnomad_constraint_ecdf = pd.read_csv('../data/constraint/gnomad/all_genes_constraint_ecdf_t200.tsv',sep='\t')

# gnomad_constraint_compare = pd.merge(gnomad_constraint_exact, gnomad_constraint_ecdf, 
#     on= ['gene','transcript'], 
#     suffixes=('_ecdf','_exact'))



# fig, ax = plt.subplots(1, 3)
# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oeuf_lof_exact',
#     y = 'oeuf_lof_ecdf',
#     ax = ax[0]
# )

# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oeuf_mis_exact',
#     y = 'oeuf_mis_ecdf',
#     ax = ax[1]
# )
# sns.scatterplot(
#     data = gnomad_constraint_compare,
#     x = 'oeuf_syn_exact',
#     y = 'oeuf_syn_ecdf',
#     ax = ax[2]
# )
# plt.tight_layout()

# # %% [markdown]
# # ## Z scores

# # %%
# def chisq_zscore(count, exposure, alpha):
#     if exposure == 0:
#         return None
#     else:
#         chisq = np.sqrt((count - exposure)**2 / exposure)
#         if count < exposure:
#             return chisq
#         else:
#             return -chisq
# alpha = 0.1

# output_columns = []
# for impact in impacts:
#     print(impact)
#     gnomad_constraint[f'z_{impact}'] = gnomad_constraint.apply(
#         lambda x: chisq_zscore(x[f'obs_{impact}'], x[f'exp_{impact}'], alpha),
#         axis=1)
#     output_columns.append(f'z_{impact}')

# columns = ['gene','transcript'] + input_columns + output_columns

# gnomad_constraint[columns].to_csv('../data/constraint/gnomad/all_genes_constraint_zscores.tsv',sep='\t')


