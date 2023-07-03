import pandas as pd

#########################################################
# Read constraint and annotate with mouse lethal and ClinVar pathogenic
constraint = pd.read_csv('data/constraint/zscores_hgnc.tsv', sep='\t')
constraint = constraint.drop_duplicates(subset=['hgnc_symbol'])


mouse_essential_genes = pd.read_csv('data/essential_genes/human_mouse_knockout_lethal_genes.txt',header=None)[0].to_list()
constraint['mouse_lethal'] = constraint.hgnc_symbol.isin(mouse_essential_genes)

clinvar_genes = pd.read_csv('data/phenotypes/clinvar_genes.txt', sep='\t',header=None)[0].to_list()
constraint['clinvar_pathogenic'] = constraint.hgnc_symbol.isin(clinvar_genes)

print('Total',constraint.shape[0])
print('Mouse knockout lethal',constraint.mouse_lethal.sum())
print('ClinVar pathogenic',constraint.clinvar_pathogenic.sum())
print('Mouse knockout lethal or ClinVar pathogenic', (constraint.mouse_lethal | constraint.clinvar_pathogenic).sum())


# #########################################################
# # Add GPCR annotation
# constraint = pd.read_csv('data/constraint/oeuf_hgnc_with_mouse_lethal_and_clinvar.tsv', sep='\t')

# gpcr_genes = pd.read_csv('data/families/gpcr_genes_human_gpcrdb.tsv', sep='\t')
# constraint['is_gpcr'] = constraint.hgnc_symbol.isin(gpcr_genes['gene'].tolist())

# df_count_gpcr = pd.DataFrame(dict(
#       N_total=constraint.is_gpcr.value_counts(),
#     N_mouse_lethal=constraint[constraint.mouse_lethal].is_gpcr.value_counts(),
#     N_ClinVar_path=constraint[constraint.clinvar_pathogenic].is_gpcr.value_counts(),
#     N_mouse_lethal_or_ClinVar_path = constraint[
#         constraint.mouse_lethal | constraint.clinvar_pathogenic
#         ].is_gpcr.value_counts()
# ))
# df_count_gpcr.index = df_count_gpcr.index.map(dict(zip([True, False],['GPCRs','non-GPCRs'])))
# df_count_gpcr = df_count_gpcr.T
# df_count_gpcr['Total'] = df_count_gpcr.sum(axis=1)
# df_count_gpcr.to_csv('results/oeuf_hgnc_with_mouse_lethal_and_clinvar_gpcrs_counts.tsv', sep='\t')
# print(df_count_gpcr)

#########################################################
# Add IUPhar families 

constraint = pd.read_csv('data/constraint/oeuf_hgnc_with_mouse_lethal_and_clinvar.tsv', sep='\t')
iuphar_families = pd.read_csv('data/families/iuphar_targets_clean.csv',index_col=0)
iuphar_families = iuphar_families[['hgnc_symbol','target_class','target_family']]
constraint = constraint.merge(iuphar_families, on='hgnc_symbol', how='left')
constraint[['target_class','target_family']] = constraint[['target_class','target_family']].fillna('None')
constraint.to_csv('data/constraint/zscores_hgnc_iuphar.tsv', sep='\t', index=False)
df_count_iuphar = pd.DataFrame(dict(
    N_total=constraint.target_class.value_counts(),
    N_mouse_lethal=constraint[constraint.mouse_lethal].target_class.value_counts(),
    N_ClinVar_path=constraint[constraint.clinvar_pathogenic].target_class.value_counts(),
    N_mouse_lethal_or_ClinVar_path = constraint[
        constraint.mouse_lethal | constraint.clinvar_pathogenic
        ].target_class.value_counts()
))
df_count_iuphar = df_count_iuphar.T
df_count_iuphar['Total'] = df_count_iuphar.sum(axis=1)
df_count_iuphar.to_csv('results/oeuf_hgnc_with_mouse_lethal_and_clinvar_iuphar_families_counts.tsv', sep='\t')
print(df_count_iuphar)