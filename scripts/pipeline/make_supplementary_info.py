import pandas as pd
# %% [markdown]
# # Make supplementary information

# %%
gpcrs_with_families = pd.read_csv('../data/gpcr_genes_human_gpcrdb.tsv',sep='\t')
gpcrs_with_constraint = pd.read_csv('../data/gpcr_genes_constraint_gnomad.tsv',sep='\t',index_col=0)
gpcrs_with_phenotypes = pd.read_csv('../data/gpcrs_with_phenotypes.tsv',sep='\t',index_col=0)
gpcrs_with_drug_targets = pd.read_csv('../data/gpcrs_with_approved_drug_target_status.tsv',sep='\t',index_col=0)

with pd.ExcelWriter('../data/Supplementary_Data.xlsx') as xls:
    gpcrs_with_families.to_excel(xls, sheet_name='Supplementary Table 1',index=False)
    gpcrs_with_constraint.to_excel(xls, sheet_name='Supplementary Table 2',index=False)
    gpcrs_with_phenotypes.to_excel(xls, sheet_name='Supplementary Table 3',index=False)
    gpcrs_with_drug_targets.to_excel(xls, sheet_name='Supplementary Table 4',index=False)