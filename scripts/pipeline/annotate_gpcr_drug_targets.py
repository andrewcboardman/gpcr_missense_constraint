import pandas as pd
import numpy as np

drugs = pd.read_csv('data/drug_targets/gpcr_drugs.tsv',sep='\t')
drugs_approved = drugs[drugs['year of approval']!='-']
drugs_approved = drugs_approved[['gene','drug_name','moa_simplified']]
drugs_approved['moa_simplified'] = drugs_approved.moa_simplified
drug_targets = drugs_approved.groupby('gene').agg({'drug_name':list,'moa_simplified':set})
print(drugs_approved.shape[0], 'drug-target pairs')
print(drugs_approved.drug_name.nunique(), 'unique drugs')

gpcr_human_genes = pd.read_csv('data/families/gpcr_genes_human_gpcrdb.tsv',sep='\t')
gpcr_human_genes = gpcr_human_genes[['gene','entry_name']]
gpcr_drug_targets = gpcr_human_genes.merge(drug_targets, on ='gene',how='left')
print(gpcr_drug_targets.shape[0], 'GPCR-drug-targets')

gpcr_drug_targets['moa_simplified'] = gpcr_drug_targets.moa_simplified.fillna('')
gpcr_drug_targets['moa_simplified'] = gpcr_drug_targets.moa_simplified.apply(lambda x: '_'.join(sorted(list(x))))
print(gpcr_drug_targets.moa_simplified.value_counts())
gpcr_drug_targets.to_csv('data/drug_targets/gpcrs_with_approved_drug_target_status.tsv',sep='\t')

print(gpcr_drug_targets.head())
# gpcr_drug_targets = gpcr_drug_targets[['gene','entry_name','approved_drugs_moa','adverse_event_associations']]
# gpcr_drug_targets['approved_drugs_moa'] = gpcr_drug_targets.approved_drugs_moa.fillna('none')
# gpcr_drug_targets['adverse_event_associations'] = gpcr_drug_targets.adverse_event_associations.fillna('No')
# 