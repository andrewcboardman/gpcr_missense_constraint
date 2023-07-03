import pandas as pd

clinvar_df = pd.read_csv('data/phenotypes/ClinVar_SNVs_plp_20230503.csv')
clinvar_genes = clinvar_df['GeneSymbol'].drop_duplicates().sort_values()
clinvar_genes = clinvar_genes[~clinvar_genes.str.contains(',')]
clinvar_genes.to_csv('data/phenotypes/clinvar_genes.txt', index=False, header=False)
print(len(clinvar_genes))

