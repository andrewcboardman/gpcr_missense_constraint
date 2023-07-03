import pandas as pd

# read the rvis_grch37.tsv file into a pandas dataframe and store it in the rvis_scores variable
rvis_scores = pd.read_csv('data/constraint/rvis_grch37.tsv', sep='\t')
# extract unique gene symbols in the CCDSr20 column of the rvis_scores dataframe and write them to a new TSV file
rvis_scores['CCDSr20'].drop_duplicates().to_csv('data/mappings/rvis_gene_symbols.txt', sep='\t', index=False, header=False)
# print the number of rows (gene symbols) in the rvis_scores dataframe
print(rvis_scores.shape[0], 'gene symbols with RVIS scores')
# read the rvis_hgnc_lookup.csv file into a pandas dataframe and store it in the rvis_lookup variable, skipping the first row
rvis_lookup = pd.read_csv('data/mappings/rvis_hgnc_lookup.csv', skiprows=1)
# print the number of rows (mappings) in the rvis_lookup dataframe
print(rvis_lookup.shape[0], 'mappings found')
# merge the rvis_scores and rvis_lookup dataframes on their respective CCDSr20 and Input columns, using left join method
# and store the resulting dataframe in the rvis_merge variable
rvis_merge = rvis_scores.merge(rvis_lookup, left_on='CCDSr20', right_on='Input', how='left')
# remove rows with missing values in the Match type column of the rvis_merge dataframe and store the resulting dataframe in rvis_merge
rvis_merge = rvis_merge[~rvis_merge['Match type'].isna()]
# create a new dataframe (rvis_merge_approved) containing only rows with Match type = 'Approved symbol' from the rvis_merge dataframe
rvis_merge_approved = rvis_merge[rvis_merge['Match type'] == 'Approved symbol']
# create a new dataframe (rvis_merge_previous) containing only rows with Match type = 'Previous symbol' from the rvis_merge dataframe
rvis_merge_previous = rvis_merge[rvis_merge['Match type'] == 'Previous symbol']
# remove rows from rvis_merge_previous that are already present in rvis_merge_approved, based on the Input column, and store the resulting dataframe in rvis_merge_previous
rvis_merge_previous = rvis_merge_previous[~rvis_merge_previous['Input'].isin(rvis_merge_approved['Input'].tolist())]
# concatenate rvis_merge_approved and rvis_merge_previous dataframes into a single dataframe and store it in rvis_merge
rvis_merge = pd.concat([rvis_merge_approved, rvis_merge_previous])
# remove duplicate rows in rvis_merge based on the CCDSr20 column and store the resulting dataframe in rvis_merge
rvis_merge = rvis_merge.drop_duplicates(subset=['CCDSr20'])
# select specific columns from the rvis_merge dataframe, rename them, and write the resulting dataframe to a new TSV file
rvis_merge = rvis_merge[['Approved symbol', 'Approved name', 'CCDSr20', 'RVIS[pop_maf_0.05%(any)]']]
rvis_merge.columns = ['hgnc_symbol', 'hgnc_name', 'previous_symbol', 'rvis_score']
rvis_merge.to_csv('data/constraint/rvis_hgnc.tsv', sep='\t', index=False)
print(rvis_merge.shape[0], 'gene symbols with RVIS scores and HGNC mappings')
rvis_scores[~rvis_scores['CCDSr20'].isin(rvis_merge['previous_symbol'].tolist())]['CCDSr20'].to_csv('data/mappings/rvis_missing.txt', sep='\t', index=False,header=False)