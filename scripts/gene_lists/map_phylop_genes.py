import pandas as pd

phylop = pd.read_csv('data/constraint/phylop_grch38.tsv', sep='\t',index_col=False)
# get unique gene symbols
phylop['gene_name'].drop_duplicates().to_csv('data/mappings/phylop_gene_symbols.txt', sep='\t', index=False, header=False)
# print number of phylop scores
print(phylop.shape[0], 'gene symbols with PhyloP scores')
# load phylop hgnc lookup
phylop_lookup = pd.read_csv('data/mappings/phylop_hgnc_lookup.csv', skiprows=1)
# print number of mappings
print(phylop_lookup.shape[0], 'mappings found')
# merge phylop and phylop_lookup
phylop_merge = phylop.merge(phylop_lookup, left_on='gene_name', right_on='Input', how='left')
# record genes with no match
phylop_merge[phylop_merge['Match type'].isna()]['gene_name'].drop_duplicates().to_csv('data/mappings/phylop_missing_symbols.txt', sep='\t', index=False, header=False)
# remove rows with missing matches
phylop_merge = phylop_merge[~phylop_merge['Match type'].isna()]
phylop_merge_approved = phylop_merge[phylop_merge['Match type'] == 'Approved symbol']
phylop_merge_previous = phylop_merge[phylop_merge['Match type'] == 'Previous symbol']
phylop_merge_previous = phylop_merge_previous[~phylop_merge_previous['Input'].isin(phylop_merge_approved['Input'].tolist())]
phylop_merge = pd.concat([phylop_merge_approved, phylop_merge_previous])
phylop_merge = phylop_merge.drop_duplicates(subset=['gene_name'])
print(phylop_merge.shape[0], 'gene symbols with phylop scores and HGNC mappings')
phylop_merge = phylop_merge[[
    'Approved symbol', 'Approved name','gene_name',
    'fracCdsCons', 'fracConsPr', ]]
phylop_merge.columns = ['hgnc_symbol', 'hgnc_name', 'phylop_gene_symbol','phylop_score', 'phylop_score_primate']
phylop_merge.to_csv('data/constraint/phylop_hgnc.tsv', sep='\t', index=False)
phylop[~phylop['gene_name'].isin(phylop_merge['phylop_gene_symbol'].tolist())]['gene_name'].to_csv('data/mappings/phylop_missing.txt', sep='\t', index=False,header=False)