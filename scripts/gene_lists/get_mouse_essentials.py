import pandas as pd
import os

lethal_mouse_gene_symbols = pd.read_csv('https://github.com/broadinstitute/gnomad_lof/raw/master/R/ko_gene_lists/list_mouse_lethal_genes.tsv',sep='\t',header=None)
lethal_mouse_gene_symbols.to_csv(
    'data/essential_genes/mouse_knockout_lethal_genes.txt',
    index=False,header=False)

column_names = ['gene','gene_id','mouse_gene_symbol','mouse_gene_id','mouse_phenotype_associations']
MGI_lookup_path = 'data/mappings/impc_mouse_human_orthologs.txt'
# Convert mouse genes to human genes using MGI homolog list http://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt
if not os.path.exists(MGI_lookup_path):
    MGI_homologs_with_phenotypes = pd.read_csv(
        MGI_lookup_path, sep='\t',header=None, names = column_names, usecols=range(5))
else:
    MGI_homologs_with_phenotypes = pd.read_csv(
        'http://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt',
        sep='\t',header=None, names = column_names, usecols=range(5))
    MGI_homologs_with_phenotypes.to_csv(MGI_lookup_path)
print(lethal_mouse_gene_symbols.columns)
MGI_homologs_lethal = MGI_homologs_with_phenotypes[
    MGI_homologs_with_phenotypes.mouse_gene_symbol.isin(lethal_mouse_gene_symbols[0].tolist())
    ]
mouse_lethals = list(MGI_homologs_lethal.gene.unique())
print(len(mouse_lethals))
with open('data/essential_genes/human_mouse_knockout_lethal_genes.txt','w') as fid:
    fid.writelines('\n'.join(mouse_lethals))
    
