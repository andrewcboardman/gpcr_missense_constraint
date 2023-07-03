import os
import requests
import pandas as pd
import shutil
import gzip

# Get constraint for all gene regions in gnomad from gnomad summary statistics
gnomad_constraint_path = "data/constraint/oeuf_grch37.tsv"
if not os.path.exists(gnomad_constraint_path):
    r = requests.get("https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz",stream=True)
    if r.status_code == 200:
        with open(gnomad_constraint_path,'wb') as f:
            r.raw.decode_content = True  # just in case transport encoding was applied
            gzip_file = gzip.GzipFile(fileobj=r.raw)
            shutil.copyfileobj(gzip_file, f)
            

oeuf_scores = pd.read_csv(gnomad_constraint_path,sep='\t')

if not os.path.exists('data/mappings/oeuf_hgnc_lookup.csv'):
    # get unique gene symbols
    oeuf_scores['gene'].drop_duplicates().to_csv('data/mappings/oeuf_gene_symbols.txt', sep='\t', index=False, header=False)
else:
    # print number of oeuf scores
    print(oeuf_scores.shape[0], 'gene symbols with OEUF scores')
    # load oeuf hgnc lookup
    oeuf_lookup = pd.read_csv('data/mappings/oeuf_hgnc_lookup.csv', skiprows=1)
    
    # merge oeuf and oeuf_lookup
    oeuf_merge = oeuf_scores.merge(oeuf_lookup, left_on='gene', right_on='Input', how='left')
    # remove rows with missing matches
    oeuf_merge = oeuf_merge[~oeuf_merge['Match type'].isna()]
    oeuf_merge_approved = oeuf_merge[oeuf_merge['Match type'] == 'Approved symbol']
    oeuf_merge_previous = oeuf_merge[oeuf_merge['Match type'] == 'Previous symbol']
    oeuf_merge_previous = oeuf_merge_previous[
        ~oeuf_merge_previous['Input'].isin(oeuf_merge_approved['Input'].tolist())
        ]
    oeuf_merge = pd.concat([oeuf_merge_approved, oeuf_merge_previous])
    oeuf_merge = oeuf_merge.drop_duplicates(subset=['gene'])
    print(oeuf_merge.shape[0], 'gene symbols with OEUF scores and HGNC mappings')
    # write unmapped genes to file
    oeuf_genes_unmapped = pd.Series([gene for gene in oeuf_scores['gene'].tolist() \
        if gene not in oeuf_merge['gene'].tolist()])
    oeuf_genes_unmapped.to_csv('data/mappings/oeuf_missing.txt', sep='\t', index=False,header=False)
    
    # save oeuf_merge
    oeuf_merge = oeuf_merge[[
        'Approved symbol', 'Approved name', 
        'gene', 'transcript', 
        'obs_mis', 'exp_mis', 
        'obs_mis_pphen', 'exp_mis_pphen',
        'obs_syn', 'exp_syn', 'oe_syn', 
        'obs_lof', 'exp_lof', 'oe_lof', 
        'oe_mis_lower', 'oe_mis_upper', 
        'oe_lof_lower', 'oe_lof_upper',
        'oe_syn_lower', 'oe_syn_upper',
        'syn_z', 'mis_z', 'lof_z', 
        'transcript_type', 'cds_length', 'num_coding_exons',
        ]]
    oeuf_merge = oeuf_merge.rename(columns={
        'Approved symbol':'hgnc_symbol', 
        'Approved name': 'hgnc_name',
    })
    oeuf_merge = oeuf_merge.sort_values(by=['hgnc_symbol'])
    oeuf_merge.to_csv('data/constraint/oeuf_hgnc.tsv', sep='\t', index=False)
    