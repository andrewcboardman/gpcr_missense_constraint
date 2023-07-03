import time
import requests
import json
from tqdm import tqdm
import pandas as pd


def get_residues(protein):
    # fetch structures
    url = f'https://gpcrdb.org/services/residues/{protein}'
    response = requests.get(url).json()
    output = {'protein':protein,'residues':response}
    return output

def main():
    genes = pd.read_csv('data/genes/target_gene_names_combined.csv')
    proteins = genes['uniprot_name'].str.lower().unique()

    residues = []
    for protein in tqdm(proteins):
        residues.append(get_residues(protein))
        time.sleep(0.1)
        
    with open('data/protein_sequences/GPCRdb_generic_labelled_residues.json','w') as fid:
        json.dump(residues,fid)

if __name__ == '__main__':
    main()
