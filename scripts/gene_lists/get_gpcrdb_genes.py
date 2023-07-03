import pandas as pd
import requests

receptors = requests.get('https://gpcrdb.org/services/receptorlist/').json()
print(len(receptors))
human_receptors = [receptor for receptor in receptors 
    if (receptor['species'] == 'Homo sapiens')]
print(len(human_receptors))
human_nonolf_receptors = [receptor for receptor in human_receptors 
    if (receptor['receptor_family'] not in ['Olfactory'])]
print(len(human_nonolf_receptors))
human_receptors = pd.DataFrame(human_nonolf_receptors)
human_receptors = human_receptors[~human_receptors.entry_name.duplicated()]
print(human_receptors.shape[0])

human_receptors.to_csv('data/families/gpcr_genes_human_gpcrdb.tsv',sep='\t')