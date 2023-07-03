import pandas as pd

df_iuphar = pd.read_csv('https://www.guidetopharmacology.org/DATA/targets_and_families.tsv', skiprows=1,sep='\t')
print(df_iuphar.head())
df_iuphar = df_iuphar[~df_iuphar['HGNC symbol'].isna()] # remove multi-unit targets
df_iuphar = df_iuphar[~df_iuphar['Human SwissProt'].isna()] # remove pseudogenes
df_iuphar = df_iuphar[~df_iuphar['HGNC   name'].str.contains('pseudogene')] # remove pseudogenes
df_iuphar = df_iuphar.drop_duplicates('HGNC symbol') # remove duplicates

# Rename columns
df_iuphar = df_iuphar[['HGNC symbol', 'HGNC name', 'Type', 'Family name', 'Target name',  'Human SwissProt']]
df_iuphar.columns = ['hgnc_symbol', 'hgnc_name','target_class', 'target_family', 'target_name', 'human_swissprot']

# Check no NAs present
print('NAs',df_iuphar.isna().sum())
print('Total targets', df_iuphar.shape[0])
print('Breakdown by class:\n',df_iuphar.value_counts('target_class'))
df_iuphar.to_csv('data/families/iuphar_targets_clean.csv')