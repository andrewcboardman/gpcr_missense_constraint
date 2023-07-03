import hail as hl


def alter_path(path, table):
    return path.replace('.ht',f'_{table}.ht')


def main(snakemake):
      # Loop over autosomes, x y: aggregate by chosen groupings & get proportion observed
    # if tables == 'all':
    #     tables_ = ('auto','x','y')
    # elif tables in ('auto','x','y'):
    #     tables_ = (tables)
    # else: 
    #     tables_ = ()
    # for table in tables_:
    # # Take union of answers and write to file
    # if tables in ('all','union'):
    
    root = './data'
    paths = dict(
        po_output_path = snakemake.output[0].replace('.txt.gz','.ht'),
    )

    prop_observed = {}
    for region in snakemake.params.regions:
        prop_observed[region] = hl.read_table(alter_path(paths['po_output_path'].replace('.ht','_region.ht'),region))
    prop_observed_ht = (
                prop_observed['auto']
                .union(prop_observed['x'])
                .union(prop_observed['y'])
                )
    prop_observed_ht.write(paths['po_output_path'], overwrite=True)
    prop_observed_ht.export(snakemake.output[0])

if __name__ == "__main__":
    main(snakemake)