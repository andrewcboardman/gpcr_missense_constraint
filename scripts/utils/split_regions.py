import hail as hl

def split_table(full_ht, region):
    
    # filter into X, Y and autosomal regions for separate aggregation
    if region == "auto":
        region_ht = full_ht.filter(full_ht.locus.in_autosome_or_par())

    elif region == 'x':
        region_ht = hl.filter_intervals(full_ht, [hl.parse_locus_interval('X')])
        region_ht = region_ht.filter(region_ht.locus.in_x_nonpar())
    
    elif region == 'y':
        region_ht = hl.filter_intervals(full_ht, [hl.parse_locus_interval('Y')])
        region_ht = region_ht.filter(region_ht.locus.in_y_nonpar())

    return region_ht

def main(snakemake):

    paths = dict(
        context_local_path = snakemake.input[0],
        exomes_local_path = snakemake.input[1],
    )

    context_ht = hl.read_table(paths['context_local_path'].replace('.txt.gz','.ht'))
    exomes_ht = hl.read_table(paths['exomes_local_path'].replace('.txt.gz','.ht'))
    
    region = snakemake.params.region

    # split into tables for autosomes, X and Y chromosomes
    context_region_ht = split_table(context_ht, region)
    exomes_region_ht = split_table(exomes_ht, region)

    context_region_ht.write(paths['context_local_path'].replace('.txt.gz',f'_region_{region}.ht'),overwrite=True)
    context_region_ht.export(paths['context_local_path'].replace('.txt.gz',f'_region_{region}.txt.gz'))

    exomes_region_ht.write(paths['exomes_local_path'].replace('.txt.gz',f'_region_{region}.ht'),overwrite=True)
    exomes_region_ht.export(paths['exomes_local_path'].replace('.txt.gz',f'_region_{region}.txt.gz'))

if __name__ == "__main__":
    main(snakemake)