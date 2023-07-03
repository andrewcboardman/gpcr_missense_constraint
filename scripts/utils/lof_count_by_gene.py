import hail as hl; hl.init()

context = hl.read_table('../data/gnomad_standard/context.ht')
lof_context = context.filter(context.modifier == 'HC')
lof_context.write('../data/possible_lof.ht')
lof_context = hl.read_table('../data/gene_constraint/possible_lof.ht')
lof_types_count = lof_context.group_by('gene','transcript','canonical','annotation').aggregate(variant_count=hl.agg.count())
df_lof_types_count = lof_types_count.to_pandas().to_csv('../data/gene_constraint/possible_lof.csv')