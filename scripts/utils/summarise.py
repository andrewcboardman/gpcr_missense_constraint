import hail as hl
import pandas as pd
import utils



# def summarise(po_ht):
    

#     return constraint_ht


def main(snakemake):
    root = './data'
    paths = dict(
        po_output_path = snakemake.input[0].replace('.txt.gz','.ht')
    )
    
    po_ht = hl.read_table(paths['po_output_path'])

    groups = ('gene','transcript','canonical','impact')
    agg_expr = {
        'obs': hl.agg.sum(po_ht.obs),
        'exp': hl.agg.sum(po_ht.exp),
        'poss': hl.agg.sum(po_ht.poss)
    }

    summary = po_ht.group_by(*groups).aggregate(**agg_expr)
    summary.select_globals().export(snakemake.output[0])


if __name__ == "__main__":
    main(snakemake)

    # Split up into missense and others
    # po_ht_mis = po_ht.filter(po_ht.impact=='mis')
    # po_ht_non_mis = po_ht.filter(po_ht.impact!='mis')

    # # # Custom impact of missense mutations from lookup table
    # # po_ht_mis = po_ht_mis.annotate(
    # #     missense_details=hl.struct(
    # #         **{'wt':po_ht_mis.aa_wt, 'mut':po_ht_mis.aa_mut,'pos':po_ht_mis.aa_pos_start}
    # #         )
    # # )

    # # impact_2 = mis_impact_ht[po_ht.missense_details].impact
    # # po_ht_mis.annotate(impact = impact_2)

    # # Join together again
    # po_ht_final = po_ht_non_mis.union(po_ht_mis)



    # keys = ('gene','transcript','annotation','modifier')
    # variants = variants.key_by(*keys)
    # impacts = impacts.key_by(*keys)
    # annotated_variants = variants.annotate(
    #     {'impact':impacts[variants.key]}
    # )





# def summarise_prop_observed(po_ht, summary_path):
#     """ Function for drawing final inferences from observed and expected variant counts"""

#     # Finish annotation groups    
#     classic_lof_annotations = hl.literal({'stop_gained', 'splice_donor_variant', 'splice_acceptor_variant'})
#     variant_class = (hl.case()
#         .when(classic_lof_annotations.contains(po_ht.annotation) & (po_ht.modifier == 'HC'), 'lof_hc')
#         .when(classic_lof_annotations.contains(po_ht.annotation) & (po_ht.modifier == 'LC'), 'lof_lc')
#         .when((po_ht.annotation == 'missense_variant') & (po_ht.modifier == 'probably_damaging'), 'mis_pphen')
#         .when((po_ht.annotation == 'missense_variant') & (po_ht.modifier != 'probably_damaging'), 'mis_non_pphen')
#         .when(po_ht.annotation == 'synonymous_variant', 'syn')
#         .default('non-coding variants')
#         )
#     po_ht = po_ht.annotate(variant_class = variant_class)
    
#     # Final aggregation by group 
#     groups = ('gene','transcript','canonical','variant_class')    
#     agg_expr = {
#         'obs': hl.agg.sum(po_ht.observed_variants),
#         'exp': hl.agg.sum(po_ht.expected_variants),
#         'oe': hl.agg.sum(po_ht.observed_variants) / hl.agg.sum(po_ht.expected_variants),
#         'adj_mu': hl.agg.sum(po_ht.adjusted_mutation_rate),
#         'raw_mu': hl.agg.sum(po_ht.raw_mutation_rate),
#         'poss': hl.agg.sum(po_ht.possible_variants)
#     }
#     constraint_ht = po_ht.group_by(*groups).aggregate(**agg_expr)
    
#     # calculate confidence intervals, join tables and label
#     constraint_ht = utils.oe_confidence_interval(constraint_ht, constraint_ht.obs, constraint_ht.exp, select_only_ci_metrics=False)
#     constraint_df = constraint_ht.select_globals().to_pandas()
#     constraint_df.to_csv(summary_path.replace('.ht','.csv.gz'),compression='gzip')
#     return constraint_df

