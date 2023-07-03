import hail as hl
import utils

def prepare_ht(ht, trimer: bool = False, annotate_coverage: bool = True):
    if trimer:
        ht = utils.trimer_from_heptamer(ht)
    str_len = 3 if trimer else 7

    if isinstance(ht, hl.Table): 
        ht = ht.annotate(ref=ht.alleles[0], alt=ht.alleles[1])
        ht = ht.filter((hl.len(ht.ref) == 1) & (hl.len(ht.alt) == 1) & ht.context.matches(f'[ATCG]{{{str_len}}}'))
        ht = utils.annotate_variant_types(utils.collapse_strand(ht), not trimer)
    else: # handle case where ht is a matrix table
        ht = ht.annotate_rows(ref=ht.alleles[0], alt=ht.alleles[1])
        ht = ht.filter_rows((hl.len(ht.ref) == 1) & (hl.len(ht.alt) == 1) & ht.context.matches(f'[ATCG]{{{str_len}}}'))
        ht = utils.annotate_variant_types(utils.collapse_strand(ht), not trimer)
    annotation = {
        'methylation_level': hl.case().when(
            ht.cpg & (ht.methylation.MEAN > 0.6), 2
        ).when(
            ht.cpg & (ht.methylation.MEAN > 0.2), 1
        ).default(0)
    }
    if annotate_coverage:
        annotation['exome_coverage'] = ht.coverage.exomes.median
    return ht.annotate(**annotation) if isinstance(ht, hl.Table) else ht.annotate_rows(**annotation)


def preprocess_ht(ht, model='standard', trimer=True):
    
    ht = utils.prepare_ht(ht, trimer=trimer)

    # Extract relevant parts of VEP struct and set as groupings for annotation join
    ht = utils.add_most_severe_csq_to_tc_within_ht(ht)
    ht = ht.transmute(transcript_consequences=ht.vep.transcript_consequences) 
    ht = ht.explode(ht.transcript_consequences)
    ht, groupings = utils.annotate_constraint_groupings(ht, model)
    
    # Extract coverage as exome coverage
    ht = ht.transmute(coverage=ht.exome_coverage)

    return ht, groupings


def main(snakemake):
    '''
    This is the new master function for loading all necessary data for constraint analysis on the given genes
    Paths are passed in from the main program. 
    The exomes and context data should always be downloaded as new gene intervals are passed in. 
    The mutation rate by methylation and proportion observed by coverage tables are stored locally.
    They should be downloaded if not present but the control flow to do this isn't yet implemented 
    '''
    paths = dict(
        context_full_local_path = snakemake.input[0],
        exomes_full_local_path = snakemake.input[1],
        context_local_path = snakemake.output[0],
        exomes_local_path = snakemake.output[1],
    )

    model = 'standard'
    trimer=True
    fields = ['ref', 'alt', 'context', 'methylation_level', 'coverage'] 


    context_full_ht = hl.read_table(paths['context_full_local_path'].replace('.txt.gz','.ht'))
    context_ht, groupings = prepare_ht(context_full_ht, model=model, trimer=trimer)
    context_ht = context_ht.select(*fields, *groupings)
    context_ht.write(paths['context_local_path'].replace('.txt.gz','.ht'),overwrite=True)
    context_ht.export(paths['context_local_path'])

    # Select fields for modelling  
    exomes_full_ht = hl.read_table(paths['exomes_full_local_path'].replace('.txt.gz','.ht'))   
    additional_fields = ['freq', 'filters', 'AC_ex_mnv']
    exomes_ht = prepare_ht(exomes_full_ht, model=model, trimer=trimer)
    exomes_ht = exomes_ht.select(*fields,*groupings,*additional_fields)  
    exomes_ht.write(paths['exomes_local_path'].replace('.txt.gz','.ht'),overwrite=True)
    exomes_ht.export(paths['exomes_local_path'])


if __name__ == "__main__":
    main(snakemake)