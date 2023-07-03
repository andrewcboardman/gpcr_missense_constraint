import pandas as pd
from gnomadIC import *
def setup_paths(run_ID,root = './data'):
    #.src.summariseorage paths 
    gs_paths = dict(
        exomes_path = 'gs://gcp-public-data--gnomad/papers/2019-flagship-lof/v1.0/model/exomes_processed.ht/',
        context_path = 'gs://gcp-public-data--gnomad/resources/context/grch37_context_vep_annotated.ht/',
        mutation_rate_path = 'gs://gcp-public-data--gnomad/papers/2019-flagship-lof/v1.0/model/mutation_rate_methylation_bins.ht',
        po_coverage_path = 'gs://gcp-public-data--gnomad/papers/2019-flagship-lof/v1.0/model/prop_observed_by_coverage_no_common_pass_filtered_bins.ht'
    )
    # Local paths
    output_subdir = f'{root}/{run_ID}'
    if not os.path.isdir(output_subdir):
        os.mkdir(output_subdir)
    local_paths = dict(
        # models - shared between runs
        mutation_rate_local_path = f'{root}/models/mutation_rate_methylation_bins.ht',
        po_coverage_local_path = f'{root}/models/prop_observed_by_coverage_no_common_pass_filtered_bins.ht',
        coverage_models_local_path = f'{root}/models/coverage_models.pkl',
        # outputs - specific to run
        exomes_local_path = f'{output_subdir}/exomes.ht',
        context_local_path = f'{output_subdir}/context.ht',        
        possible_variants_ht_path = f'{output_subdir}/possible_transcript_pop.ht',
        po_output_path = f'{output_subdir}/prop_observed.ht',
        finalized_output_path = f'{output_subdir}/constraint.ht',
        summary_output_path = f'{output_subdir}/constraint_final.csv.gz'
    )
    paths = {**gs_paths, **local_paths}

    return paths


def run_tasks(tasks, targets, paths, dataset, model, annotations, tables='all',download_data=False, rebuild_models=False,overwrite=False):
    '''Runs all requested tasks in specified path'''
    data = {}

    if 'preprocess' in tasks:
        # Load gene intervals
        genes = targets.keys()
        gene_intervals = [hl.parse_locus_interval(interval_txt) for interval_txt in targets.values()]
        print(gene_intervals)
        # If in test mode only load 1 gene
        print('Getting data from Google Cloud...')
        data = preprocess(paths, genes, gene_intervals, dataset, download_data=download_data,overwrite=overwrite)
        print('Variants loaded successfully!')
  
    if 'model' in tasks:
        print('Modelling expected number of variants')
        data = model_variants(paths, tables = tables, rebuild_models=rebuild_models, overwrite=overwrite)
        print('Modelled expected number of variants successfully!')
    
    if 'summarise' in tasks:
        print('Running aggregation by variant classes')
        data = summarise(paths, annotations)
        print('Aggregated variants successfully!')
