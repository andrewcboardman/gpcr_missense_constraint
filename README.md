
# Mapping patterns of mutational constraint in GPCR drug targets
## Overview
* gnomadIC (gnomAD Inference of Constraint) is a package for estimation of constraint from the gnomAD dataset, using custom annotations and aggregation groups. The constraint computation pipeline is written in [Hail 0.2](https://hail.is) and is based on the workflow from the gnomAD flagship manuscript [Karczewski et al., 2019](https://www.biorxiv.org/content/10.1101/531210v2)
* Analysis can be run using a set of chosen target genes using constraint_analysis.py --targets data/targets/targets.txt. You can either keep standard settings (aggegrating missense variants using provided PolyPhen2 labels) or add your own annotations. Custom annotations must be linked to HGVSP ids for variants of your choice. These can then be used to modify the annotation column before aggregation.
* This repo also contains notebooks required to perform the analyses detailed in our manuscript (Boardman et al, in preparation). 

## Contents

* gnomadIC contains code for custom constraint calculations
* data contains constraint values, gene lists and annotations for analysis
* scripts contains notebooks for analysis of constraint values
* plots and results contain plots and data tables generated by scripts 
* slides contains marp slidedeck overview for this project
## Data summary 
| Description | File |
|---|---|
| Constraint for 19,704 gene regions from gnomAD (v2.1) | `all_genes_gnomad_constraint.tsv` |
| Essential genes from gnomAD  | `all_genes_essential_gnomad.tsv` |
| List of 391 GPCR genes from GPCRdb | `gpcr_genes_human_gpcrdb.tsv` |
| Constraint from gnomAD release for 391 GPCR genes | `gpcr_genes_gnomad_constraint.tsv` |
| Essential GPCR genes from gnomAD paper \& literature | `gpcr_genes_essential_curated.tsv` |
| GPCR genes with disease associations from OMIM & OpenTargets GWAS | `gpcr_genes_disease_associated.tsv`|
| GPCR drug targets with associated adverse events | `gpcr_drug_targets_w_adverse_events.tsv ` |
## Script summary

| Description | Script |
| --- | --- |
| Description of data gathering | `get_data.ipynb` |
| Distribution of constraint in GPCRs | `Fig1_constraint.ipynb` |
| Correlation between constraint and essentiality | `Fig2_essentiality.ipynb` |
| Constraint in drug targets | `Fig3_drug_targets.ipynb` |

## Analysis notebooks
<!-- * `constraint_estimation.py` 
  * Identifies genomic locations and performs custom constraint calculation using curation rules
  * `data/genome_locations/gpcr_gene_locations.tsv` contains genomic locations in Grch38 for 
  * `data/constraint/constraint_by_target_gene_custom.tsv` -->
* `underpowered_genes_by_group.R` counts underpowered genes by group
  * It takes the full set of HGNC genes (`data/hgnc_groups/hgnc_complete_set.txt`) and matches this to the set of HGNC groups (`data/hgnc_groups/hgnc_gene_families.txt`). 
  * It then identifies underpowered genes from the gnomAD summary statistics (`data/constraint/precalc_obs_exp.tsv.gz`) and counts these by group. 
  * It then identifies underpowered GPCR genes using the gene symbol table (`data/gene_lists/gpcr_gene_symbols.tsv`) and counts these by family.



* 
* `check_variants_and_constraint.ipynb`
  * Check that constraint values match gnomAD
* `figure_2_variant_impact.ipynb`
  * Analyse the evidence for constraint against different types of variants, for GPCRs and controls
* `figure_3_constraint_by_family.ipynb`
  * This plots the constraint by gene across GPCR families
* `figure_4_phenotypes_venn_diagram.ipynb`
  * analyses phenotype gene lists and makes Venn diagram
* `figure_5_phenotype_constraint_associations.ipynb`  
  * Compute ROC curves and perform analysis of classification
* `figure_6_case_studies.ipynb`
  * Analyse region-level constraint in selected chemokine receptors