
library(data.table)
library(magrittr)


load_and_tidy_constraint <- function() {
  ## Pre-calculated constraint estimates (Grch37)
  precalc_constraint <- fread('data/constraint/precalc_obs_exp.tsv.gz')

  precalc_constraint <- precalc_constraint[
    (gene_type == "protein_coding"),
    .(
      symbol = gene,
      transcript,
      obs_lof,
      exp_lof,
      obs_mis_pphen,
      exp_mis_pphen,
      obs_mis_non_pphen = obs_mis - obs_mis_pphen,
      exp_mis_non_pphen = exp_mis - exp_mis_pphen,
      obs_syn,
      exp_syn,
      cds_length,
      num_coding_exons
      )
    ]

  ### observed variant count
  precalc_obs <- (
    melt(
      precalc_constraint,
      id.vars = c("symbol","transcript"),
      measure.vars = c(
        "obs_lof",
        "obs_mis_pphen",
        "obs_mis_non_pphen",
        "obs_syn")
    )
    [,
      .(
        symbol,transcript,
        obs = nafill(value, fill = 0),
        impact = substr(variable, 5, 100)
      )
    ]
  )

  ### expected variant count
  precalc_exp <- (
    melt(
      precalc_constraint,
      id.vars = c("symbol","transcript"),
      measure.vars = c(
        "exp_lof",
        "exp_mis_pphen",
        "exp_mis_non_pphen",
        "exp_syn")
      )
      [ 
        ,
        .(
          symbol,transcript,
          exp = nafill(value, fill = 0),
          impact = substr(variable, 5, 100)
        )
      ]
  )

  ### size of gene
  gene_size  <-  precalc_constraint[, 
  .(symbol, transcript, cds_length, num_coding_exons)
  ]

  ### Join tables
  precalc_constraint_long <- (
    merge(
      precalc_obs,
      precalc_exp,
      by = c("symbol", "transcript", "impact")
    ) %>%
    merge(
      gene_size,
      by = c("symbol","transcript")
    )
  )
  precalc_constraint_long
  }



run_poisson_tests <- function(tidy_constraint) {
  
  # Run Poisson tests for constraint

  poisson.test.custom  <- function(obs, exp) {
    if (exp <= 0) {
      list(
        bound.lower = NA,
        log.pval.more = NA,
        bound.upper = NA,
        log.pval.less = NA
      )
    }
    else {
      pt.less <- poisson.test(obs, T = exp, alternative = "less", conf.level = 0.95)
      pt.more <- poisson.test(obs, T = exp, alternative = "greater", conf.level = 0.95)
      list(
        oe.bound.lower = pt.more$conf.int[1],
        log.pval.more = log(pt.more$p.val),
        oe.bound.upper = pt.less$conf.int[2],
        log.pval.less = log(pt.less$p.val)
      )
    }
  }

  tidy_constraint$oe <- (
    tidy_constraint$obs / tidy_constraint$exp
  )

  poisson.test.results <- (
    mapply(
        poisson.test.custom,
        tidy_constraint$obs,
        tidy_constraint$exp,
        SIMPLIFY = F
      ) %>%
    do.call(rbind, .) %>%
    cbind(tidy_constraint, .)
  )
  poisson.test.results
}

main <- function() {
  download.file(
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz",
    "data/constraint/precalc_obs_exp.tsv.gz"
  )
  constraint <- load_and_tidy_constraint()
  constraint <- run_poisson_tests(constraint)

  fwrite(
    constraint,"data/constraint/precalc_constraint.tsv", sep = "\t"
    )
  constraint <- fread(
    "data/constraint/precalc_constraint.tsv",
    sep = "\t"
    )
  # gpcr_gene_symbols  <- fread(
  #   "data/gene_lists/gpcr_gene_symbols.tsv",
  #   sep = "\t"
  #   )
  # constraint  <- constraint[,
  #   .(Grch37_symbol=symbol, 
  #   transcript,impact,
  #   obs, exp, oe,
  #   oeuf=oe.bound.upper,oelf=oe.bound.lower,
  #   log.pval.less,log.pval.more)
  # ]
  # gpcrs_w_constraint <- merge(
  #   gpcr_gene_symbols, 
  #   constraint, 
  #   by = "Grch37_symbol"
  #   )
  
  # # Remove 1 transcript for LTB4R2 which has 2 'canonical' transcripts
  # gpcrs_w_constraint <- gpcrs_w_constraint[!((symbol == "LTB4R2") & (transcript == "ENST00000528054"))]
  # fwrite(
  #   gpcrs_w_constraint,
  #   "./data/constraint/gpcrs_precalc_constraint.tsv",
  #   sep="\t"
  # )

  print("Written constraint to \"data/constraint/gpcrs_precalc_constraint.tsv\"")

}

if (sys.nframe() == 0) {
  main()
}