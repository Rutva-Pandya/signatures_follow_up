library(tidyverse)
library(lme4)
library(lmerTest)
library(broom)
library(bayestestR)

# global variables
MODELS <- c(
  "gpt2", "gpt2-medium", "gpt2-xl",
  "Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf",
  "Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.1-405B",
  "gemma-2-2b", "gemma-2-9b", "gemma-2-27b",
  "OLMo-2-1124-7B", "OLMo-2-1124-13B", "OLMo-2-0325-32B",
  "Falcon3-1B-Base", "Falcon3-3B-Base", "Falcon3-10B-Base",
  "pythia-160m", "pythia-1.4b", "pythia-6.9b", "pythia-12b",
  "mamba-130m-hf", "mamba-370m-hf", "mamba-790m-hf", "mamba-1.4b-hf",
  "rwkv-4-169m-pile", "rwkv-4-430m-pile", "rwkv-4-1b5-pile"
)

OUT_DIR <- "analysis/r_outputs"

PROCESS_METRICS <- list(
  "auc_entropy",
  "layer_biggest_change_entropy",
  "auc_rank_correct",
  "layer_biggest_change_rank_correct",
  "auc_logprob_correct",
  "layer_biggest_change_logprob_correct",
  "auc_logprobdiff_pos",
  "auc_logprobdiff_neg",
  "layer_biggest_change_logprobdiff",
  "auc_boosting_pos",
  "auc_boosting_neg",
  "layer_argmax_boosting",
  "twostage_magnitude",
  "twostage_magnitude_latter34",
  "twostage_layer"
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions ----

get_nonNan_metrics <- function(metrics, df) {
  good_metrics <- list()
  for (i in 1:length(metrics)) {
    metric <- metrics[[i]]
    if (all(is.na(df[[metric]]))) {
      message(sprintf("Removing the following metric (all NA): %s", metric))
    }
    else {
      good_metrics <- append(good_metrics, metric)
    }
  }
  return(good_metrics)
}

load_data <- function(file_path) {
  read_csv(file_path) |>
    mutate(
      # ~~~~~~~~~~~~~~~~~~ STATIC output (final layer) measures
      output_entropy                = scale(output_entropy),
      output_rank_correct        = scale(output_rank_correct),
      output_logprob_correct        = scale(output_logprob_correct),
      output_logprobdiff            = scale(output_logprobdiff),

      # ~~~~~~~~~~~~~~~~~~ STATIC intermediate (midpoint layer) measures
      midpoint_entropy                = scale(midpoint_entropy),
      midpoint_rank_correct        = scale(midpoint_rank_correct),
      midpoint_logprob_correct        = scale(midpoint_logprob_correct),
      midpoint_logprobdiff            = scale(midpoint_logprobdiff),

      # ~~~~~~~~~~~~~~~~~~ DYNAMIC processing measures
      # Uncertainty
      auc_entropy                   = scale(auc_entropy),
      layer_biggest_change_entropy        = scale(layer_biggest_change_entropy),

      # Confidence in correct answer
      auc_rank_correct               = scale(auc_rank_correct),
      auc_logprob_correct            = scale(auc_logprob_correct),
      layer_biggest_change_rank_correct    = scale(layer_biggest_change_rank_correct),
      layer_biggest_change_logprob_correct = scale(layer_biggest_change_logprob_correct),

      # Confidence in correct answer relative to intuitive
      # (based on probability differences)
      auc_logprobdiff_pos           = scale(auc_logprobdiff_pos),
      auc_logprobdiff_neg           = scale(auc_logprobdiff_neg),
      layer_biggest_change_logprobdiff    = scale(layer_biggest_change_logprobdiff),

      # Metrics given CONTROL prefix

      # THESE PREDICTORS ARE CONSTANT ACROSS ITEMS
      # control_auc_entropy     = scale(control_auc_entropy),
      # control_layer_biggest_change_entropy  = scale(control_layer_biggest_change_entropy),
      
      # control_auc_rank_correct     = scale(control_auc_rank_correct),
      # control_layer_biggest_change_rank_correct     = scale(control_layer_biggest_change_rank_correct),
      # control_auc_logprob_correct     = scale(control_auc_logprob_correct),
      # control_layer_biggest_change_logprob_correct     = scale(control_layer_biggest_change_logprob_correct),
      # control_auc_logprobdiff_pos     = scale(control_auc_logprobdiff_pos),
      # control_auc_logprobdiff_neg     = scale(control_auc_logprobdiff_neg),
      # control_layer_biggest_change_logprobdiff     = scale(control_layer_biggest_change_logprobdiff),

      # Boosting of correct answer relative to intuitive
      auc_boosting_pos            = scale(auc_boosting_pos),
      auc_boosting_neg            = scale(auc_boosting_neg),
      layer_argmax_boosting = scale(layer_argmax_boosting),

      # Two-stage processing
      twostage_magnitude            = scale(twostage_magnitude),
      twostage_magnitude_latter34            = scale(twostage_magnitude_latter34),
      twostage_layer            = scale(twostage_layer)
    )
}

load_vision_data <- function(file_path) {
  read_csv(file_path) |> drop_na() |>
    mutate_if(is.character, as.factor) |>
    mutate(
      # ~~~~~~~~~~~~~~~~~~ STATIC output (final layer) measures
      output_entropy                = scale(output_entropy),
      output_rank_correct        = scale(output_rank_correct),
      output_logprob_correct        = scale(output_logprob_correct),

      # ~~~~~~~~~~~~~~~~~~ STATIC intermediate (midpoint layer) measures
      midpoint_entropy                = scale(midpoint_entropy),
      midpoint_rank_correct        = scale(midpoint_rank_correct),
      midpoint_logprob_correct        = scale(midpoint_logprob_correct),

      # ~~~~~~~~~~~~~~~~~~ DYNAMIC processing measures
      # Uncertainty
      auc_entropy                   = scale(auc_entropy),
      layer_biggest_change_entropy  = scale(layer_biggest_change_entropy),

      # Confidence in correct answer
      auc_rank_correct               = scale(auc_rank_correct),
      auc_logprob_correct            = scale(auc_logprob_correct),
      layer_biggest_change_rank_correct    = scale(layer_biggest_change_rank_correct),
      layer_biggest_change_logprob_correct = scale(layer_biggest_change_logprob_correct)
    )
}

get_baseline_var_str <- function(df, task, baseline_iv, vision_dataset=NA) {
  if (task == "vision") {
    if (baseline_iv == "baseline_final") {
      metrics <- list(
        "output_entropy",
        "output_rank_correct",
        "output_logprob_correct"
      )
    }
    else if (baseline_iv == "baseline_midpoint") {
      metrics <- list(
        "midpoint_entropy",
        "midpoint_rank_correct",
        "midpoint_logprob_correct"
      )
    }
    
  }
  else {
    if (baseline_iv == "baseline_final") {
      metrics <- list(
        "output_entropy",
        "output_rank_correct",
        "output_logprob_correct",
        "output_logprobdiff"
      )
    }
    else if (baseline_iv == "baseline_midpoint") {
      metrics <- list(
        "midpoint_entropy",
        "midpoint_rank_correct",
        "midpoint_logprob_correct",
        "midpoint_logprobdiff"
      )
    }
  }
  
  # get variables that aren't all NaN
  baseline_metrics <- get_nonNan_metrics(metrics, df)

  # get string defining baseline variables
  baseline_var_str <- paste(baseline_metrics, collapse = " * ")

  # additionally include condition-related variables
  if (task == "animals") {
    baseline_var_str <- paste("group", baseline_var_str, sep=" + ")
  }
  else if (task == "vision") {
    # drop condition
    if (!vision_dataset %in% c("cue_conflict", "edge", "silhouette", "sketch", "stylized")) {
      baseline_var_str <- paste("condition", baseline_var_str, sep=" + ")
    }
  }
  
  # additionally include random intercepts
  if (task != "syllogism") {
    # add random intercept for participants
    # (unfortunately, we don't have participant identifiers for syllogism)
    baseline_var_str <- sprintf("%s + (1 | subject_id)", baseline_var_str)
  }
  else {
    # we do have within-item variation for syllogism, but not the other tasks
    baseline_var_str <- sprintf("%s + (1 | syllogism_name)", baseline_var_str)
  }

  return(baseline_var_str)
}

fit_model <- function(df, dv, iv, task, baseline_iv="baseline_final", family=NA, use_glmer=F, log_y=F, vision_dataset=NA) {
  # log-transform the DV, if specified
  dv_str <- ifelse(log_y, sprintf("log(%s)", dv), dv)

  if (iv == "baseline_final" | iv == "baseline_midpoint") {
    stopifnot(baseline_iv == iv)
  }

  # get full string of formula, depending on which baseline we are using
  baseline_var_str <- get_baseline_var_str(df, task, baseline_iv, vision_dataset=vision_dataset)
  if (iv == "baseline_final" | iv == "baseline_midpoint") {
    f_str <- sprintf("%s ~ %s", dv_str, baseline_var_str)
  }
  else {
    f_str <- sprintf("%s ~ %s + %s", dv_str, iv, baseline_var_str)
  }
  message(sprintf("Formula: %s", f_str))
  f <- formula(f_str)
  
  # fit model
  if (use_glmer) {
    m <- glmer(f, df, family=family, control=glmerControl(optimizer="bobyqa"))
  }
  else {
    m <- lmer(f, df, na.action=na.omit, REML=F)
  }
  return(m)
}

fit_single <- function(file_ID, df, dv, iv, baseline_iv, ...) {
  # Define path to cached model.
  model_path <- sprintf("analysis/r_models/%s/%s_%s.rds", baseline_iv, file_ID, iv)

  # Try to load cached model if it exists, otherwise fit it from scratch.
  if (file.exists(model_path)) {
    message(sprintf("Reading model from %s", model_path))
    fit <- readRDS(model_path)
  }
  else {
    fit <- NA
    tryCatch(
      expr = {
        fit <- fit_model(df, dv, iv, baseline_iv=baseline_iv, ...)
      },
      error = function(e){
        print(e)
      }
    )
    if (!is.na(fit)) {
      saveRDS(fit, file=model_path)
    }
  }
  return(fit)
}

get_all_fits <- function(file_ID, df, dv, iv_list, ...) {
  baseline_ivs <- c("baseline_final", "baseline_midpoint")

  fit_list <- list()
  for (i in 1:length(iv_list)) {
    iv <- iv_list[[i]]

    if (iv %in% baseline_ivs) {
      # For baseline IVs, just fit the baseline model.
      baseline_fit <- fit_single(file_ID, df, dv, iv, iv, ...)
      fit_list <- append(
        fit_list, 
        list(list("model"=baseline_fit, "iv"=iv, "baseline_type"=iv, "is_baseline"=TRUE))
      )
    }
    else {
      # For critical IVs, fit two models: one with predictors from the
      # final-layer baseline, and one with predictors from the midpoint-layer.
      for (j in 1:length(baseline_ivs)) {
        baseline_iv <- baseline_ivs[[j]]
        critical_fit <- fit_single(file_ID, df, dv, iv, baseline_iv, ...)
        fit_list <- append(
          fit_list, 
          list(list("model"=critical_fit, "iv"=iv, "baseline_type"=baseline_iv, "is_baseline"=FALSE))
        )
      }
    }
  }
  return(fit_list)
}

compare_fits <- function(fit_list) {
  # Assumes that baselines are first.
  if ((fit_list[[1]]$iv != "baseline_final") | (fit_list[[2]]$iv != "baseline_midpoint")) {
    stop()
  }
  baseline_final_fit <- fit_list[[1]]$model
  baseline_midpoint_fit <- fit_list[[2]]$model

  lrt_results <- list()
  bf_results <- list()

  # Iterate over fits corresponding to critical models.
  for (i in 3:length(fit_list)) {
    critical_fit <- fit_list[[i]]$model
    critical_iv <- fit_list[[i]]$iv
    critical_baseline_iv <- fit_list[[i]]$baseline_type

    if (is.na(critical_fit)){
      print(paste("SKIPPING:", critical_iv))
      next
    }
    else {
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FINAL-LAYER BASELINE
      if (critical_baseline_iv == "baseline_final") {
        # Bayes Factor
        bf <- bayesfactor_models(baseline_final_fit, critical_fit, denominator=baseline_final_fit)
        bf <- rownames_to_column(bf, var = "term") %>% as_tibble() %>% mutate(
          iv = ifelse(term=="baseline_final_fit", "baseline_final", critical_iv)
        )
        # Likelihood ratio test
        lrt <- anova(baseline_final_fit, critical_fit)
        lrt <- broom::tidy(lrt) %>% mutate(
          iv = ifelse(term=="baseline_final_fit", "baseline_final", critical_iv)
        )
      }
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MIDPOINT-LAYER BASELINE
      else if (critical_baseline_iv == "baseline_midpoint") {
        # Bayes Factor
        bf <- bayesfactor_models(baseline_midpoint_fit, critical_fit, denominator=baseline_midpoint_fit)
        bf <- rownames_to_column(bf, var = "term") %>% as_tibble() %>% mutate(
          iv = ifelse(term=="baseline_midpoint_fit", "baseline_midpoint", critical_iv)
        )
        # Likelihood ratio test
        lrt <- anova(baseline_midpoint_fit, critical_fit)
        lrt <- broom::tidy(lrt) %>% mutate(
          iv = ifelse(term=="baseline_midpoint_fit", "baseline_midpoint", critical_iv)
        )
      }

      # Annotate with current baseline type (final or midpoint).
      bf$baseline_type <- critical_baseline_iv
      lrt$baseline_type <- critical_baseline_iv

      # Update lists of results.
      bf_results[[i-2]] <- bf
      lrt_results[[i-2]] <- lrt
    }
  }
  bf_results <- bind_rows(bf_results)
  lrt_results <- bind_rows(lrt_results)
  return(list(bf=bf_results, lrt=lrt_results))
}

run_model_comparison <- function(df, dv, iv_list, task, data_subset, lm, file_suffix=NA, ...) {
  message(sprintf(
    "Getting model fits for TASK = %s; DV = %s; DATA SUBSET = %s; LANGUAGE MODEL = %s; FILE PREFIX = %s", 
    task, dv, data_subset, lm, file_suffix
  ))

  if (is.na(file_suffix)) {
    file_ID <- sprintf("%s_%s_%s_%s", task, dv, data_subset, lm)
  }
  else {
    file_ID <- sprintf("%s_%s_%s_%s_%s", task, dv, data_subset, lm, file_suffix)
  }

  # fit all models
  fits <- get_all_fits(file_ID, df, dv, iv_list, task, ...)
  if (length(fits) <= 1) {
    message("Skipping model comparison")
    return()
  }
  
  # compare model fits (critical fit vs. each baseline fit)
  comparison_results <- compare_fits(fits)
  
  # save Bayes Factor results
  bf_results <- comparison_results$bf 
  if ("Model" %in% colnames(bf_results)) {
    bf_results <- bf_results %>% rename("formula"="Model")
  }
  # remove duplicates, since there will be repeated baseline rows by construction
  bf_results <- bf_results[!duplicated(bf_results), ]
  bf_results$dv <- dv
  bf_results$data_subset <- data_subset
  bf_results$model <- lm
  bf_results$file_suffix <- file_suffix
  bf_out_path <- sprintf("%s/bf_%s.csv", OUT_DIR, file_ID)

  write.csv(bf_results, bf_out_path, row.names=F)
  message("Wrote BF results!")
  
  # save Likelihood Ratio Test results
  lrt_results <- comparison_results$lrt
  # remove duplicates, since there will be repeated baseline rows by construction
  lrt_results <- lrt_results[!duplicated(lrt_results), ]
  lrt_results$dv <- dv
  lrt_results$data_subset <- data_subset
  lrt_results$model <- lm
  lrt_results$file_suffix <- file_suffix
  lrt_out_path <- sprintf("%s/lrt_%s.csv", OUT_DIR, file_ID)
  write.csv(lrt_results, lrt_out_path, row.names=F)
  message("Wrote LRT results!")
}

# strip_glm <- function(cm) {
#   cm$y = c()
#   cm$model = c()
  
#   cm$residuals = c()
#   cm$fitted.values = c()
#   cm$effects = c()
#   cm$qr$qr = c()  
#   cm$linear.predictors = c()
#   cm$weights = c()
#   cm$prior.weights = c()
#   cm$data = c()
  
  
#   cm$family$variance = c()
#   cm$family$dev.resids = c()
#   cm$family$aic = c()
#   cm$family$validmu = c()
#   cm$family$simulate = c()
#   attr(cm$terms,".Environment") = c()
#   attr(cm$formula,".Environment") = c()
  
#   cm
# }
