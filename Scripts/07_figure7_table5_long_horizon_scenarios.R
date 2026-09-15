# ============================================================
# 07_figure7_table5_long_horizon_scenarios.R
#
# PURPOSE
#   Reproduce Figure 7 and Table 5 for:
#
#     Wearable sensing reveals the structure of cardiac
#     activation associated with everyday driving
#
#   Figure 7A:
#     Illustrative trait-anxiety scenario projections of annual
#     cumulative baseline-referenced cardiac activation.
#
#   Figure 7B:
#     Normalized out-of-fold ENet RMSE after temporal aggregation
#     over contiguous 1-, 2-, 5-, 10-, 15-, 30-, and 60-min windows.
#
#   Table 5:
#     Minute-level predicted activation and annual cumulative
#     NHR-hours for the lowest and highest observed trait-anxiety
#     values, with participant-bootstrap 5th-95th percentile ranges.
#     The high-minus-low contrast reports uncertainty from leave-one-participant-
#     out refits of the same penalized model. All 57 deletions are audited;
#     degenerate/boundary-zero refits are reported and excluded, and the CI is
#     computed from the remaining usable refits.
#
# MANUSCRIPT ALIGNMENT
#   - Primary resolution is fixed at 60 s.
#   - The DRIVING model frame is reconstructed to match Script 00,
#     including all saved short-term dynamic predictors.
#   - Nuisance-model hyperparameters use the component-wise median of the
#     outer-fold selections in best_params_DRIVING.csv.
#   - Figure 7A/Table 5 use the SAME fully penalized ENet/LASSO estimator
#     as the predictive decomposition; trait anxiety is penalized normally.
#   - The reviewer-requested high-minus-low uncertainty is assessed by
#     leave-one-participant-out refitting: the same penalized model is refit
#     57 times, each time omitting one participant. Degenerate/boundary-zero
#     refits are reported but excluded from the empirical LOPO interval calculation.
#   - Every observed DRIVING epoch is evaluated twice, with trait
#     anxiety fixed to the lowest and highest observed participant
#     values; all other predictors and participant baselines remain
#     unchanged.
#   - Existing uncertainty intervals for the low/high projected levels
#     are obtained by participant bootstrap conditional on the final fit.
#   - Uncertainty for the high-minus-low trait-anxiety contrast reflects
#     participant-level sensitivity of the fitted trait relationship via the
#     usable leave-one-participant-out refits.
#   - Figure 7B uses held-out outer-fold ENet predictions only.
#
# INPUTS
#   Data/NUBI_Data_60sec_Level_MASTER_CLEAN.csv
#
#   A compatible run folder under Results/nubi_ml containing:
#     predictor_list_DRIVING.csv
#     best_params_DRIVING.csv
#     best_params_DRIVING.csv
#     one compatible all-model out-of-fold prediction CSV
#
# OUTPUTS
#   Results/paper_figs/<timestamp>_60sec_figure7_table5_long_horizon/
#
# REPOSITORY SCOPE
#   The public repository begins from the final clean MASTER dataset
#   and curated nested-CV outputs. It does not reconstruct raw sensor
#   ingestion or the full upstream model-selection pipeline.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(lubridate)
  library(stringr)
  library(tidymodels)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(scales)
  library(patchwork)
})

options(warn = 1)
if (exists("tidymodels_prefer", mode = "function")) tidymodels_prefer()
set.seed(20260309)

# ============================================================
# USER SETTINGS
# ============================================================

LOCAL_TZ <- "America/Chicago"
RES_SECONDS <- 60L

# Use an exact folder name inside Results/nubi_ml, or leave NULL
# to select the most recently modified compatible 60-s run.
RUN_DIR_NAME <- NULL

TRAIT_VAR_CANDIDATES <- c(
  "trait_anxiety",
  "stai_trait",
  "traitanxiety",
  "stai_trait_total"
)

TRAIT_LABEL_LOW  <- "Lowest trait anxiety"
TRAIT_LABEL_HIGH <- "Highest trait anxiety"

DAYS_PER_YEAR <- 365
SCENARIOS_HOURS_PER_DAY <- c(
  "30 min/day commute"            = 0.5,
  "2 hr/day commute"              = 2.0,
  "8 hr/day professional driving" = 8.0
)

N_BOOT <- 20000L

# Original LOW/HIGH scenario intervals remain participant-bootstrap
# 5th-95th percentile ranges conditional on the full fitted model.
BOOT_SEED <- 20260911L

# Reviewer-requested HIGH-LOW contrast stability is assessed with a
# leave-one-participant-out refit analysis. Degenerate/boundary refits in which
# the penalized trait-anxiety coefficient collapses numerically to zero are
# reported explicitly but excluded from the empirical LOPO stability interval. The
# central 95% interval is the 2.5th-97.5th percentile range of the remaining
# usable leave-one-out contrast estimates.
JACKKNIFE_CONF_LEVEL <- 0.95
JACKKNIFE_ZERO_TOL <- 1e-12
EXPECTED_EXCLUDED_JACKKNIFE_REFITS <- 2L

HORIZONS_MIN <- c(1L, 2L, 5L, 10L, 15L, 30L, 60L)
MAX_GAP_MULTIPLIER <- 1.5

PAL_TRAIT <- c(
  "Lowest trait anxiety"  = "#4DD3D3",
  "Highest trait anxiety" = "#F4A3A3"
)

PAL_STRATUM <- c(
  "Driving" = "#E69F00",
  "Non-driving sedentary" = "gray45"
)

PNG_DPI <- 300

# Manuscript reference values used only for an end-of-run
# reproducibility check. The script does not force these values.
MANUSCRIPT_LOW_MINUTE_NHR  <- 10.91
MANUSCRIPT_HIGH_MINUTE_NHR <- 11.94
MANUSCRIPT_DIFF_MINUTE_NHR <- 1.03
MANUSCRIPT_CHECK_TOL_BPM   <- 0.02

# ============================================================
# PATHS
# ============================================================

this_script <- tryCatch(
  normalizePath(sys.frame(1)$ofile),
  error = function(e) NA_character_
)

script_dir <- if (!is.na(this_script) && file.exists(this_script)) {
  dirname(this_script)
} else {
  getwd()
}
script_dir <- normalizePath(script_dir, mustWork = TRUE)
setwd(script_dir)

project_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
data_path <- file.path(
  project_root,
  "Data",
  sprintf("NUBI_Data_%dsec_Level_MASTER_CLEAN.csv", RES_SECONDS)
)
ml_root <- file.path(project_root, "Results", "nubi_ml")
paper_fig_root <- file.path(project_root, "Results", "paper_figs")

if (!file.exists(data_path)) stop("Missing MASTER dataset: ", data_path)
if (!dir.exists(ml_root)) stop("Missing ML results directory: ", ml_root)
dir.create(paper_fig_root, recursive = TRUE, showWarnings = FALSE)

message("Script directory: ", script_dir)
message("Project root: ", project_root)
message("Resolution: ", RES_SECONDS, " sec")

# ============================================================
# HELPERS
# ============================================================

snakeify <- function(x) {
  x <- tolower(as.character(x))
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("_+", "_", x)
  gsub("^_|_$", "", x)
}

norm_chr <- function(x) trimws(tolower(as.character(x)))

parse_time_local <- function(x, tz = LOCAL_TZ) {
  if (inherits(x, "POSIXt")) return(with_tz(x, tzone = tz))
  x <- as.character(x)
  z <- suppressWarnings(ymd_hms(x, tz = tz, quiet = TRUE))
  if (!all(is.na(z))) return(z)
  suppressWarnings(parse_date_time(
    x,
    orders = c("ymd HMS", "ymd HM", "mdy HMS", "mdy HM",
               "dmy HMS", "dmy HM"),
    tz = tz
  ))
}

canonicalize_names <- function(dt) {
  stopifnot(is.data.table(dt))

  old <- names(dt)
  sn <- snakeify(old)

  aliases <- c(
    pid = "p_id",
    participant_id = "p_id",
    participantid = "p_id",
    timestamp = "time",
    datetime = "time",
    date_time = "time",
    datasource = "activity",
    data_source = "activity",
    source = "activity",
    hr_bl = "bl_hr",
    hrbl = "bl_hr",
    baseline = "bl_hr",
    daynum = "day_num",
    weather = "weather_info",
    traitanxiety = "trait_anxiety"
  )

  new <- sn
  hit <- sn %in% names(aliases)
  new[hit] <- unname(aliases[sn[hit]])

  if (!identical(old, new)) {
    new <- make.unique(new, sep = "_")
    setnames(dt, old, new)
  }

  dt
}

normalize_stratum <- function(x) {
  z <- toupper(trimws(as.character(x)))
  z <- str_replace_all(z, "[[:space:]/-]+", "_")

  case_when(
    z %in% c("DRIVING", "DRIVE") ~ "DRIVING",
    z %in% c(
      "NONDRIVING_SEDENTARY",
      "NON_DRIVING_SEDENTARY",
      "NONDRIVING",
      "NON_DRIVING",
      "SEDENTARY_NONDRIVING",
      "SEDENTARY_NON_DRIVING",
      "NONDRIVINGSEDENTARY"
    ) ~ "NONDRIVING_SEDENTARY",
    TRUE ~ NA_character_
  )
}

normalize_model <- function(x) {
  z <- tolower(trimws(as.character(x)))
  z <- str_replace_all(z, "[[:space:]/-]+", "_")

  case_when(
    z %in% c("enet", "elastic_net", "elasticnet", "full_enet") ~ "enet",
    z %in% c(
      "baseline_offset", "baseline_plus_offset",
      "baseline_context_offset", "offset"
    ) ~ "baseline_offset",
    z %in% c(
      "baseline0", "baseline_0", "baseline", "baseline_only",
      "baselineonly"
    ) ~ "baseline0",
    TRUE ~ NA_character_
  )
}

first_existing_col <- function(df, candidates, required = TRUE, what = "column") {
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) == 0L) {
    if (required) {
      stop("Could not find ", what, ". Tried: ",
           paste(candidates, collapse = ", "))
    }
    return(NA_character_)
  }
  hit[1]
}

safe_save_pdf <- function(plot_obj, path, w, h) {
  ok <- tryCatch({
    ggsave(
      filename = path,
      plot = plot_obj,
      width = w,
      height = h,
      device = grDevices::cairo_pdf
    )
    TRUE
  }, error = function(e) FALSE)

  if (!ok) {
    ggsave(
      filename = path,
      plot = plot_obj,
      width = w,
      height = h,
      device = "pdf",
      useDingbats = FALSE
    )
  }
}

find_exact_file <- function(root, basename_required) {
  hits <- list.files(
    root,
    pattern = paste0("^", basename_required, "$"),
    full.names = TRUE,
    recursive = TRUE
  )
  if (length(hits) == 0L) return(NA_character_)

  rel <- substring(hits, nchar(root) + 2L)
  depth <- lengths(strsplit(rel, .Platform$file.sep, fixed = TRUE))
  info <- file.info(hits)
  hits[order(depth, -as.numeric(info$mtime))[1]]
}

prediction_basenames <- c(
  "predictions_all_models_both_strata.csv",
  "predictions_all_models.csv",
  "oof_predictions_all_models_both_strata.csv",
  "oof_predictions_all_models.csv"
)

resolve_run_inputs <- function(run_dir) {
  predictor <- find_exact_file(run_dir, "predictor_list_DRIVING.csv")

  # Figure 7 and Table 5 must use the manuscript-selected
  # hyperparameters. The importance-refit file belongs to the
  # Figure 6 interpretation workflow and is not interchangeable.
  param_file <- find_exact_file(run_dir, "best_params_DRIVING.csv")

  # Retain the Figure 6 full-refit parameter file only as a
  # diagnostic comparison, never as a fallback for Figure 7.
  importance_param_file <- find_exact_file(
    run_dir,
    "best_params_full_refit_for_importance_DRIVING.csv"
  )

  pred_hits <- vapply(
    prediction_basenames,
    function(x) find_exact_file(run_dir, x),
    character(1)
  )
  pred_file <- unname(pred_hits[!is.na(pred_hits)][1])
  if (length(pred_file) == 0L) pred_file <- NA_character_

  c(
    predictor = predictor,
    params = param_file,
    importance_params = importance_param_file,
    predictions = pred_file
  )
}

auto_pick_run_dir <- function(root, res_seconds) {
  dirs <- list.dirs(root, recursive = FALSE, full.names = TRUE)
  dirs <- dirs[file.info(dirs)$isdir %in% TRUE]
  res_pat <- paste0("(^|[^0-9])", res_seconds, "sec([^0-9]|$)")
  dirs <- dirs[grepl(res_pat, basename(dirs), ignore.case = TRUE, perl = TRUE)]

  if (length(dirs) == 0L) {
    stop("No ", res_seconds, "-s run folders found under: ", root)
  }

  ok <- vapply(
    dirs,
    function(d) {
      p <- resolve_run_inputs(d)
      all(!is.na(p[c("predictor", "params", "predictions")]))
    },
    logical(1)
  )

  cand <- dirs[ok]
  if (length(cand) == 0L) {
    inventory <- vapply(
      dirs,
      function(d) {
        p <- resolve_run_inputs(d)
        paste0(
          basename(d), ": predictor=", !is.na(p["predictor"]),
          ", manuscript_params=", !is.na(p["params"]),
          ", importance_params=", !is.na(p["importance_params"]),
          ", predictions=", !is.na(p["predictions"])
        )
      },
      character(1)
    )
    stop(
      "Could not find a compatible 60-s ML run.\n",
      paste(inventory, collapse = "\n")
    )
  }

  cand[which.max(file.info(cand)$mtime)]
}

make_day_key <- function(dt) {
  if ("day_num" %in% names(dt)) {
    x <- suppressWarnings(as.integer(str_extract(as.character(dt$day_num), "\\d+")))
    if (!all(is.na(x))) return(x)
  }
  if ("days" %in% names(dt)) {
    x <- suppressWarnings(as.integer(str_extract(as.character(dt$days), "\\d+")))
    if (!all(is.na(x))) return(x)
  }
  as.integer(as.Date(dt$dt_time, tz = LOCAL_TZ))
}

add_bl_hr_person <- function(dt) {
  by_day <- dt[
    is.finite(bl_hr) & !is.na(day_key),
    .(bl_hr_day = median(bl_hr, na.rm = TRUE)),
    by = .(p_id, day_key)
  ]

  if (nrow(by_day) == 0L) {
    stop("Could not calculate participant-day baselines.")
  }

  by_person <- by_day[
    ,
    .(bl_hr_person = mean(bl_hr_day, na.rm = TRUE)),
    by = p_id
  ]

  merge(dt, by_person, by = "p_id", all.x = TRUE)
}

dyn_base_vars <- c(
  "speed", "ff", "ff_speed", "atp", "rtp", "jf",
  "energy_acc", "energy_rot"
)

add_dynamics <- function(dt,
                         id_col = "p_id",
                         time_col = "dt_time",
                         vars = dyn_base_vars,
                         res_seconds = RES_SECONDS,
                         windows_min = c(1, 3, 5)) {
  stopifnot(is.data.table(dt))
  vars <- intersect(vars, names(dt))
  if (length(vars) == 0L) return(dt)

  setorderv(dt, c(id_col, time_col))

  for (v in vars) {
    dt[, (v) := suppressWarnings(as.numeric(get(v)))]

    dt[, paste0(v, "_lag1") := shift(get(v), 1L), by = id_col]
    dt[, paste0(v, "_diff1") := get(v) - get(paste0(v, "_lag1")),
       by = id_col]

    for (wm in windows_min) {
      k <- max(2L, as.integer(round((wm * 60) / res_seconds)))
      tag <- paste0("_", wm, "m")

      rm_name <- paste0(v, "_rm", tag)
      rs_name <- paste0(v, "_rs", tag)
      sl_name <- paste0(v, "_slope", tag)

      dt[, (rm_name) := frollmean(
        get(v), n = k, align = "right", fill = NA_real_
      ), by = id_col]

      dt[, (rs_name) := {
        m1 <- frollmean(get(v), n = k, align = "right", fill = NA_real_)
        m2 <- frollmean(get(v)^2, n = k, align = "right", fill = NA_real_)
        sqrt(pmax(m2 - m1^2, 0))
      }, by = id_col]

      dt[, (sl_name) := (
        get(v) - shift(get(v), k - 1L)
      ) / ((k - 1L) * res_seconds), by = id_col]
    }
  }

  dt
}

# ============================================================
# SELECT RUN AND CREATE OUTPUT DIRECTORY
# ============================================================

run_dir <- if (!is.null(RUN_DIR_NAME)) {
  file.path(ml_root, RUN_DIR_NAME)
} else {
  auto_pick_run_dir(ml_root, RES_SECONDS)
}

if (!dir.exists(run_dir)) stop("Selected run folder does not exist: ", run_dir)
run_inputs <- resolve_run_inputs(run_dir)

message("Selected ML run: ", basename(run_dir))
message("Predictor list: ", run_inputs["predictor"])
message("Manuscript parameters: ", run_inputs["params"])
message("Importance-refit parameters (diagnostic only): ", run_inputs["importance_params"])
message("OOF predictions: ", run_inputs["predictions"])

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(
  paper_fig_root,
  paste0(stamp, "_", RES_SECONDS, "sec_figure7_table5_long_horizon")
)
fig_dir <- file.path(out_dir, "Figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(out_dir, "run_log.txt")
log_msg <- function(...) {
  msg <- paste0(
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    " | ",
    paste(..., collapse = "")
  )
  message(msg)
  cat(msg, "\n", file = log_file, append = TRUE)
}

log_msg("Script: 07_figure7_table5_long_horizon_scenarios.R")
log_msg("Project root: ", project_root)
log_msg("Resolution: ", RES_SECONDS, " sec")
log_msg("MASTER data: ", data_path)
log_msg("ML run: ", run_dir)
log_msg("Predictor list: ", run_inputs["predictor"])
log_msg("Manuscript parameter file: ", run_inputs["params"])
log_msg("Importance-refit parameter file (diagnostic only): ", run_inputs["importance_params"])
log_msg("OOF prediction file: ", run_inputs["predictions"])
log_msg("N_BOOT (LOW/HIGH conditional participant bootstrap): ", N_BOOT)
log_msg("Output directory: ", out_dir)

# ============================================================
# READ SAVED MODEL SPECIFICATION
# ============================================================

pred_list <- read_csv(run_inputs["predictor"], show_col_types = FALSE)
pred_col <- first_existing_col(
  pred_list,
  c("predictor", "term", "variable", "feature"),
  TRUE,
  "predictor-list column"
)
preds_drive <- unique(as.character(pred_list[[pred_col]]))
preds_drive <- preds_drive[!is.na(preds_drive) & nzchar(preds_drive)]

param_df <- read_csv(run_inputs["params"], show_col_types = FALSE)
if (!all(c("penalty", "mixture") %in% names(param_df))) {
  stop("Parameter file must contain penalty and mixture columns: ",
       run_inputs["params"])
}

# Use the same component-wise median of outer-fold-selected hyperparameters
# that Script 00 uses for its full-data descriptive refit. This avoids making
# the long-horizon analysis depend on an arbitrary single outer fold.
penalty_vals <- suppressWarnings(as.numeric(param_df$penalty))
mixture_vals <- suppressWarnings(as.numeric(param_df$mixture))

penalty_val <- median(penalty_vals[is.finite(penalty_vals)], na.rm = TRUE)
mixture_val <- median(mixture_vals[is.finite(mixture_vals)], na.rm = TRUE)

if (!is.finite(penalty_val) || !is.finite(mixture_val)) {
  stop("Invalid penalty/mixture values in: ", run_inputs["params"])
}

log_msg(
  "Figure 7A nuisance-model hyperparameters (component-wise median across outer folds): penalty=",
  penalty_val, " | mixture=", mixture_val
)

write_csv(
  tibble(
    resolution_seconds = RES_SECONDS,
    master_file = basename(data_path),
    run_folder = basename(run_dir),
    predictor_file = basename(run_inputs["predictor"]),
    parameter_file = basename(run_inputs["params"]),
    parameter_selection = "component-wise median across outer folds",
    oof_prediction_file = basename(run_inputs["predictions"]),
    penalty = penalty_val,
    mixture = mixture_val,
    bootstrap_replicates = N_BOOT,
    lopo_empirical_interval_level = JACKKNIFE_CONF_LEVEL,
    focal_trait_penalty_factor = 1,
    nuisance_penalty_factor = 1,
    contrast_estimator = "leave-one-participant-out refits; boundary-zero/failed refits excluded from CI",
    saved_driving_predictor_count = length(preds_drive)
  ),
  file.path(out_dir, "figure7_model_specification.csv")
)

# ============================================================
# READ MASTER DATA AND RECREATE MODEL FRAME
# ============================================================

# Recreate the DRIVING model frame exactly as Script 00 does:
#   - use activity3 for the behavioral stratum;
#   - retain the common participant set across driving and non-driving sedentary;
#   - compute the same participant-level baseline;
#   - construct the same short-term driving dynamics BEFORE dropping rows with
#     missing raw HR; and
#   - require every saved DRIVING predictor to be present.
# This removes the previous mismatch in which only predictors already present
# in MASTER were used and the dynamically constructed predictors were omitted.
dt <- fread(data_path, showProgress = TRUE)
dt <- canonicalize_names(dt)

required_master <- c("p_id", "time", "raw_hr", "activity3", "bl_hr")
missing_master <- setdiff(required_master, names(dt))
if (length(missing_master) > 0L) {
  stop("MASTER data missing: ", paste(missing_master, collapse = ", "))
}

if (!("dt_time" %in% names(dt)) || !inherits(dt$dt_time, "POSIXt")) {
  dt[, dt_time := parse_time_local(time, tz = LOCAL_TZ)]
}
dt <- dt[!is.na(dt_time)]

dt[, activity3_norm := norm_chr(activity3)]
dt[, raw_hr_num := suppressWarnings(as.numeric(raw_hr))]
dt[, bl_hr_num := suppressWarnings(as.numeric(bl_hr))]
dt[, day_key := make_day_key(dt)]

baseline_by_subj_day <- dt[
  is.finite(bl_hr_num) & !is.na(day_key),
  .(bl_hr_day = median(bl_hr_num, na.rm = TRUE)),
  by = .(p_id, day_key)
]
if (nrow(baseline_by_subj_day) == 0L) {
  stop("Could not calculate participant-day baselines.")
}

baseline_by_subj <- baseline_by_subj_day[
  , .(bl_hr_person = mean(bl_hr_day, na.rm = TRUE)),
  by = p_id
]

dt <- merge(dt, baseline_by_subj, by = "p_id", all.x = TRUE)
dt <- dt[is.finite(bl_hr_person)]

dt_drive <- copy(dt[activity3_norm == "driving"])
dt_nond <- copy(dt[activity3_norm == "non_driving_sedentary"])
if (nrow(dt_drive) == 0L) stop("No DRIVING rows found.")
if (nrow(dt_nond) == 0L) stop("No NONDRIVING_SEDENTARY rows found.")

common_subj <- intersect(unique(dt_drive$p_id), unique(dt_nond$p_id))
dt_drive <- dt_drive[p_id %in% common_subj]
log_msg("Figure 7A common participants: ", length(common_subj))

# Match Script 00 driving dynamics exactly.
dt_drive <- add_dynamics(
  dt_drive,
  id_col = "p_id",
  time_col = "dt_time",
  vars = dyn_base_vars,
  res_seconds = RES_SECONDS,
  windows_min = c(1, 3, 5)
)

missing_saved_predictors <- setdiff(preds_drive, names(dt_drive))
if (length(missing_saved_predictors) > 0L) {
  stop(
    "Could not reconstruct all saved DRIVING predictors. Missing: ",
    paste(missing_saved_predictors, collapse = ", ")
  )
}

# Preserve day_key only as an ID/summary variable; it is not a predictor in
# Script 00. raw_hr is filtered only after the dynamics have been generated.
model_cols <- unique(c("p_id", "day_key", "raw_hr_num", preds_drive))
model_df <- as.data.frame(dt_drive[, ..model_cols])
model_df <- model_df[is.finite(model_df$raw_hr_num), , drop = FALSE]
model_df$p_id <- factor(model_df$p_id)
model_df$raw_hr <- model_df$raw_hr_num
model_df$raw_hr_num <- NULL

trait_var <- TRAIT_VAR_CANDIDATES[TRAIT_VAR_CANDIDATES %in% names(model_df)][1]
if (length(trait_var) == 0L || is.na(trait_var)) {
  stop(
    "Could not find a supported trait-anxiety variable. Tried: ",
    paste(TRAIT_VAR_CANDIDATES, collapse = ", ")
  )
}

log_msg(
  "Corrected full-refit DRIVING rows: ", nrow(model_df),
  " | participants: ", n_distinct(model_df$p_id),
  " | saved predictors used: ", length(preds_drive),
  " / ", length(preds_drive)
)
log_msg("Trait variable: ", trait_var)

trait_vals <- suppressWarnings(as.numeric(model_df[[trait_var]]))
if (all(!is.finite(trait_vals))) {
  stop("Trait variable has no finite numeric values: ", trait_var)
}
model_df[[trait_var]] <- trait_vals

trait_audit <- model_df %>%
  filter(is.finite(.data[[trait_var]])) %>%
  group_by(p_id) %>%
  summarise(
    trait_min = min(.data[[trait_var]], na.rm = TRUE),
    trait_max = max(.data[[trait_var]], na.rm = TRUE),
    trait_range = trait_max - trait_min,
    n_rows = n(),
    .groups = "drop"
  )

write_csv(
  trait_audit,
  file.path(out_dir, "figure7_trait_value_consistency_by_subject.csv")
)

if (any(!is.finite(trait_audit$trait_range)) ||
    any(trait_audit$trait_range > 1e-8, na.rm = TRUE)) {
  stop("Trait anxiety is not constant within participant.")
}

trait_subject <- model_df %>%
  filter(is.finite(.data[[trait_var]])) %>%
  arrange(p_id) %>%
  group_by(p_id) %>%
  summarise(
    trait_value = dplyr::first(.data[[trait_var]]),
    bl_hr_person = dplyr::first(bl_hr_person),
    n_rows_drive = n(),
    n_days_drive = n_distinct(day_key),
    .groups = "drop"
  ) %>%
  arrange(trait_value)

trait_low <- trait_subject$trait_value[1]
trait_high <- trait_subject$trait_value[nrow(trait_subject)]

log_msg("Observed trait-anxiety extremes: ", trait_low, " and ", trait_high)

# ============================================================
# REFIT FINAL DRIVING MODEL FOR TRAIT CONTRAST
# ============================================================

# IMPORTANT:
# The final ENet is fit once using the same model specification and
# hyperparameters as the long-horizon scenario analysis. Trait anxiety is
# penalized normally, exactly as in the predictive model. The participant
# participant bootstrap below holds this fitted model fixed and reproduces the
# original LOW/HIGH scenario intervals. The reviewer-requested HIGH-LOW
# contrast uncertainty is estimated separately by leave-one-participant-out
# jackknife refits of this same model specification.
make_recipe <- function(training_df) {
  recipe(raw_hr ~ ., data = training_df) %>%
    update_role(p_id, day_key, new_role = "id") %>%
    step_string2factor(all_nominal_predictors()) %>%
    step_unknown(all_nominal_predictors(), new_level = "Unknown") %>%
    step_novel(all_nominal_predictors()) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())
}

fit_penalized_model <- function(training_df) {
  rec_b <- make_recipe(training_df)
  prep_b <- prep(rec_b, training = training_df, verbose = FALSE)

  x_train <- juice(prep_b, all_predictors(), composition = "matrix")
  y_train <- as.numeric(juice(prep_b, all_outcomes(), composition = "matrix")[, 1])

  if (!(trait_var %in% colnames(x_train))) {
    stop("Trait predictor not present after recipe preprocessing: ", trait_var)
  }

  # Same penalty for every predictor, matching the predictive ENet/LASSO.
  penalty_factor <- rep(1, ncol(x_train))
  names(penalty_factor) <- colnames(x_train)

  fit_b <- glmnet::glmnet(
    x = x_train,
    y = y_train,
    alpha = mixture_val,
    lambda = penalty_val,
    penalty.factor = penalty_factor,
    family = "gaussian"
  )

  list(
    prep = prep_b,
    fit = fit_b,
    penalty_factor = penalty_factor
  )
}

predict_raw_penalized <- function(fitted_obj, new_data) {
  x_new <- bake(
    fitted_obj$prep,
    new_data = new_data,
    all_predictors(),
    composition = "matrix"
  )
  as.numeric(stats::predict(
    fitted_obj$fit,
    newx = x_new,
    s = penalty_val,
    type = "response"
  ))
}

predict_nhr <- function(fitted_obj, new_data) {
  predict_raw_penalized(fitted_obj, new_data) - new_data$bl_hr_person
}

extract_trait_coef <- function(fitted_obj) {
  cc <- as.matrix(stats::coef(fitted_obj$fit, s = penalty_val))
  if (!(trait_var %in% rownames(cc))) return(NA_real_)
  as.numeric(cc[trait_var, 1])
}

log_msg(
  "Fitting full DRIVING predictive model for Figure 7A/Table 5: ",
  "all 119 saved predictors restored; trait anxiety penalized normally."
)
fit_full <- fit_penalized_model(model_df)

log_msg(
  "Full-fit standardized trait coefficient: ",
  sprintf("%.6f", extract_trait_coef(fit_full))
)

make_counterfactual <- function(base_df, trait_value, condition_label) {
  nd <- base_df
  nd[[trait_var]] <- trait_value

  tibble(
    condition = condition_label,
    p_id = as.character(nd$p_id),
    row_id = seq_len(nrow(nd)),
    trait_value = trait_value,
    bl_hr_person = nd$bl_hr_person,
    nhr_hat = predict_nhr(fit_full, nd)
  )
}

cf_rows <- bind_rows(
  make_counterfactual(model_df, trait_low, TRAIT_LABEL_LOW),
  make_counterfactual(model_df, trait_high, TRAIT_LABEL_HIGH)
)

write_csv(
  cf_rows,
  file.path(out_dir, "figure7_trait_counterfactual_predictions_by_row.csv")
)

cf_subject <- cf_rows %>%
  group_by(condition, p_id) %>%
  summarise(
    n_rows = n(),
    mean_nhr_hat = mean(nhr_hat, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  cf_subject,
  file.path(out_dir, "figure7_trait_counterfactual_subject_summary.csv")
)

# --------------------------------------------------------------------
# Original conditional participant bootstrap for LOW and HIGH levels
# --------------------------------------------------------------------
# Preserve the original manuscript procedure for the individual projected
# levels: hold the full fitted ENet fixed, resample participants with
# replacement, and summarize LOW and HIGH separately. The same sampled IDs
# are used for both scenarios simply to preserve their natural pairing; these
# draws are NOT used to estimate uncertainty in the HIGH-LOW contrast.
cf_subject_wide <- cf_subject %>%
  select(condition, p_id, mean_nhr_hat) %>%
  pivot_wider(
    names_from = condition,
    values_from = mean_nhr_hat
  )

if (anyNA(cf_subject_wide[[TRAIT_LABEL_LOW]]) ||
    anyNA(cf_subject_wide[[TRAIT_LABEL_HIGH]])) {
  stop("Missing paired LOW/HIGH participant summaries.")
}

bootstrap_paired_levels <- function(subject_summary_wide, B) {
  ids <- unique(subject_summary_wide$p_id)
  if (length(ids) < 3L) stop("Too few participants for bootstrap.")

  set.seed(BOOT_SEED)

  bind_rows(lapply(seq_len(B), function(b) {
    sampled_ids <- sample(ids, size = length(ids), replace = TRUE)

    dd <- tibble(p_id = sampled_ids) %>%
      left_join(subject_summary_wide, by = "p_id")

    tibble(
      sim = b,
      low = mean(dd[[TRAIT_LABEL_LOW]], na.rm = TRUE),
      high = mean(dd[[TRAIT_LABEL_HIGH]], na.rm = TRUE)
    )
  }))
}

boot_levels <- bootstrap_paired_levels(
  cf_subject_wide,
  N_BOOT
)

write_csv(
  boot_levels,
  file.path(out_dir, "figure7_trait_counterfactual_level_bootstrap_draws.csv")
)

boot_long <- bind_rows(
  boot_levels %>%
    transmute(sim, condition = TRAIT_LABEL_LOW, mean_nhr_hat = low),
  boot_levels %>%
    transmute(sim, condition = TRAIT_LABEL_HIGH, mean_nhr_hat = high)
)

write_csv(
  boot_long,
  file.path(out_dir, "figure7_trait_counterfactual_bootstrap_draws.csv")
)

# Point estimates from the reconstructed single full-data penalized fit.
point_low <- mean(
  cf_rows$nhr_hat[cf_rows$condition == TRAIT_LABEL_LOW],
  na.rm = TRUE
)
point_high <- mean(
  cf_rows$nhr_hat[cf_rows$condition == TRAIT_LABEL_HIGH],
  na.rm = TRUE
)
point_difference <- point_high - point_low

# --------------------------------------------------------------------
# Leave-one-participant-out refits for HIGH-LOW contrast uncertainty
# --------------------------------------------------------------------
# Each refit contains 56 DISTINCT participants (for n=57), avoiding the loss
# of participant-level support caused by ordinary bootstrap resampling with
# replacement. Trait anxiety remains penalized exactly as in the original
# predictive model.
#
# IMPORTANT FOR THIS SMALL/SPARSE PARTICIPANT-LEVEL PREDICTOR:
# If a LOPO refit collapses the trait-anxiety coefficient to the numerical
# boundary (approximately zero), that refit is retained in the audit output but
# is NOT allowed to contribute a literal zero to the uncertainty estimate.
# The uncertainty calculation uses only the non-degenerate, finite refits.
jackknife_ids <- sort(unique(as.character(model_df$p_id)))
n_jack_attempted <- length(jackknife_ids)
if (n_jack_attempted < 3L) stop("Too few participants for leave-one-out refitting.")

make_counterfactual_with_fit <- function(fitted_obj, base_df, trait_value) {
  nd <- base_df
  nd[[trait_var]] <- trait_value
  predict_nhr(fitted_obj, nd)
}

jackknife_rows <- bind_rows(lapply(seq_along(jackknife_ids), function(j) {
  omitted_id <- jackknife_ids[j]

  train_j <- model_df[as.character(model_df$p_id) != omitted_id, , drop = FALSE]
  train_j$p_id <- droplevels(train_j$p_id)

  distinct_j <- n_distinct(train_j$p_id)
  if (distinct_j != n_jack_attempted - 1L) {
    stop(
      "LOPO refit for omitted participant ", omitted_id,
      " has ", distinct_j, " distinct participants; expected ",
      n_jack_attempted - 1L, "."
    )
  }

  # Catch genuine fitting/prediction failures so that they are reported rather
  # than terminating the entire sensitivity analysis.
  refit_result <- tryCatch({
    fit_j <- fit_penalized_model(train_j)
    trait_coef_j <- extract_trait_coef(fit_j)

    low_j <- mean(
      make_counterfactual_with_fit(fit_j, train_j, trait_low),
      na.rm = TRUE
    )
    high_j <- mean(
      make_counterfactual_with_fit(fit_j, train_j, trait_high),
      na.rm = TRUE
    )

    contrast_j <- high_j - low_j
    boundary_zero <- is.finite(trait_coef_j) && abs(trait_coef_j) <= JACKKNIFE_ZERO_TOL
    finite_result <- all(is.finite(c(trait_coef_j, low_j, high_j, contrast_j)))

    # For inference, a numerical boundary-zero trait coefficient is treated as
    # a degenerate/non-usable refit, rather than as evidence for a true zero
    # biological contrast.
    usable_for_ci <- finite_result && !boundary_zero

    status <- if (!finite_result) {
      "non_finite_refit"
    } else if (boundary_zero) {
      "boundary_zero_excluded"
    } else {
      "usable"
    }

    list(
      trait_coef = trait_coef_j,
      low = low_j,
      high = high_j,
      contrast = contrast_j,
      boundary_zero = boundary_zero,
      usable_for_ci = usable_for_ci,
      status = status,
      error_message = NA_character_
    )
  }, error = function(e) {
    list(
      trait_coef = NA_real_,
      low = NA_real_,
      high = NA_real_,
      contrast = NA_real_,
      boundary_zero = FALSE,
      usable_for_ci = FALSE,
      status = "fit_or_prediction_error",
      error_message = conditionMessage(e)
    )
  })

  if (j %% 10L == 0L || j == 1L || j == n_jack_attempted || !refit_result$usable_for_ci) {
    log_msg(
      "LOPO refit ", j, "/", n_jack_attempted,
      " | omitted participant=", omitted_id,
      " | status=", refit_result$status,
      " | trait coef=", ifelse(is.finite(refit_result$trait_coef),
                               sprintf("%.6f", refit_result$trait_coef), "NA"),
      " | HIGH-LOW=", ifelse(is.finite(refit_result$contrast),
                              sprintf("%.3f", refit_result$contrast), "NA"), " bpm"
    )
  }

  tibble(
    jackknife_index = j,
    omitted_p_id = omitted_id,
    n_distinct_participants = distinct_j,
    n_training_rows = nrow(train_j),
    trait_coefficient_standardized = refit_result$trait_coef,
    trait_coefficient_zero = refit_result$boundary_zero,
    low_mean_bpm = refit_result$low,
    high_mean_bpm = refit_result$high,
    minute_difference_bpm = refit_result$contrast,
    refit_status = refit_result$status,
    included_in_ci = refit_result$usable_for_ci,
    error_message = refit_result$error_message
  )
}))

# Preserve ALL attempted LOPO refits in the audit file, including the two
# degenerate/boundary-zero cases.
write_csv(
  jackknife_rows,
  file.path(out_dir, "figure7_trait_contrast_jackknife_leave_one_out.csv")
)

jackknife_valid <- jackknife_rows %>%
  filter(included_in_ci, is.finite(minute_difference_bpm))

jackknife_excluded <- jackknife_rows %>%
  filter(!included_in_ci)

n_jack_valid <- nrow(jackknife_valid)
n_jack_excluded <- nrow(jackknife_excluded)

if (n_jack_valid < 3L) {
  stop("Fewer than three usable LOPO refits remain after excluding failed/degenerate cases.")
}

# This check is deliberately a warning rather than a stop: the CSV still shows
# exactly what happened if the data/model behavior changes in a future run.
if (n_jack_excluded != EXPECTED_EXCLUDED_JACKKNIFE_REFITS) {
  warning(
    "Expected ", EXPECTED_EXCLUDED_JACKKNIFE_REFITS,
    " excluded LOPO refits, but observed ", n_jack_excluded, "."
  )
}

write_csv(
  jackknife_excluded,
  file.path(out_dir, "figure7_trait_contrast_jackknife_excluded_refits.csv")
)

# --------------------------------------------------------------------
# EMPIRICAL CENTRAL 95% INTERVAL FROM THE 55 USABLE LOPO REFITS
# --------------------------------------------------------------------
# The two boundary-zero refits are retained in the audit output but excluded
# from the stability interval. Because the resulting deletion set is incomplete,
# we do NOT apply the classical jackknife variance formula. Instead, we summarize
# the empirical distribution of the usable leave-one-participant-out contrasts
# and report its central 95% interval (2.5th and 97.5th percentiles). This is a
# LOPO stability/sensitivity interval, not a classical sampling-theory CI.
jack_mean <- mean(jackknife_valid$minute_difference_bpm)
jack_median <- median(jackknife_valid$minute_difference_bpm)
jack_sd <- sd(jackknife_valid$minute_difference_bpm)
jack_min <- min(jackknife_valid$minute_difference_bpm)
jack_max <- max(jackknife_valid$minute_difference_bpm)
jack_positive_n <- sum(jackknife_valid$minute_difference_bpm > 0)

alpha_lopo <- 1 - JACKKNIFE_CONF_LEVEL
contrast_ci <- as.numeric(quantile(
  jackknife_valid$minute_difference_bpm,
  probs = c(alpha_lopo / 2, 1 - alpha_lopo / 2),
  na.rm = TRUE,
  names = FALSE,
  type = 7
))

excluded_ids <- paste(jackknife_excluded$omitted_p_id, collapse = ";")
excluded_statuses <- paste(
  paste0(jackknife_excluded$omitted_p_id, ":", jackknife_excluded$refit_status),
  collapse = ";"
)

contrast_summary <- tibble(
  quantity = "High-minus-low trait-anxiety predicted activation",
  full_data_point_estimate_bpm = point_difference,
  valid_lopo_mean_bpm = jack_mean,
  valid_lopo_median_bpm = jack_median,
  valid_lopo_sd_bpm = jack_sd,
  valid_lopo_min_bpm = jack_min,
  valid_lopo_max_bpm = jack_max,
  valid_lopo_positive_refits = jack_positive_n,
  valid_lopo_positive_fraction = jack_positive_n / n_jack_valid,
  ci_level = JACKKNIFE_CONF_LEVEL,
  ci_method = paste0(
    "empirical central ", sprintf("%.0f", 100 * JACKKNIFE_CONF_LEVEL),
    "% LOPO interval: percentile 2.5-97.5 of ", n_jack_valid,
    " usable refits after excluding boundary-zero/failed refits"
  ),
  ci_lower_bpm = contrast_ci[1],
  ci_upper_bpm = contrast_ci[2],
  lopo_refits_attempted = n_jack_attempted,
  lopo_refits_included_in_ci = n_jack_valid,
  lopo_refits_excluded_from_ci = n_jack_excluded,
  excluded_omitted_participants = excluded_ids,
  excluded_refit_statuses = excluded_statuses,
  participants_per_refit = n_jack_attempted - 1L,
  model_refit_each_replicate = TRUE,
  trait_penalized_normally = TRUE,
  zero_trait_coefficients_total = sum(jackknife_rows$trait_coefficient_zero, na.rm = TRUE),
  zero_trait_coefficients_excluded = sum(jackknife_excluded$trait_coefficient_zero, na.rm = TRUE)
)

write_csv(
  contrast_summary,
  file.path(out_dir, "figure7_trait_contrast_jackknife_summary.csv")
)

log_msg(
  "HIGH-LOW minute-level contrast: ", sprintf("%.3f", point_difference),
  " bpm | usable LOPO refits=", n_jack_valid, "/", n_jack_attempted,
  " | excluded=", n_jack_excluded,
  if (n_jack_excluded > 0L) paste0(" [", excluded_statuses, "]") else "",
  " | valid LOPO median=", sprintf("%.3f", jack_median),
  " | valid LOPO range=[", sprintf("%.3f", jack_min), ", ", sprintf("%.3f", jack_max), "]",
  " | positive=", jack_positive_n, "/", n_jack_valid,
  " | empirical central ", sprintf("%.0f", 100 * JACKKNIFE_CONF_LEVEL), "% interval [",
  sprintf("%.3f", contrast_ci[1]), ", ",
  sprintf("%.3f", contrast_ci[2]), "]"
)

scenarios <- tibble(
  scenario = names(SCENARIOS_HOURS_PER_DAY),
  hours_per_day = as.numeric(SCENARIOS_HOURS_PER_DAY),
  annual_hours = hours_per_day * DAYS_PER_YEAR
)

annual_draws <- crossing(
  boot_long,
  scenarios
) %>%
  mutate(annual_nhr_hours = mean_nhr_hat * annual_hours)

scenario_summary <- annual_draws %>%
  group_by(scenario, hours_per_day, annual_hours, condition) %>%
  summarise(
    minute_mean = mean(mean_nhr_hat),
    minute_q05 = quantile(mean_nhr_hat, 0.05),
    minute_q95 = quantile(mean_nhr_hat, 0.95),
    annual_mean = mean(annual_nhr_hours),
    annual_q05 = quantile(annual_nhr_hours, 0.05),
    annual_q95 = quantile(annual_nhr_hours, 0.95),
    .groups = "drop"
  ) %>%
  mutate(
    scenario = factor(
      scenario,
      levels = names(SCENARIOS_HOURS_PER_DAY)
    ),
    condition = factor(
      condition,
      levels = c(TRAIT_LABEL_LOW, TRAIT_LABEL_HIGH)
    )
  ) %>%
  arrange(scenario, condition)

write_csv(
  scenario_summary,
  file.path(out_dir, "figure7_trait_counterfactual_load_summary.csv")
)

# High-low difference point estimate and empirical LOPO stability interval.
# Annual differences are deterministic scalings of the minute-level contrast
# under each illustrative exposure schedule.
difference_summary <- scenarios %>%
  mutate(
    scenario = factor(
      scenario,
      levels = names(SCENARIOS_HOURS_PER_DAY)
    ),
    minute_difference = point_difference,
    minute_difference_ci_lower = contrast_ci[1],
    minute_difference_ci_upper = contrast_ci[2],
    annual_difference = minute_difference * annual_hours,
    annual_difference_ci_lower = minute_difference_ci_lower * annual_hours,
    annual_difference_ci_upper = minute_difference_ci_upper * annual_hours
  ) %>%
  arrange(scenario)

write_csv(
  difference_summary,
  file.path(out_dir, "figure7_trait_counterfactual_differences.csv")
)

# Table 5 support file, including the minute-level row.
minute_summary <- scenario_summary %>%
  filter(scenario == names(SCENARIOS_HOURS_PER_DAY)[1]) %>%
  select(condition, minute_mean, minute_q05, minute_q95)

table5_minute <- minute_summary %>%
  select(condition, mean = minute_mean, q05 = minute_q05, q95 = minute_q95) %>%
  pivot_wider(
    names_from = condition,
    values_from = c(mean, q05, q95)
  ) %>%
  transmute(
    driving_schedule = "Minute-level predicted activation (bpm)",
    lowest_trait_anxiety = sprintf(
      "%.2f [%.2f-%.2f]",
      .data[[paste0("mean_", TRAIT_LABEL_LOW)]],
      .data[[paste0("q05_", TRAIT_LABEL_LOW)]],
      .data[[paste0("q95_", TRAIT_LABEL_LOW)]]
    ),
    highest_trait_anxiety = sprintf(
      "%.2f [%.2f-%.2f]",
      .data[[paste0("mean_", TRAIT_LABEL_HIGH)]],
      .data[[paste0("q05_", TRAIT_LABEL_HIGH)]],
      .data[[paste0("q95_", TRAIT_LABEL_HIGH)]]
    ),
    high_minus_low = sprintf(
      "%.2f [%.2f-%.2f]",
      point_difference,
      contrast_ci[1],
      contrast_ci[2]
    )
  )

table5_annual <- scenario_summary %>%
  mutate(
    scenario = factor(
      scenario,
      levels = names(SCENARIOS_HOURS_PER_DAY)
    )
  ) %>%
  arrange(scenario, condition) %>%
  select(
    scenario, condition,
    annual_mean, annual_q05, annual_q95
  ) %>%
  pivot_wider(
    names_from = condition,
    values_from = c(annual_mean, annual_q05, annual_q95)
  ) %>%
  left_join(
    difference_summary %>% select(
      scenario, annual_difference,
      annual_difference_ci_lower, annual_difference_ci_upper
    ),
    by = "scenario"
  ) %>%
  transmute(
    driving_schedule = paste0(scenario, " (NHR-hours/year)"),
    lowest_trait_anxiety = sprintf(
      "%.0f [%.0f-%.0f]",
      .data[[paste0("annual_mean_", TRAIT_LABEL_LOW)]],
      .data[[paste0("annual_q05_", TRAIT_LABEL_LOW)]],
      .data[[paste0("annual_q95_", TRAIT_LABEL_LOW)]]
    ),
    highest_trait_anxiety = sprintf(
      "%.0f [%.0f-%.0f]",
      .data[[paste0("annual_mean_", TRAIT_LABEL_HIGH)]],
      .data[[paste0("annual_q05_", TRAIT_LABEL_HIGH)]],
      .data[[paste0("annual_q95_", TRAIT_LABEL_HIGH)]]
    ),
    high_minus_low = sprintf(
      "%.0f [%.0f-%.0f]",
      annual_difference,
      annual_difference_ci_lower,
      annual_difference_ci_upper
    )
  )

table5 <- bind_rows(table5_minute, table5_annual)

write_csv(
  table5,
  file.path(out_dir, "table5_long_horizon_scenario_projections.csv")
)

write_csv(
  tibble(
    condition = c(TRAIT_LABEL_LOW, TRAIT_LABEL_HIGH),
    trait_variable = trait_var,
    trait_value = c(trait_low, trait_high),
    n_driving_rows_evaluated = nrow(model_df),
    n_participants = n_distinct(model_df$p_id),
    baseline_handling = "Observed participant-specific baseline retained",
    other_predictors = "All other observed row-level predictors retained"
  ),
  file.path(out_dir, "figure7_trait_extreme_parameters.csv")
)


# ============================================================
# MANUSCRIPT REPRODUCIBILITY CHECK
# ============================================================

minute_check <- scenario_summary %>%
  filter(
    as.character(scenario) == names(SCENARIOS_HOURS_PER_DAY)[1]
  ) %>%
  select(condition, minute_mean) %>%
  mutate(condition = as.character(condition)) %>%
  pivot_wider(
    names_from = condition,
    values_from = minute_mean
  )

regen_low <- minute_check[[TRAIT_LABEL_LOW]]
regen_high <- minute_check[[TRAIT_LABEL_HIGH]]
regen_diff <- point_difference

check_tbl <- tibble(
  quantity = c(
    "Lowest trait-anxiety minute NHR",
    "Highest trait-anxiety minute NHR",
    "High-minus-low minute NHR"
  ),
  manuscript_value = c(
    MANUSCRIPT_LOW_MINUTE_NHR,
    MANUSCRIPT_HIGH_MINUTE_NHR,
    MANUSCRIPT_DIFF_MINUTE_NHR
  ),
  regenerated_value = c(
    regen_low,
    regen_high,
    regen_diff
  )
) %>%
  mutate(
    absolute_difference = abs(regenerated_value - manuscript_value),
    within_tolerance = absolute_difference <= MANUSCRIPT_CHECK_TOL_BPM
  )

write_csv(
  check_tbl,
  file.path(out_dir, "figure7_manuscript_reproducibility_check.csv")
)

log_msg(
  "Manuscript check | low=", sprintf("%.3f", regen_low),
  " | high=", sprintf("%.3f", regen_high),
  " | difference=", sprintf("%.3f", regen_diff)
)

if (!all(check_tbl$within_tolerance)) {
  log_msg(
    "NOTE: Reconstructed Figure 7A/Table 5 values differ from the legacy manuscript references, ",
    "because all saved driving predictors are now restored and the full-data model uses the Script 00 median hyperparameters. ",
    "See figure7_manuscript_reproducibility_check.csv."
  )
} else {
  log_msg(
    "Reconstructed Figure 7A/Table 5 values happen to agree with legacy manuscript references within ",
    MANUSCRIPT_CHECK_TOL_BPM,
    " bpm."
  )
}

# ============================================================
# FIGURE 7A
# ============================================================

plot_a_df <- scenario_summary %>%
  mutate(
    scenario = factor(
      scenario,
      levels = names(SCENARIOS_HOURS_PER_DAY),
      labels = c(
        "30 min/day\ncommute",
        "2 hr/day\ncommute",
        "8 hr/day\nprofessional driving"
      )
    ),
    condition = factor(
      condition,
      levels = c(TRAIT_LABEL_LOW, TRAIT_LABEL_HIGH)
    )
  )

pA <- ggplot(
  plot_a_df,
  aes(x = scenario, y = annual_mean, fill = condition)
) +
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.70,
    color = "black",
    linewidth = 0.35
  ) +
  geom_errorbar(
    aes(ymin = annual_q05, ymax = annual_q95),
    position = position_dodge(width = 0.78),
    width = 0.16,
    linewidth = 0.65
  ) +
  scale_fill_manual(values = PAL_TRAIT) +
  scale_y_continuous(
    labels = label_number(big.mark = ",", accuracy = 1),
    expand = expansion(mult = c(0, 0.06))
  ) +
  labs(
    x = NULL,
    y = "Annual cumulative NHR-hours\n[bpm·hours]",
    fill = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(lineheight = 0.92),
    panel.grid.major.y = element_line(color = "grey88", linewidth = 0.3),
    plot.margin = margin(6, 8, 4, 4)
  )

# ============================================================
# FIGURE 7B: HORIZON-SCALING FROM HELD-OUT ENET PREDICTIONS
# ============================================================

# Reproduce the former Figure 8 analysis exactly: reconstruct the row map
# from the clean MASTER dataset, retain common subjects across strata,
# join OOF predictions by stratum and row_id, build contiguous segments,
# and retain complete non-overlapping blocks only.
USE_COMMON_SUBJECTS <- TRUE
MIN_BLOCKS_WARN <- 30L

dt_full <- fread(data_path, showProgress = TRUE)
dt_full <- canonicalize_names(dt_full)

need_b <- c("p_id", "time", "raw_hr", "activity3", "bl_hr")
miss_b <- setdiff(need_b, names(dt_full))
if (length(miss_b) > 0L) {
  stop("MASTER data missing for Figure 7B: ", paste(miss_b, collapse = ", "))
}

dt_full[, dt_time := parse_time_local(time, tz = LOCAL_TZ)]
dt_full <- dt_full[!is.na(dt_time)]
dt_full[, activity3_norm := norm_chr(activity3)]
dt_full[, day_key := make_day_key(dt_full)]

dt_drive_b <- dt_full[activity3_norm == "driving"]
dt_nond_b <- dt_full[activity3_norm == "non_driving_sedentary"]
if (nrow(dt_drive_b) == 0L) stop("No DRIVING rows found for Figure 7B.")
if (nrow(dt_nond_b) == 0L) stop("No NONDRIVING_SEDENTARY rows found for Figure 7B.")

if (USE_COMMON_SUBJECTS) {
  common_subj <- intersect(unique(dt_drive_b$p_id), unique(dt_nond_b$p_id))
  dt_drive_b <- dt_drive_b[p_id %in% common_subj]
  dt_nond_b <- dt_nond_b[p_id %in% common_subj]
  log_msg("Figure 7B common subjects: ", length(common_subj))
}

setorderv(dt_drive_b, c("p_id", "dt_time"))
setorderv(dt_nond_b, c("p_id", "dt_time"))

rowmap_drive <- copy(dt_drive_b)[
  , .(p_id, dt_time, day_key, stratum = "DRIVING")
][, row_id := seq_len(.N)]

rowmap_nond <- copy(dt_nond_b)[
  , .(p_id, dt_time, day_key, stratum = "NONDRIVING_SEDENTARY")
][, row_id := seq_len(.N)]

rowmap <- rbindlist(list(rowmap_drive, rowmap_nond), use.names = TRUE)
fwrite(rowmap_drive, file.path(out_dir, "figure7B_diagnostics_rowmap_DRIVING.csv"))
fwrite(rowmap_nond, file.path(out_dir, "figure7B_diagnostics_rowmap_NONDRIVING_SEDENTARY.csv"))

pred <- fread(run_inputs["predictions"])
need_pred <- c("model", "stratum", "fold", "row_id", "raw_hr_obs", "raw_hr_hat")
miss_pred <- setdiff(need_pred, names(pred))
if (length(miss_pred) > 0L) {
  stop("Prediction file missing columns: ", paste(miss_pred, collapse = ", "))
}

pred <- pred[model == "enet"]
if (nrow(pred) == 0L) stop("No ENet rows in prediction file.")

# The current prediction export may already contain participant/time mapping
# columns. Remove them before joining the authoritative row map reconstructed
# from the MASTER dataset; otherwise data.table::merge() creates p_id.x/p_id.y
# and dt_time.x/dt_time.y, leaving no column named p_id or dt_time.
mapping_cols_in_pred <- intersect(
  c("p_id", "dt_time", "day_key"),
  names(pred)
)
if (length(mapping_cols_in_pred) > 0L) {
  log_msg(
    "Dropping pre-existing prediction mapping columns before Figure 7B join: ",
    paste(mapping_cols_in_pred, collapse = ", ")
  )
  pred[, (mapping_cols_in_pred) := NULL]
}

pred <- merge(
  pred,
  rowmap,
  by = c("stratum", "row_id"),
  all.x = TRUE,
  all.y = FALSE,
  sort = FALSE
)

if (pred[, any(is.na(p_id) | is.na(dt_time))]) {
  bad_n <- pred[is.na(p_id) | is.na(dt_time), .N]
  stop("Failed to reconstruct row mapping for ", bad_n, " prediction rows.")
}

pred[, raw_resid := as.numeric(raw_hr_obs) - as.numeric(raw_hr_hat)]
setorderv(pred, c("stratum", "p_id", "dt_time"))

pred[, dt_diff_sec := as.numeric(
  difftime(dt_time, shift(dt_time), units = "secs")
), by = .(stratum, p_id)]
pred[, day_key_prev := shift(day_key), by = .(stratum, p_id)]

pred[, new_segment := fifelse(
  is.na(dt_diff_sec) |
    dt_diff_sec <= 0 |
    dt_diff_sec > MAX_GAP_MULTIPLIER * RES_SECONDS |
    is.na(day_key_prev) |
    day_key != day_key_prev,
  1L, 0L
), by = .(stratum, p_id)]

pred[, segment_id := cumsum(new_segment), by = .(stratum, p_id)]

compute_horizon_blocks <- function(dt_in, horizon_min, res_seconds) {
  k <- as.integer(round((horizon_min * 60) / res_seconds))
  if (k < 1L) stop("Invalid horizon.")

  copy(dt_in)[
    order(stratum, p_id, dt_time),
    {
      idx <- seq_len(.N)
      block_id <- ((idx - 1L) %/% k) + 1L
      .(
        horizon_min = horizon_min,
        block_id = block_id,
        raw_resid = raw_resid
      )
    },
    by = .(stratum, p_id, segment_id)
  ][
    ,
    .(
      n_rows = .N,
      raw_mean_resid = mean(raw_resid, na.rm = TRUE)
    ),
    by = .(stratum, p_id, segment_id, horizon_min, block_id)
  ][n_rows == k]
}

blocks_all <- rbindlist(
  lapply(HORIZONS_MIN, function(hm) {
    compute_horizon_blocks(pred, hm, RES_SECONDS)
  }),
  use.names = TRUE,
  fill = TRUE
)
if (nrow(blocks_all) == 0L) stop("No complete Figure 7B blocks formed.")

fwrite(blocks_all, file.path(out_dir, "figure7_horizon_block_means_long.csv"))

horizon_summary <- blocks_all[
  ,
  .(
    rmse = sqrt(mean(raw_mean_resid^2, na.rm = TRUE)),
    n_blocks = .N
  ),
  by = .(stratum, horizon_min)
]
horizon_summary[, normalized_rmse := rmse / rmse[horizon_min == min(horizon_min)], by = stratum]
horizon_summary[, warn_sparse := n_blocks < MIN_BLOCKS_WARN]
horizon_summary[, stratum_plot := factor(
  stratum,
  levels = c("DRIVING", "NONDRIVING_SEDENTARY"),
  labels = c("Driving", "Non-driving sedentary")
)]

write_csv(
  as_tibble(horizon_summary),
  file.path(out_dir, "figure7_horizon_scaling_summary.csv")
)

pB <- ggplot(
  horizon_summary,
  aes(
    x = horizon_min,
    y = normalized_rmse,
    color = stratum_plot,
    group = stratum_plot
  )
) +
  geom_line(linewidth = 1.25, lineend = "round") +
  geom_point(aes(shape = warn_sparse), size = 3.1, stroke = 1.15) +
  scale_color_manual(values = PAL_STRATUM, name = NULL) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1), guide = "none") +
  scale_x_log10(
    breaks = HORIZONS_MIN,
    labels = HORIZONS_MIN,
    expand = expansion(mult = c(0.04, 0.04))
  ) +
  scale_y_log10(
    breaks = c(0.80, 0.90, 1.00),
    labels = label_number(accuracy = 0.01),
    expand = expansion(mult = c(0.04, 0.06))
  ) +
  labs(
    x = "Averaging horizon [min, log scale]",
    y = "Normalized RMSE"
  ) +
  theme_classic(base_size = 14, base_family = "sans") +
  theme(
    legend.position = "bottom",
    legend.justification = "center",
    legend.text = element_text(size = 12),
    legend.key.width = grid::unit(0.85, "cm"),
    axis.title = element_text(size = 14),
    axis.title.x = element_text(margin = margin(t = 9)),
    axis.title.y = element_text(margin = margin(r = 9)),
    axis.text = element_text(size = 12),
    axis.line = element_line(linewidth = 0.55, color = "black"),
    axis.ticks = element_line(linewidth = 0.45, color = "black"),
    panel.grid.major = element_line(linewidth = 0.30, color = "grey88"),
    panel.grid.minor = element_blank(),
    plot.margin = margin(8, 10, 4, 8)
  )

# ============================================================
# SAVE FIGURE 7
# ============================================================

# Combine panels A and B into a single Figure 7.
# Lowercase panel labels are generated automatically by patchwork.
fig7 <- (pA | pB) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(face = "bold", size = 13)
  )

pdf_path <- file.path(fig_dir, "Figure7_LongHorizon_Behavior.pdf")
png_path <- file.path(fig_dir, "Figure7_LongHorizon_Behavior.png")

safe_save_pdf(fig7, pdf_path, w = 11.2, h = 5.4)
ggsave(
  png_path,
  fig7,
  width = 11.2,
  height = 5.4,
  dpi = PNG_DPI
)

log_msg("Saved: ", pdf_path)
log_msg("Saved: ", png_path)
log_msg("Wrote Table 5 support file and manuscript reproducibility check.")
log_msg("DONE. Outputs in: ", out_dir)
