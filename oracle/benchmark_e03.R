suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
})

argument_value <- function(arguments, name) {
  index <- match(name, arguments)
  if (is.na(index) || index == length(arguments)) {
    stop(sprintf("missing required argument %s", name))
  }
  arguments[[index + 1L]]
}

elapsed <- function(expression) {
  started <- proc.time()[["elapsed"]]
  value <- force(expression)
  list(value = value, seconds = unname(proc.time()[["elapsed"]] - started))
}

percentile <- function(samples, probability) {
  ordered <- sort(as.numeric(samples))
  ordered[[max(1L, ceiling(probability * length(ordered)))]]
}

summary_samples <- function(samples) {
  list(
    samples_seconds = as.numeric(samples),
    median_seconds = unname(median(samples)),
    p95_seconds = unname(percentile(samples, 0.95))
  )
}

peak_rss_megabytes <- function() {
  lines <- readLines("/proc/self/status", warn = FALSE)
  value <- grep("^VmHWM:", lines, value = TRUE)
  if (length(value) != 1L) return(NA_real_)
  as.numeric(gsub("[^0-9]", "", value)) / 1000
}

sha256_file <- function(path) {
  output <- system2("sha256sum", path, stdout = TRUE)
  if (!identical(attr(output, "status"), NULL)) {
    stop(sprintf("sha256sum failed for %s", path))
  }
  strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
}

control <- lmerControl(
  optimizer = "nloptwrap",
  restart_edge = TRUE,
  boundary.tol = 1e-5,
  check.conv.singular = .makeCC(action = "ignore", tol = 1e-4),
  optCtrl = list(
    algorithm = "NLOPT_LN_BOBYQA",
    xtol_abs = 1e-8,
    ftol_abs = 1e-8,
    maxeval = 100000L
  )
)

single_group_problem <- function(manifest, scenario_id) {
  generator <- manifest$generator
  observations <- as.integer(generator$observations)
  groups <- as.integer(generator$groups)
  observations_per_group <- observations %/% groups
  group_indices <- rep(0:(groups - 1L), each = observations_per_group)
  row <- 0:(observations - 1L)
  group_signal <- 0.8 * sin((0:(groups - 1L)) * 0.013)
  residual <- 0.2 * sin(row * 0.17) + 0.1 * cos(row * 0.071)
  argument_offset <- 0.05 * cos(row * 0.031)
  weights <- 0.75 + (row %% 11L) / 10.0
  levels <- sprintf("group-%05d", 0:(groups - 1L))
  group <- factor(levels[group_indices + 1L], levels = levels)

  if (scenario_id == "random-intercept") {
    y <- 2.5 + group_signal[group_indices + 1L] + residual + argument_offset
    frame <- data.frame(y = y, group = group)
  } else if (scenario_id == "fixed-categorical") {
    x <- rep(seq(-1.0, 1.0, length.out = observations_per_group), groups)
    factor_codes <- row %% 3L
    factor_levels <- c("baseline", "high", "low")
    fixed_factor <- factor(
      factor_levels[factor_codes + 1L], levels = factor_levels
    )
    factor_signal <- c(0.0, 0.35, -0.2)[factor_codes + 1L]
    formula_offset <- 0.03 * sin(row * 0.023)
    y <- 2.5 + 0.7 * x + factor_signal + group_signal[group_indices + 1L] +
      residual + argument_offset + formula_offset
    frame <- data.frame(
      y = y, x = x, factor = fixed_factor,
      formula_offset = formula_offset, group = group
    )
  } else if (scenario_id %in% c(
    "correlated-random-slope", "independent-random-terms"
  )) {
    x <- rep(seq(-1.0, 1.0, length.out = observations_per_group), groups)
    slope_signal <- 0.3 * cos((0:(groups - 1L)) * 0.019)
    y <- 2.5 + 0.7 * x + group_signal[group_indices + 1L] +
      slope_signal[group_indices + 1L] * x + residual + argument_offset
    frame <- data.frame(y = y, x = x, group = group)
  } else {
    stop(sprintf("unknown scenario %s", scenario_id))
  }
  list(frame = frame, weights = weights, offset = argument_offset)
}

worker <- function(arguments) {
  single_manifest_path <- argument_value(arguments, "--single-manifest")
  sparse_manifest_path <- argument_value(arguments, "--sparse-manifest")
  scenario_id <- argument_value(arguments, "--scenario")
  kind <- argument_value(arguments, "--kind")
  reml <- identical(kind, "reml")
  warnings <- character()

  if (scenario_id == "insteval") {
    manifest <- fromJSON(sparse_manifest_path, simplifyVector = FALSE)
    fixture <- fromJSON(manifest$fixture, simplifyVector = FALSE)
    data("InstEval", package = "lme4", envir = environment())
    frame <- InstEval
    formula <- as.formula(fixture$formula)
    weights <- NULL
    argument_offset <- NULL
    expected <- Filter(function(item) identical(item$kind, kind), fixture$fits)[[1L]]
    repetitions <- as.integer(manifest$measurement$end_to_end_repetitions)
    fixed_repetitions <- as.integer(manifest$measurement$fixed_theta_repetitions)
  } else {
    manifest <- fromJSON(single_manifest_path, simplifyVector = FALSE)
    scenario <- Filter(
      function(item) identical(item$id, scenario_id), manifest$scenarios
    )[[1L]]
    problem <- single_group_problem(manifest, scenario_id)
    frame <- problem$frame
    weights <- problem$weights
    argument_offset <- problem$offset
    formula <- as.formula(scenario$formula)
    expected <- NULL
    repetitions <- as.integer(manifest$measurement$end_to_end_repetitions)
    fixed_repetitions <- as.integer(
      manifest$measurement$fixed_theta_warm_repetitions
    )
  }

  fit_once <- function() {
    withCallingHandlers(
      lmer(
        formula, data = frame, REML = reml, weights = weights,
        offset = argument_offset, control = control
      ),
      warning = function(condition) {
        warnings <<- c(warnings, conditionMessage(condition))
        invokeRestart("muffleWarning")
      }
    )
  }

  fit_samples <- numeric(repetitions)
  fit <- NULL
  for (index in seq_len(repetitions)) {
    gc()
    measurement <- elapsed(fit_once())
    fit <- measurement$value
    fit_samples[[index]] <- measurement$seconds
  }

  parsed_measurement <- elapsed(
    lFormula(
      formula, data = frame, REML = reml, weights = weights,
      offset = argument_offset, control = control
    )
  )
  parsed <- parsed_measurement$value
  devfun_measurement <- elapsed(
    do.call(mkLmerDevfun, c(parsed, list(control = control)))
  )
  devfun <- devfun_measurement$value
  theta <- getME(fit, "theta")
  fixed_samples <- numeric(fixed_repetitions)
  for (index in seq_len(fixed_repetitions)) {
    fixed_samples[[index]] <- elapsed(devfun(theta))$seconds
  }
  optimization_measurement <- elapsed(
    optimizeLmer(
      devfun,
      optimizer = control$optimizer,
      restart_edge = control$restart_edge,
      boundary.tol = control$boundary.tol,
      control = control$optCtrl
    )
  )

  convergence_messages <- fit@optinfo$conv$lme4$messages
  objective <- unname(as.numeric(-2 * logLik(fit, REML = reml)))
  correctness <- is.finite(objective) &&
    all(is.finite(fixef(fit))) &&
    all(is.finite(theta)) &&
    is.null(convergence_messages)
  errors <- list()
  if (!is.null(expected)) {
    errors <- list(
      objective_absolute_error = abs(objective - as.numeric(expected$objective)),
      theta_max_absolute_error = max(abs(theta - unlist(expected$theta))),
      beta_max_absolute_error = max(abs(fixef(fit) - unlist(expected$beta))),
      sigma_absolute_error = abs(sigma(fit) - as.numeric(expected$sigma))
    )
    tolerances <- manifest$tolerances
    correctness <- correctness &&
      errors$objective_absolute_error <= as.numeric(tolerances$objective_absolute) &&
      errors$theta_max_absolute_error <= as.numeric(tolerances$theta_absolute) &&
      errors$beta_max_absolute_error <= as.numeric(tolerances$beta_absolute) &&
      errors$sigma_absolute_error <= as.numeric(tolerances$sigma_absolute)
  }

  result <- list(
    scenario = scenario_id,
    kind = kind,
    dimensions = list(
      n = nrow(frame),
      p = length(fixef(fit)),
      q = nrow(getME(fit, "Lambda")),
      d = length(theta),
      random_design_nonzeros = Matrix::nnzero(getME(fit, "Zt"))
    ),
    timings = list(
      end_to_end = summary_samples(fit_samples),
      end_to_end_cold_seconds = fit_samples[[1L]],
      end_to_end_warm = summary_samples(if (length(fit_samples) > 1L) {
        fit_samples[-1L]
      } else {
        fit_samples
      }),
      parse_and_encoding_seconds = parsed_measurement$seconds,
      devfun_assembly_seconds = devfun_measurement$seconds,
      symbolic_analysis_status = "included-in-lme4-devfun-and-factorization",
      fixed_theta = summary_samples(fixed_samples),
      fixed_theta_cold_seconds = fixed_samples[[1L]],
      fixed_theta_warm = summary_samples(if (length(fixed_samples) > 1L) {
        fixed_samples[-1L]
      } else {
        fixed_samples
      }),
      optimization_seconds_preassembled = optimization_measurement$seconds,
      inference_status = "not-measured-fit-benchmark"
    ),
    optimizer = list(
      name = "nloptwrap/NLOPT_LN_BOBYQA",
      evaluations = fit@optinfo$feval,
      singular = isSingular(fit, tol = 1e-4)
    ),
    fit = list(
      objective = objective,
      theta = as.numeric(theta),
      beta = as.numeric(fixef(fit)),
      sigma = unname(sigma(fit))
    ),
    peak_rss_megabytes = peak_rss_megabytes(),
    checks = c(list(
      correctness_pass = correctness,
      convergence_messages = if (is.null(convergence_messages)) NULL else {
        as.character(convergence_messages)
      },
      recorded_warnings = unique(warnings)
    ), errors)
  )
  cat(toJSON(result, auto_unbox = TRUE, digits = 17, null = "null"))
}

parent <- function(arguments) {
  single_manifest_path <- argument_value(arguments, "--single-manifest")
  sparse_manifest_path <- argument_value(arguments, "--sparse-manifest")
  output_path <- argument_value(arguments, "--output")
  revision <- argument_value(arguments, "--revision")
  single_manifest <- fromJSON(single_manifest_path, simplifyVector = FALSE)
  sparse_manifest <- fromJSON(sparse_manifest_path, simplifyVector = FALSE)
  scenarios <- c(vapply(single_manifest$scenarios, `[[`, "", "id"), "insteval")
  kinds <- c("ml", "reml")
  cases <- list()
  for (scenario in scenarios) {
    for (kind in kinds) {
      stderr_path <- tempfile("kamino-e03-r-stderr-")
      command <- c(
        "oracle/benchmark_e03.R", "--worker",
        "--single-manifest", single_manifest_path,
        "--sparse-manifest", sparse_manifest_path,
        "--scenario", scenario, "--kind", kind
      )
      timeout_seconds <- if (scenario == "insteval") {
        sparse_manifest$measurement$worker_timeout_seconds
      } else {
        single_manifest$measurement$worker_timeout_seconds
      }
      lines <- system2(
        "timeout",
        c("--signal=TERM", paste0(timeout_seconds, "s"), "Rscript", command),
        stdout = TRUE,
        stderr = stderr_path
      )
      status <- attr(lines, "status")
      if (!is.null(status) && status != 0L) {
        stop(paste(readLines(stderr_path, warn = FALSE), collapse = "\n"))
      }
      unlink(stderr_path)
      cases[[length(cases) + 1L]] <- fromJSON(
        paste(lines, collapse = "\n"), simplifyVector = FALSE
      )
    }
  }
  report <- list(
    schema_version = "1.0.0",
    benchmark_id = "e03-lme4-comparison-v1",
    revision = revision,
    manifests = list(
      single_group = list(
        path = single_manifest_path,
        sha256 = sha256_file(single_manifest_path)
      ),
      general_sparse = list(
        path = sparse_manifest_path,
        sha256 = sha256_file(sparse_manifest_path)
      )
    ),
    environment = list(
      runtime = R.version.string,
      platform = R.version$platform,
      lme4 = as.character(packageVersion("lme4")),
      Matrix = as.character(packageVersion("Matrix")),
      nloptr = as.character(packageVersion("nloptr")),
      blas = extSoftVersion()[["BLAS"]],
      blas_threads = 1L,
      numeric_precision = "IEEE-754 binary64",
      hardware_profile = single_manifest$hardware_profile,
      execution_boundary = "Docker Desktop Linux/ARM64 oracle",
      oracle_image = paste0(
        "ghcr.io/joshuamyers22/kamino-oracle@sha256:",
        "17e45268be294316967064d0600a727463d738d7dfb0c68ec43769baebe8eb4d"
      )
    ),
    cases = cases,
    correctness_pass = all(vapply(
      cases, function(item) isTRUE(item$checks$correctness_pass), logical(1)
    )),
    comparative_timing_status = "observational-only"
  )
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(
    toJSON(report, auto_unbox = TRUE, pretty = TRUE, digits = 17, null = "null"),
    output_path
  )
  cat(toJSON(report, auto_unbox = TRUE, pretty = TRUE, digits = 17, null = "null"))
}

arguments <- commandArgs(trailingOnly = TRUE)
if ("--worker" %in% arguments) {
  worker(arguments)
} else {
  parent(arguments)
}
