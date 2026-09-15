#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
})

output_dir <- Sys.getenv("KAMINO_ORACLE_OUTPUT", "oracle/output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data <- data.frame(
  row_id = sprintf("r%02d", seq_len(12)),
  y = c(1.10, 1.95, 2.70, 4.15, 2.20, 2.85, 4.10, 4.75, 0.70, 1.80, 2.25, 3.90),
  x = rep(c(-1.5, -0.5, 0.5, 1.5), 3),
  g = factor(rep(c("beta", "alpha", "gamma"), each = 4), levels = c("gamma", "alpha", "beta")),
  f = factor(
    c("middle", "low", "high", "low", "high", "middle", "low", "high", "low", "high", "middle", "middle"),
    levels = c("middle", "high", "low")
  ),
  h = factor(rep(c("one", "two"), 6), levels = c("two", "one")),
  w = c(0.5, 1.0, 1.5, 2.0, 1.25, 0.75, 2.5, 1.75, 0.8, 1.2, 1.8, 2.2),
  o = c(0.10, -0.05, 0.20, 0.00, -0.10, 0.15, 0.05, -0.20, 0.12, -0.08, 0.18, -0.02),
  stringsAsFactors = FALSE
)
rownames(data) <- data$row_id

theta_cases <- list(
  diagonal = c(0.8, 0.0, 0.4),
  correlated = c(0.8, 0.35, 0.4),
  singular_slope = c(0.8, 0.0, 0.0),
  zero = c(0.0, 0.0, 0.0)
)

scalar <- function(x) unname(as.numeric(x))

matrix_sha256 <- function(value) {
  path <- tempfile("kamino-matrix-")
  connection <- file(path, open = "wb")
  canonical <- as.double(value)
  canonical[canonical == 0] <- 0.0
  writeBin(canonical, connection, size = 8, endian = "little")
  close(connection)
  output <- system2("sha256sum", path, stdout = TRUE)
  unlink(path)
  strsplit(output, " ", fixed = TRUE)[[1]][[1]]
}

make_case <- function(reml, case_name, theta) {
  parsed <- lFormula(
    y ~ x + offset(o) + (1 + x | g),
    data = data,
    weights = w,
    REML = reml,
    control = lmerControl(
      optimizer = "nloptwrap",
      optCtrl = list(
        algorithm = "NLOPT_LN_BOBYQA",
        xtol_abs = 1e-8,
        ftol_abs = 1e-8,
        maxeval = 100000
      )
    )
  )
  devfun <- do.call(mkLmerDevfun, parsed)
  objective <- devfun(theta)
  state <- environment(devfun)

  lambdat <- parsed$reTrms$Lambdat
  lambdat@x <- theta[parsed$reTrms$Lind]
  lambda <- t(as.matrix(lambdat))
  random_names <- unlist(lapply(
    levels(parsed$fr$g),
    function(level) sprintf("g[%s]:%s", level, parsed$reTrms$cnms[[1]])
  ))

  list(
    id = sprintf("synthetic_random_slope_%s_%s", if (reml) "reml" else "ml", case_name),
    reml = reml,
    theta_case = case_name,
    theta = unname(theta),
    formula = "y ~ x + offset(o) + (1 + x | g)",
    row_ids = as.character(data$row_id),
    fixed_names = colnames(parsed$X),
    random_names = unname(random_names),
    y = unname(parsed$fr[[as.character(parsed$formula[[2]])]]),
    weights = unname(model.weights(parsed$fr)),
    offset = unname(model.offset(parsed$fr)),
    X = unname(as.matrix(parsed$X)),
    Z = unname(t(as.matrix(parsed$reTrms$Zt))),
    Lambda = unname(lambda),
    objective = scalar(objective),
    components = list(
      ldL2 = scalar(state$pp$ldL2()),
      ldRX2 = scalar(state$pp$ldRX2()),
      wrss = scalar(state$resp$wrss()),
      pwrss = scalar(state$pp$sqrL(1) + state$resp$wrss()),
      beta = scalar(state$pp$beta(1)),
      u = scalar(state$pp$u(1))
    )
  )
}

cases <- list()
for (reml in c(FALSE, TRUE)) {
  for (case_name in names(theta_cases)) {
    cases[[length(cases) + 1L]] <- make_case(reml, case_name, theta_cases[[case_name]])
  }
}

make_fit <- function(reml) {
  parsed <- lFormula(
    y ~ x + offset(o) + (1 + x | g),
    data = data,
    weights = w,
    REML = reml
  )
  devfun <- do.call(mkLmerDevfun, parsed)
  optimum <- optimizeLmer(
    devfun,
    optimizer = "nloptwrap",
    restart_edge = TRUE,
    boundary.tol = 1e-5,
    control = list(
      algorithm = "NLOPT_LN_BOBYQA",
      xtol_abs = 1e-8,
      ftol_abs = 1e-8,
      maxeval = 100000
    )
  )
  model <- mkMerMod(
    environment(devfun),
    optimum,
    parsed$reTrms,
    parsed$fr,
    mc = match.call()
  )
  list(
    id = sprintf("synthetic_random_slope_%s_fit", if (reml) "reml" else "ml"),
    reml = reml,
    objective = scalar(optimum$fval),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    u = scalar(getME(model, "u")),
    sigma = scalar(sigma(model)),
    evaluations = unname(as.integer(optimum$feval)),
    convergence_code = unname(as.integer(optimum$conv)),
    convergence_message = unname(as.character(optimum$message))
  )
}

fits <- lapply(c(FALSE, TRUE), make_fit)

make_dyestuff_fit <- function(reml, frame, dataset_id) {
  control <- lmerControl(
    optimizer = "nloptwrap",
    restart_edge = TRUE,
    boundary.tol = 1e-5,
    optCtrl = list(
      algorithm = "NLOPT_LN_BOBYQA",
      xtol_abs = 1e-8,
      ftol_abs = 1e-8,
      maxeval = 100000
    )
  )
  model <- lmer(Yield ~ 1 + (1 | Batch), data = frame, REML = reml, control = control)
  levels <- levels(frame$Batch)
  existing <- data.frame(Batch = factor(levels, levels = levels))
  unseen <- data.frame(Batch = factor("new", levels = c(levels, "new")))
  optimizer <- model@optinfo
  convergence_message <- optimizer$conv$lme4$messages
  if (is.null(convergence_message)) convergence_message <- character()

  list(
    id = sprintf("%s_random_intercept_%s", dataset_id, if (reml) "reml" else "ml"),
    kind = if (reml) "reml" else "ml",
    formula = "Yield ~ 1 + (1 | Batch)",
    objective = scalar(-2 * logLik(model, REML = reml)),
    log_likelihood = scalar(logLik(model, REML = reml)),
    theta = unname(as.list(scalar(getME(model, "theta")))),
    beta = unname(as.list(scalar(fixef(model)))),
    sigma = scalar(sigma(model)),
    random_variance = scalar(as.matrix(VarCorr(model)$Batch)[1, 1]),
    residual_variance = scalar(sigma(model)^2),
    beta_covariance = unname(as.matrix(vcov(model))),
    u = scalar(getME(model, "u")),
    b = scalar(getME(model, "b")),
    fitted = scalar(fitted(model)),
    residuals = scalar(residuals(model)),
    predictions = list(
      training_population = scalar(predict(model, re.form = NA)),
      training_conditional = scalar(predict(model, re.form = NULL)),
      existing_levels = levels,
      existing_population = scalar(predict(model, newdata = existing, re.form = NA)),
      existing_conditional = scalar(predict(model, newdata = existing, re.form = NULL)),
      new_level_population = scalar(predict(model, newdata = unseen, re.form = NA, allow.new.levels = TRUE)),
      new_level_conditional = scalar(predict(model, newdata = unseen, re.form = NULL, allow.new.levels = TRUE))
    ),
    evaluations = unname(as.integer(optimizer$feval)),
    convergence_code = unname(as.integer(optimizer$conv$opt)),
    convergence_messages = I(unname(as.character(convergence_message)))
  )
}

make_dyestuff_fixture <- function(frame, dataset_name, dataset_id, citation) {
  parsed <- lFormula(
    Yield ~ 1 + (1 | Batch),
    data = frame,
    REML = TRUE,
    na.action = na.fail
  )
  group_levels <- levels(parsed$fr$Batch)
  list(
    schema_version = "1.0.0",
    source = list(
      package = "lme4",
      dataset = dataset_name,
      citation = citation,
      license = "GPL (>= 2), following lme4 2.0-6 package metadata",
      modification = "Converted to JSON and augmented with lme4 fit, diagnostic, and prediction outputs"
    ),
    reference = list(
      profile = "lme4-2.0.6-unstructured-gaussian-v1",
      lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
    ),
    data = list(
      row_ids = sprintf("%s-%02d", dataset_id, seq_len(nrow(frame))),
      response_name = "Yield",
      response = scalar(frame$Yield),
      group_name = "Batch",
      groups = unname(as.character(frame$Batch)),
      group_levels = unname(group_levels),
      fixed_names = I(unname(colnames(parsed$X))),
      random_names = unname(sprintf("Batch[%s]:(Intercept)", group_levels)),
      X = unname(as.matrix(parsed$X)),
      Z = unname(t(as.matrix(parsed$reTrms$Zt)))
    ),
    fits = lapply(
      c(FALSE, TRUE),
      make_dyestuff_fit,
      frame = frame,
      dataset_id = dataset_id
    )
  )
}

dyestuff <- make_dyestuff_fixture(
  Dyestuff,
  "Dyestuff",
  "dyestuff",
  "O.L. Davies and P.L. Goldsmith (eds), Statistical Methods in Research and Production, 4th ed. (1972), section 6.4"
)
dyestuff2 <- make_dyestuff_fixture(
  Dyestuff2,
  "Dyestuff2",
  "dyestuff2",
  "G.E.P. Box and G.C. Tiao, Bayesian Inference in Statistical Analysis (1973), section 5.1.2; generated data"
)

sleepstudy_theta_cases <- list(
  zero = c(0.0, 0.0, 0.0),
  intercept_only = c(1.0, 0.0, 0.0),
  diagonal = c(1.0, 0.0, 0.25),
  correlated = c(1.0, 0.15, 0.25),
  near_boundary = c(0.8, -0.2, 1e-8)
)

make_sleepstudy_fixed_case <- function(reml, case_name, theta) {
  parsed <- lFormula(
    Reaction ~ Days + (1 + Days | Subject),
    data = sleepstudy,
    REML = reml,
    na.action = na.fail
  )
  devfun <- do.call(mkLmerDevfun, parsed)
  objective <- devfun(theta)
  state <- environment(devfun)
  lambdat <- parsed$reTrms$Lambdat
  lambdat@x <- theta[parsed$reTrms$Lind]
  lambda <- t(as.matrix(lambdat))
  u <- scalar(state$pp$u(1))
  list(
    id = sprintf("sleepstudy_%s_%s", if (reml) "reml" else "ml", case_name),
    kind = if (reml) "reml" else "ml",
    theta_case = case_name,
    theta = unname(theta),
    objective = scalar(objective),
    components = list(
      ldL2 = scalar(state$pp$ldL2()),
      ldRX2 = scalar(state$pp$ldRX2()),
      wrss = scalar(state$resp$wrss()),
      pwrss = scalar(state$pp$sqrL(1) + state$resp$wrss()),
      beta = scalar(state$pp$beta(1)),
      u = u,
      b = scalar(lambda %*% u)
    )
  )
}

sleepstudy_control <- function() {
  lmerControl(
    optimizer = "nloptwrap",
    restart_edge = TRUE,
    boundary.tol = 1e-5,
    optCtrl = list(
      algorithm = "NLOPT_LN_BOBYQA",
      xtol_abs = 1e-8,
      ftol_abs = 1e-8,
      maxeval = 100000
    )
  )
}

make_sleepstudy_fit <- function(reml) {
  model <- lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = sleepstudy,
    REML = reml,
    control = sleepstudy_control()
  )
  levels <- levels(sleepstudy$Subject)
  existing <- data.frame(
    Days = c(0, 5, 10),
    Subject = factor(levels[1:3], levels = levels)
  )
  unseen <- data.frame(
    Days = c(0, 5, 10),
    Subject = factor(rep("new", 3), levels = c(levels, "new"))
  )
  optimizer <- model@optinfo
  convergence_message <- optimizer$conv$lme4$messages
  if (is.null(convergence_message)) convergence_message <- character()
  covariance <- as.matrix(VarCorr(model)$Subject)
  list(
    id = sprintf("sleepstudy_correlated_slope_%s", if (reml) "reml" else "ml"),
    kind = if (reml) "reml" else "ml",
    formula = "Reaction ~ 1 + Days + (1 + Days | Subject)",
    objective = scalar(-2 * logLik(model, REML = reml)),
    log_likelihood = scalar(logLik(model, REML = reml)),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    sigma = scalar(sigma(model)),
    random_covariance = unname(covariance),
    residual_variance = scalar(sigma(model)^2),
    beta_covariance = unname(as.matrix(vcov(model))),
    u = scalar(getME(model, "u")),
    b = scalar(getME(model, "b")),
    fitted = scalar(fitted(model)),
    residuals = scalar(residuals(model)),
    predictions = list(
      training_population = scalar(predict(model, re.form = NA)),
      training_conditional = scalar(predict(model, re.form = NULL)),
      existing_levels = unname(as.character(existing$Subject)),
      existing_days = scalar(existing$Days),
      existing_population = scalar(predict(model, newdata = existing, re.form = NA)),
      existing_conditional = scalar(predict(model, newdata = existing, re.form = NULL)),
      new_days = scalar(unseen$Days),
      new_level_population = scalar(predict(model, newdata = unseen, re.form = NA, allow.new.levels = TRUE)),
      new_level_conditional = scalar(predict(model, newdata = unseen, re.form = NULL, allow.new.levels = TRUE))
    ),
    evaluations = unname(as.integer(optimizer$feval)),
    convergence_code = unname(as.integer(optimizer$conv$opt)),
    convergence_messages = I(unname(as.character(convergence_message)))
  )
}

make_sleepstudy_fixture <- function() {
  parsed <- lFormula(
    Reaction ~ Days + (1 + Days | Subject),
    data = sleepstudy,
    REML = TRUE,
    na.action = na.fail
  )
  levels <- levels(parsed$fr$Subject)
  random_names <- unlist(lapply(
    levels,
    function(level) sprintf("Subject[%s]:%s", level, parsed$reTrms$cnms[[1]])
  ))
  fixed_cases <- list()
  for (reml in c(FALSE, TRUE)) {
    for (case_name in names(sleepstudy_theta_cases)) {
      fixed_cases[[length(fixed_cases) + 1L]] <- make_sleepstudy_fixed_case(
        reml, case_name, sleepstudy_theta_cases[[case_name]]
      )
    }
  }
  list(
    schema_version = "1.0.0",
    source = list(
      package = "lme4",
      dataset = "sleepstudy",
      citation = "Belenky et al. (2003), Patterns of performance degradation and restoration during sleep restriction and subsequent recovery: a sleep dose-response study",
      license = "GPL (>= 2), following lme4 2.0-6 package metadata",
      modification = "Converted to JSON and augmented with lme4 fixed-theta, fit, diagnostic, and prediction outputs"
    ),
    reference = list(
      profile = "lme4-2.0.6-unstructured-gaussian-v1",
      lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
    ),
    data = list(
      row_ids = sprintf("sleepstudy-%03d", seq_len(nrow(sleepstudy))),
      response_name = "Reaction",
      response = scalar(sleepstudy$Reaction),
      predictor_name = "Days",
      predictor = scalar(sleepstudy$Days),
      group_name = "Subject",
      groups = unname(as.character(sleepstudy$Subject)),
      group_levels = unname(levels),
      group_indices = unname(as.integer(sleepstudy$Subject) - 1L),
      fixed_names = I(unname(colnames(parsed$X))),
      random_coefficient_names = I(unname(parsed$reTrms$cnms[[1]])),
      random_names = unname(random_names),
      X = unname(as.matrix(parsed$X)),
      random_design = unname(cbind(1.0, sleepstudy$Days)),
      Z_sha256 = matrix_sha256(t(as.matrix(parsed$reTrms$Zt)))
    ),
    fixed_theta_cases = fixed_cases,
    fits = lapply(c(FALSE, TRUE), make_sleepstudy_fit)
  )
}

sleepstudy_fixture <- make_sleepstudy_fixture()

sleepstudy_independent_theta_cases <- list(
  zero = c(0.0, 0.0),
  intercept_only = c(1.0, 0.0),
  slope_only = c(0.0, 0.25),
  diagonal = c(1.0, 0.25),
  near_slope_boundary = c(0.8, 1e-8)
)

group_major <- function(value, groups) {
  unname(as.vector(t(cbind(value[seq_len(groups)], value[groups + seq_len(groups)]))))
}

make_sleepstudy_independent_fixed_case <- function(reml, case_name, theta) {
  parsed <- lFormula(
    Reaction ~ Days + (1 + Days || Subject),
    data = sleepstudy,
    REML = reml,
    na.action = na.fail
  )
  devfun <- do.call(mkLmerDevfun, parsed)
  objective <- devfun(theta)
  state <- environment(devfun)
  groups <- nlevels(parsed$fr$Subject)
  lambdat <- parsed$reTrms$Lambdat
  lambdat@x <- theta[parsed$reTrms$Lind]
  lambda <- t(as.matrix(lambdat))
  u <- scalar(state$pp$u(1))
  list(
    id = sprintf("sleepstudy_independent_%s_%s", if (reml) "reml" else "ml", case_name),
    kind = if (reml) "reml" else "ml",
    theta_case = case_name,
    theta = unname(theta),
    objective = scalar(objective),
    components = list(
      ldL2 = scalar(state$pp$ldL2()),
      ldRX2 = scalar(state$pp$ldRX2()),
      wrss = scalar(state$resp$wrss()),
      pwrss = scalar(state$pp$sqrL(1) + state$resp$wrss()),
      beta = scalar(state$pp$beta(1)),
      u_group_major = group_major(u, groups),
      b_group_major = group_major(scalar(lambda %*% u), groups)
    )
  )
}

make_sleepstudy_independent_fit <- function(reml) {
  model <- lmer(
    Reaction ~ Days + (1 + Days || Subject),
    data = sleepstudy,
    REML = reml,
    control = sleepstudy_control()
  )
  levels <- levels(sleepstudy$Subject)
  groups <- length(levels)
  existing <- data.frame(
    Days = c(0, 5, 10),
    Subject = factor(levels[1:3], levels = levels)
  )
  unseen <- data.frame(
    Days = c(0, 5, 10),
    Subject = factor(rep("new", 3), levels = c(levels, "new"))
  )
  optimizer <- model@optinfo
  convergence_message <- optimizer$conv$lme4$messages
  if (is.null(convergence_message)) convergence_message <- character()
  theta <- scalar(getME(model, "theta"))
  sigma_value <- scalar(sigma(model))
  list(
    id = sprintf("sleepstudy_independent_slope_%s", if (reml) "reml" else "ml"),
    kind = if (reml) "reml" else "ml",
    formula = "Reaction ~ 1 + Days + (1 + Days || Subject)",
    adapter_formula = "Reaction ~ 1 + Days + (1 | Subject) + (0 + Days | Subject)",
    objective = scalar(-2 * logLik(model, REML = reml)),
    log_likelihood = scalar(logLik(model, REML = reml)),
    theta = theta,
    beta = scalar(fixef(model)),
    sigma = sigma_value,
    random_covariance_group_major = unname(diag((sigma_value * theta)^2)),
    residual_variance = sigma_value^2,
    beta_covariance = unname(as.matrix(vcov(model))),
    u_group_major = group_major(scalar(getME(model, "u")), groups),
    b_group_major = group_major(scalar(getME(model, "b")), groups),
    fitted = scalar(fitted(model)),
    residuals = scalar(residuals(model)),
    predictions = list(
      training_population = scalar(predict(model, re.form = NA)),
      training_conditional = scalar(predict(model, re.form = NULL)),
      existing_levels = unname(as.character(existing$Subject)),
      existing_days = scalar(existing$Days),
      existing_population = scalar(predict(model, newdata = existing, re.form = NA)),
      existing_conditional = scalar(predict(model, newdata = existing, re.form = NULL)),
      new_days = scalar(unseen$Days),
      new_level_population = scalar(predict(model, newdata = unseen, re.form = NA, allow.new.levels = TRUE)),
      new_level_conditional = scalar(predict(model, newdata = unseen, re.form = NULL, allow.new.levels = TRUE))
    ),
    evaluations = unname(as.integer(optimizer$feval)),
    convergence_code = unname(as.integer(optimizer$conv$opt)),
    convergence_messages = I(unname(as.character(convergence_message)))
  )
}

make_independent_boundary_fit <- function(reml) {
  model <- lmer(
    y ~ x + offset(o) + (1 + x || g),
    data = data,
    weights = w,
    REML = reml,
    control = sleepstudy_control()
  )
  optimizer <- model@optinfo
  convergence_message <- optimizer$conv$lme4$messages
  if (is.null(convergence_message)) convergence_message <- character()
  list(
    id = sprintf("synthetic_independent_boundary_%s", if (reml) "reml" else "ml"),
    kind = if (reml) "reml" else "ml",
    objective = scalar(-2 * logLik(model, REML = reml)),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    sigma = scalar(sigma(model)),
    evaluations = unname(as.integer(optimizer$feval)),
    convergence_code = unname(as.integer(optimizer$conv$opt)),
    convergence_messages = I(unname(as.character(convergence_message)))
  )
}

make_sleepstudy_independent_fixture <- function() {
  parsed <- lFormula(
    Reaction ~ Days + (1 + Days || Subject),
    data = sleepstudy,
    REML = TRUE,
    na.action = na.fail
  )
  levels <- levels(parsed$fr$Subject)
  groups <- length(levels)
  z_term_major <- t(as.matrix(parsed$reTrms$Zt))
  permutation <- as.vector(rbind(seq_len(groups), groups + seq_len(groups)))
  random_names <- unlist(lapply(
    levels,
    function(level) c(
      sprintf("Subject[%s]:(Intercept)", level),
      sprintf("Subject[%s]:Days", level)
    )
  ))
  fixed_cases <- list()
  for (reml in c(FALSE, TRUE)) {
    for (case_name in names(sleepstudy_independent_theta_cases)) {
      fixed_cases[[length(fixed_cases) + 1L]] <- make_sleepstudy_independent_fixed_case(
        reml, case_name, sleepstudy_independent_theta_cases[[case_name]]
      )
    }
  }
  list(
    schema_version = "1.0.0",
    source = list(
      package = "lme4",
      dataset = "sleepstudy",
      citation = "Belenky et al. (2003), Patterns of performance degradation and restoration during sleep restriction and subsequent recovery: a sleep dose-response study",
      license = "GPL (>= 2), following lme4 2.0-6 package metadata",
      modification = "Converted to JSON and augmented with independent-term fixed-theta, fit, covariance, diagnostic, and prediction outputs"
    ),
    reference = list(
      profile = "lme4-2.0.6-unstructured-gaussian-v1",
      lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
    ),
    data = list(
      row_ids = sprintf("sleepstudy-%03d", seq_len(nrow(sleepstudy))),
      response_name = "Reaction",
      response = scalar(sleepstudy$Reaction),
      predictor_name = "Days",
      predictor = scalar(sleepstudy$Days),
      group_name = "Subject",
      groups = unname(as.character(sleepstudy$Subject)),
      group_levels = unname(levels),
      group_indices = unname(as.integer(sleepstudy$Subject) - 1L),
      fixed_names = I(unname(colnames(parsed$X))),
      random_coefficient_names = I(c("(Intercept)", "Days")),
      covariance_term_sizes = I(c(1L, 1L)),
      random_names = unname(random_names),
      X = unname(as.matrix(parsed$X)),
      random_design = unname(cbind(1.0, sleepstudy$Days)),
      Z_group_major_sha256 = matrix_sha256(z_term_major[, permutation])
    ),
    fixed_theta_cases = fixed_cases,
    fits = lapply(c(FALSE, TRUE), make_sleepstudy_independent_fit),
    synthetic_boundary = list(
      data = list(
        y = scalar(data$y),
        x = scalar(data$x),
        groups = unname(as.character(data$g)),
        group_levels = unname(levels(data$g)),
        weights = scalar(data$w),
        offset = scalar(data$o)
      ),
      fits = lapply(c(FALSE, TRUE), make_independent_boundary_fit)
    )
  )
}

sleepstudy_independent_fixture <- make_sleepstudy_independent_fixture()

model_frame_data <- data
model_frame_data$a <- c(
  -0.03, 0.04, 0.02, -0.01, 0.05, -0.02,
  0.01, 0.03, -0.04, 0.02, -0.01, 0.04
)

make_model_frame_fit <- function(reml, contrast) {
  contrast_name <- if (contrast == "treatment") "contr.treatment" else "contr.sum"
  model <- lmer(
    y ~ x + f + offset(o) + (1 | g),
    data = model_frame_data,
    weights = w,
    offset = a,
    contrasts = list(f = contrast_name),
    REML = reml,
    na.action = na.fail,
    control = sleepstudy_control()
  )
  existing <- data.frame(
    x = c(-1.0, 0.25, 1.2),
    f = factor(c("middle", "high", "low"), levels = levels(model_frame_data$f)),
    g = factor(c("gamma", "alpha", "beta"), levels = levels(model_frame_data$g)),
    o = c(0.05, -0.10, 0.20),
    a = c(-0.02, 0.03, 0.01)
  )
  unseen <- data.frame(
    x = 0.75,
    f = factor("high", levels = levels(model_frame_data$f)),
    g = factor("new", levels = c(levels(model_frame_data$g), "new")),
    o = -0.05,
    a = 0.04
  )
  optimizer <- model@optinfo
  convergence_message <- optimizer$conv$lme4$messages
  if (is.null(convergence_message)) convergence_message <- character()
  list(
    id = sprintf("model_frame_%s_%s", contrast, if (reml) "reml" else "ml"),
    kind = if (reml) "reml" else "ml",
    contrast = contrast,
    formula = "y ~ 1 + x + f + offset(o) + (1 | g)",
    fixed_names = unname(names(fixef(model))),
    X = unname(as.matrix(getME(model, "X"))),
    X_sha256 = matrix_sha256(getME(model, "X")),
    objective = scalar(-2 * logLik(model, REML = reml)),
    log_likelihood = scalar(logLik(model, REML = reml)),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    sigma = scalar(sigma(model)),
    random_covariance = unname(as.matrix(VarCorr(model)$g)),
    beta_covariance = unname(as.matrix(vcov(model))),
    u = scalar(getME(model, "u")),
    b = scalar(getME(model, "b")),
    fitted = scalar(fitted(model)),
    residuals = scalar(residuals(model)),
    predictions = list(
      existing = list(
        x = scalar(existing$x),
        f = unname(as.character(existing$f)),
        g = unname(as.character(existing$g)),
        formula_offset = scalar(existing$o),
        argument_offset = scalar(existing$a),
        population = scalar(
          predict(model, newdata = existing, re.form = NA) + existing$a
        ),
        conditional = scalar(
          predict(model, newdata = existing, re.form = NULL) + existing$a
        )
      ),
      unseen = list(
        x = scalar(unseen$x),
        f = unname(as.character(unseen$f)),
        g = unname(as.character(unseen$g)),
        formula_offset = scalar(unseen$o),
        argument_offset = scalar(unseen$a),
        population = scalar(
          predict(model, newdata = unseen, re.form = NA, allow.new.levels = TRUE) + unseen$a
        ),
        conditional = scalar(
          predict(model, newdata = unseen, re.form = NULL, allow.new.levels = TRUE) + unseen$a
        )
      )
    ),
    evaluations = unname(as.integer(optimizer$feval)),
    convergence_code = unname(as.integer(optimizer$conv$opt)),
    convergence_messages = I(unname(as.character(convergence_message)))
  )
}

missing_model_frame <- model_frame_data
missing_model_frame[c("r02", "r03"), c("x", "f", "w", "a")] <- NA
missing_model_frame[c("r06", "r03"), c("y", "g", "o")] <- NA
model_frame_subset <- !rownames(missing_model_frame) %in% "r03"
parsed_missing_model_frame <- lFormula(
  y ~ x + f + offset(o) + (1 | g),
  data = missing_model_frame,
  weights = w,
  offset = a,
  subset = model_frame_subset,
  contrasts = list(f = "contr.treatment"),
  REML = FALSE,
  na.action = na.omit
)

model_frame_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    dataset = "Kamino model-frame synthetic v1",
    provenance = "embedded deterministic literals in oracle/generate_fixtures.R",
    license = "MIT"
  ),
  reference = list(
    profile = "lme4-2.0.6-unstructured-gaussian-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
  ),
  data = list(
    row_ids = unname(rownames(model_frame_data)),
    response = scalar(model_frame_data$y),
    predictor = scalar(model_frame_data$x),
    fixed_factor = unname(as.character(model_frame_data$f)),
    fixed_factor_levels = unname(levels(model_frame_data$f)),
    groups = unname(as.character(model_frame_data$g)),
    group_levels = unname(levels(model_frame_data$g)),
    weights = scalar(model_frame_data$w),
    formula_offset = scalar(model_frame_data$o),
    argument_offset = scalar(model_frame_data$a)
  ),
  shared_frame = list(
    subset = unname(model_frame_subset),
    retained_row_ids = unname(rownames(parsed_missing_model_frame$fr)),
    fixed_names = unname(colnames(parsed_missing_model_frame$X)),
    X = unname(as.matrix(parsed_missing_model_frame$X)),
    X_sha256 = matrix_sha256(parsed_missing_model_frame$X),
    groups = unname(as.character(parsed_missing_model_frame$fr$g)),
    weights = scalar(model.weights(parsed_missing_model_frame$fr)),
    offset = scalar(model.offset(parsed_missing_model_frame$fr))
  ),
  fits = unlist(
    lapply(
      c("treatment", "sum"),
      function(contrast) lapply(c(FALSE, TRUE), make_model_frame_fit, contrast = contrast)
    ),
    recursive = FALSE
  )
)

make_formula_case <- function(id, formula, frame = data) {
  parsed <- lFormula(formula, data = frame, REML = FALSE, na.action = na.omit)
  random_terms <- lapply(seq_along(parsed$reTrms$cnms), function(index) {
    grouping_name <- names(parsed$reTrms$cnms)[[index]]
    list(
      grouping = grouping_name,
      columns = unname(parsed$reTrms$cnms[[index]]),
      levels = unname(levels(parsed$reTrms$flist[[grouping_name]]))
    )
  })
  list(
    id = id,
    formula = paste(deparse(formula), collapse = ""),
    row_ids = unname(rownames(parsed$fr)),
    fixed_names = unname(colnames(parsed$X)),
    X = unname(as.matrix(parsed$X)),
    Z = unname(t(as.matrix(parsed$reTrms$Zt))),
    X_sha256 = matrix_sha256(parsed$X),
    Z_sha256 = matrix_sha256(t(as.matrix(parsed$reTrms$Zt))),
    random_terms = random_terms
  )
}

missing_data <- data
missing_data["r03", "x"] <- NA_real_
formula_cases <- list(
  make_formula_case("numeric_random_slope", y ~ x + (1 + x | g)),
  make_formula_case("categorical_fixed", y ~ x + f + (1 | g)),
  make_formula_case("fixed_interaction", y ~ x * f + (1 | g)),
  make_formula_case("double_bar", y ~ x + (1 + x || g)),
  make_formula_case("nested", y ~ x + (1 | g / h)),
  make_formula_case("missing_shared_row", y ~ x + f + (1 + x | g), missing_data)
)

session <- list(
  r_version = R.version.string,
  platform = R.version$platform,
  packages = as.list(vapply(
    c("lme4", "Matrix", "reformulas", "nloptr", "minqa", "Rcpp", "RcppEigen", "jsonlite"),
    function(package) as.character(packageVersion(package)),
    character(1)
  )),
  nlopt_version = system("dpkg-query -W -f='${Version}' libnlopt0", intern = TRUE),
  blas = sessionInfo()$BLAS,
  lapack = sessionInfo()$LAPACK,
  threads = list(
    OPENBLAS_NUM_THREADS = Sys.getenv("OPENBLAS_NUM_THREADS"),
    OMP_NUM_THREADS = Sys.getenv("OMP_NUM_THREADS")
  )
)

write_json(
  list(schema_version = "1.0.0", session = session, cases = cases),
  file.path(output_dir, "fixed_theta.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  list(schema_version = "1.0.0", session = session, cases = formula_cases),
  file.path(output_dir, "formula_matrices.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(session, file.path(output_dir, "session.json"), auto_unbox = TRUE, pretty = TRUE)
write_json(
  list(schema_version = "1.0.0", session = session, fits = fits),
  file.path(output_dir, "optimized.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  dyestuff,
  file.path(output_dir, "dyestuff.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  dyestuff2,
  file.path(output_dir, "dyestuff2.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  sleepstudy_fixture,
  file.path(output_dir, "sleepstudy.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  sleepstudy_independent_fixture,
  file.path(output_dir, "sleepstudy_independent.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  model_frame_fixture,
  file.path(output_dir, "model_frame.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
