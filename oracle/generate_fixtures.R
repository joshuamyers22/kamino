#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
  library(emmeans)
  library(clubSandwich)
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

general_sparse_control <- function() {
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

make_general_sparse_fixed_case <- function(frame, formula, reml, case_name, theta) {
  parsed <- lFormula(formula, data = frame, REML = reml, na.action = na.fail)
  devfun <- do.call(mkLmerDevfun, parsed)
  objective <- devfun(theta)
  state <- environment(devfun)
  lambdat <- parsed$reTrms$Lambdat
  lambdat@x <- theta[parsed$reTrms$Lind]
  lambda <- t(as.matrix(lambdat))
  u <- scalar(state$pp$u(1))
  list(
    id = sprintf("%s_%s", if (reml) "reml" else "ml", case_name),
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

make_general_sparse_fit <- function(frame, formula, reml, existing, unseen) {
  model <- lmer(
    formula,
    data = frame,
    REML = reml,
    na.action = na.fail,
    control = general_sparse_control()
  )
  optimizer <- model@optinfo
  convergence_message <- optimizer$conv$lme4$messages
  if (is.null(convergence_message)) convergence_message <- character()
  list(
    kind = if (reml) "reml" else "ml",
    objective = scalar(-2 * logLik(model, REML = reml)),
    log_likelihood = scalar(logLik(model, REML = reml)),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    sigma = scalar(sigma(model)),
    random_covariances = unname(lapply(VarCorr(model), function(value) unname(as.matrix(value)))),
    beta_covariance = unname(as.matrix(vcov(model))),
    u = scalar(getME(model, "u")),
    b = scalar(getME(model, "b")),
    fitted = scalar(fitted(model)),
    residuals = scalar(residuals(model)),
    predictions = list(
      training_population = scalar(predict(model, re.form = NA)),
      training_conditional = scalar(predict(model, re.form = NULL)),
      existing_population = scalar(predict(model, newdata = existing, re.form = NA)),
      existing_conditional = scalar(predict(model, newdata = existing, re.form = NULL)),
      unseen_population = scalar(predict(model, newdata = unseen, re.form = NA, allow.new.levels = TRUE)),
      unseen_conditional = scalar(predict(model, newdata = unseen, re.form = NULL, allow.new.levels = TRUE))
    ),
    evaluations = unname(as.integer(optimizer$feval)),
    convergence_code = unname(as.integer(optimizer$conv$opt)),
    convergence_messages = I(unname(as.character(convergence_message)))
  )
}

make_general_sparse_fixture <- function(
  frame, formula, dataset_name, response_name, group_columns, existing, unseen,
  citation
) {
  parsed <- lFormula(formula, data = frame, REML = TRUE, na.action = na.fail)
  theta_cases <- list(interior = rep(0.7, length(parsed$reTrms$theta)), partial_zero = c(0.0, rep(0.5, length(parsed$reTrms$theta) - 1L)), zero = rep(0.0, length(parsed$reTrms$theta)))
  fixed_cases <- list()
  for (reml in c(FALSE, TRUE)) {
    for (case_name in names(theta_cases)) {
      fixed_cases[[length(fixed_cases) + 1L]] <- make_general_sparse_fixed_case(
        frame, formula, reml, case_name, theta_cases[[case_name]]
      )
    }
  }
  random_terms <- lapply(seq_along(parsed$reTrms$cnms), function(index) {
    grouping_name <- names(parsed$reTrms$cnms)[[index]]
    list(
      grouping = grouping_name,
      columns = unname(parsed$reTrms$cnms[[index]]),
      levels = unname(levels(parsed$reTrms$flist[[grouping_name]]))
    )
  })
  list(
    schema_version = "1.0.0",
    source = list(
      package = "lme4",
      dataset = dataset_name,
      citation = citation,
      license = "GPL (>= 2), following lme4 2.0-6 package metadata",
      modification = "Converted to JSON and augmented with sparse fixed-theta, fitted, and prediction outputs"
    ),
    reference = list(
      profile = "lme4-2.0.6-unstructured-gaussian-v1",
      lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
    ),
    formula = paste(deparse(formula), collapse = ""),
    response_name = response_name,
    data = c(
      list(
        row_ids = sprintf("%s-%03d", tolower(dataset_name), seq_len(nrow(frame))),
        response = scalar(frame[[response_name]]),
        fixed_names = unname(colnames(parsed$X)),
        random_names = unname(rownames(parsed$reTrms$Zt)),
        random_terms = random_terms,
        X = unname(as.matrix(parsed$X)),
        Z_sha256 = matrix_sha256(t(as.matrix(parsed$reTrms$Zt)))
      ),
      lapply(group_columns, function(name) unname(as.character(frame[[name]])))
    ),
    group_columns = unname(group_columns),
    existing = lapply(existing, as.character),
    unseen = lapply(unseen, as.character),
    fixed_theta_cases = fixed_cases,
    fits = lapply(
      c(FALSE, TRUE), make_general_sparse_fit,
      frame = frame, formula = formula, existing = existing, unseen = unseen
    )
  )
}

pastes_existing <- data.frame(
  batch = factor("A", levels = levels(Pastes$batch)),
  cask = factor("a", levels = levels(Pastes$cask))
)
pastes_unseen <- data.frame(
  batch = factor("A", levels = levels(Pastes$batch)),
  cask = factor("new", levels = c(levels(Pastes$cask), "new"))
)
pastes_sparse_fixture <- make_general_sparse_fixture(
  Pastes,
  strength ~ 1 + (1 | batch / cask),
  "Pastes",
  "strength",
  c(batch = "batch", cask = "cask"),
  pastes_existing,
  pastes_unseen,
  "Davies and Goldsmith (1972), Statistical Methods in Research and Production, 4th ed., section 6.4"
)

penicillin_existing <- data.frame(
  plate = factor("a", levels = levels(Penicillin$plate)),
  sample = factor("A", levels = levels(Penicillin$sample))
)
penicillin_unseen <- data.frame(
  plate = factor("new", levels = c(levels(Penicillin$plate), "new")),
  sample = factor("A", levels = levels(Penicillin$sample))
)
penicillin_sparse_fixture <- make_general_sparse_fixture(
  Penicillin,
  diameter ~ 1 + (1 | plate) + (1 | sample),
  "Penicillin",
  "diameter",
  c(plate = "plate", sample = "sample"),
  penicillin_existing,
  penicillin_unseen,
  "Davies and Goldsmith (1972), Statistical Methods in Research and Production, 4th ed., section 6.6"
)

make_insteval_sparse_fit <- function(reml) {
  model <- lmer(
    y ~ service * dept + (1 | s) + (1 | d),
    data = InstEval,
    REML = reml,
    na.action = na.fail,
    control = general_sparse_control()
  )
  list(
    kind = if (reml) "reml" else "ml",
    objective = scalar(-2 * logLik(model, REML = reml)),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    sigma = scalar(sigma(model)),
    evaluations = unname(as.integer(model@optinfo$feval))
  )
}

insteval_parsed <- lFormula(
  y ~ service * dept + (1 | s) + (1 | d),
  data = InstEval,
  REML = TRUE,
  na.action = na.fail
)
insteval_sparse_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    package = "lme4",
    dataset = "InstEval",
    citation = "Bates et al. (2015), Fitting Linear Mixed-Effects Models Using lme4",
    license = "GPL (>= 2), following lme4 2.0-6 package metadata",
    modification = "Converted selected columns to JSON and augmented with crossed sparse fit outputs"
  ),
  reference = list(
    profile = "lme4-2.0.6-unstructured-gaussian-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
  ),
  formula = "y ~ service * dept + (1 | s) + (1 | d)",
  data = list(
    response = scalar(InstEval$y),
    s = unname(as.character(InstEval$s)),
    s_levels = unname(levels(InstEval$s)),
    d = unname(as.character(InstEval$d)),
    d_levels = unname(levels(InstEval$d)),
    service = unname(as.character(InstEval$service)),
    service_levels = unname(levels(InstEval$service)),
    dept = unname(as.character(InstEval$dept)),
    dept_levels = unname(levels(InstEval$dept)),
    fixed_names = unname(colnames(insteval_parsed$X)),
    n = nrow(InstEval),
    p = ncol(insteval_parsed$X),
    q = nrow(insteval_parsed$reTrms$Zt),
    random_design_nonzeros = length(insteval_parsed$reTrms$Zt@x)
  ),
  fits = lapply(c(FALSE, TRUE), make_insteval_sparse_fit)
)

# Phase 2 F02: rank dropping, estimability, categorical random terms, and
# adversarial new-data contracts.  All inputs below are deterministic MIT data.
f02_levels <- c("a", "b", "c")
f02_group_levels <- sprintf("g%d", 1:16)
f02_f <- factor(
  rep(rep(f02_levels, each = 2), length(f02_group_levels)),
  levels = f02_levels
)
f02_g <- factor(
  rep(f02_group_levels, each = 2 * length(f02_levels)),
  levels = f02_group_levels
)
f02_h <- factor(rep(sprintf("h%d", 1:6), length.out = length(f02_f)))
f02_group_intercept <- 1.1 * sin(seq_along(f02_group_levels) * 0.7)
f02_group_b <- 0.55 * cos(seq_along(f02_group_levels) * 1.1)
f02_group_c <- 0.45 * sin(seq_along(f02_group_levels) * 1.3)
f02_g_index <- as.integer(f02_g)
f02_f_index <- as.integer(f02_f)
f02_y <- 10 + c(0, 2, -1)[f02_f_index] +
  f02_group_intercept[f02_g_index] +
  ifelse(f02_f_index == 2, f02_group_b[f02_g_index], 0) +
  ifelse(f02_f_index == 3, f02_group_c[f02_g_index], 0) +
  rep(c(0.2, -0.2), length.out = length(f02_f)) +
  c(-0.15, 0.1, -0.05, 0.1, -0.08, 0.07)[as.integer(f02_h)] +
  0.35 * sin(seq_along(f02_f) * 1.7)
f02_data <- data.frame(y = f02_y, f = f02_f, g = f02_g, h = f02_h)
rownames(f02_data) <- sprintf("f02-%03d", seq_len(nrow(f02_data)))

f02_rank_data <- transform(
  f02_data,
  x = rep(c(-1.5, -0.5, 0.5, 1.5, 0.25, -0.25), length(f02_group_levels))
)
f02_rank_data$duplicate <- 2 * f02_rank_data$x
rank_formula <- y ~ x + duplicate + (1 | g)
rank_full_X <- model.matrix(nobars(rank_formula), f02_rank_data)
rank_parsed <- suppressMessages(lFormula(
  rank_formula, data = f02_rank_data, REML = TRUE, na.action = na.fail
))
f02_near_alias_data <- f02_rank_data
f02_near_alias_data$duplicate[[17]] <- f02_near_alias_data$duplicate[[17]] + 1e-9
near_alias_full_X <- model.matrix(nobars(rank_formula), f02_near_alias_data)
near_alias_parsed <- suppressMessages(lFormula(
  rank_formula, data = f02_near_alias_data, REML = TRUE, na.action = na.fail
))

make_f02_rank_fit <- function(reml) {
  model <- suppressMessages(lmer(
    rank_formula,
    data = f02_rank_data,
    REML = reml,
    na.action = na.fail,
    control = general_sparse_control()
  ))
  newdata <- data.frame(
    x = c(-0.75, 0.4),
    duplicate = c(-1.5, 0.8),
    g = factor(c("g1", "new"), levels = c(levels(f02_rank_data$g), "new"))
  )
  list(
    kind = if (reml) "reml" else "ml",
    objective = scalar(-2 * logLik(model, REML = reml)),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    beta_covariance = unname(as.matrix(vcov(model))),
    sigma = scalar(sigma(model)),
    population = scalar(predict(model, newdata = newdata, re.form = NA, allow.new.levels = TRUE)),
    conditional = scalar(predict(model, newdata = newdata, re.form = NULL, allow.new.levels = TRUE))
  )
}

empty_combinations <- data.frame(
  a = c("A", "A", "B", "B", "C"),
  b = c("u", "v", "u", "v", "u")
)
f02_empty_data <- data.frame(
  y = rep(c(1.0, 1.4, 2.1, 2.7, 3.2), length(f02_group_levels)) + rep(seq(-0.35, 0.35, length.out = length(f02_group_levels)), each = 5),
  a = factor(rep(empty_combinations$a, length(f02_group_levels)), levels = c("A", "B", "C")),
  b = factor(rep(empty_combinations$b, length(f02_group_levels)), levels = c("u", "v")),
  g = factor(rep(f02_group_levels, each = 5), levels = f02_group_levels)
)
empty_formula <- y ~ a * b + (1 | g)
empty_full_X <- model.matrix(nobars(empty_formula), f02_empty_data)
empty_parsed <- suppressMessages(lFormula(
  empty_formula, data = f02_empty_data, REML = TRUE, na.action = na.fail
))

f02_threshold_data <- transform(
  f02_rank_data,
  tiny = c(rep(0, 17), 1e-8, rep(0, nrow(f02_rank_data) - 18))
)
threshold_formula <- y ~ x + tiny + (1 | g)
threshold_full_X <- model.matrix(nobars(threshold_formula), f02_threshold_data)
threshold_parsed <- lFormula(
  threshold_formula, data = f02_threshold_data, REML = TRUE, na.action = na.fail
)

make_f02_categorical_fixed_case <- function(
  frame, contrast, reml, fixed_contrasts = NULL,
  fixed_contrast = contrast, random_contrast = contrast
) {
  theta <- c(0.8, 0.1, -0.15, 0.5, 0.05, 0.4)
  parsed <- lFormula(
    y ~ f + (1 + f | g), data = frame, REML = reml,
    na.action = na.fail, contrasts = fixed_contrasts
  )
  devfun <- do.call(mkLmerDevfun, parsed)
  objective <- devfun(theta)
  state <- environment(devfun)
  lambdat <- parsed$reTrms$Lambdat
  lambdat@x <- theta[parsed$reTrms$Lind]
  lambda <- t(as.matrix(lambdat))
  u <- scalar(state$pp$u(1))
  list(
    id = sprintf("categorical_%s_%s_fixed", contrast, if (reml) "reml" else "ml"),
    contrast = contrast,
    fixed_contrast = fixed_contrast,
    random_contrast = random_contrast,
    kind = if (reml) "reml" else "ml",
    theta = theta,
    objective = scalar(objective),
    fixed_names = unname(colnames(parsed$X)),
    random_coefficient_names = unname(parsed$reTrms$cnms[[1]]),
    X_sha256 = matrix_sha256(parsed$X),
    Z_sha256 = matrix_sha256(t(as.matrix(parsed$reTrms$Zt))),
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

make_f02_categorical_fit <- function(
  frame, contrast, reml, fixed_contrasts = NULL,
  fixed_contrast = contrast, random_contrast = contrast
) {
  model <- lmer(
    y ~ f + (1 + f | g),
    data = frame,
    REML = reml,
    contrasts = fixed_contrasts,
    na.action = na.fail,
    control = general_sparse_control()
  )
  existing <- data.frame(
    f = factor(c("a", "b", "c"), levels = levels(frame$f)),
    g = factor(c("g1", "g3", "g8"), levels = levels(frame$g))
  )
  contrasts(existing$f) <- contrasts(frame$f)
  unseen <- data.frame(
    f = factor(c("a", "c"), levels = levels(frame$f)),
    g = factor(c("new", "g2"), levels = c(levels(frame$g), "new"))
  )
  contrasts(unseen$f) <- contrasts(frame$f)
  list(
    id = sprintf("categorical_%s_%s_fit", contrast, if (reml) "reml" else "ml"),
    contrast = contrast,
    fixed_contrast = fixed_contrast,
    random_contrast = random_contrast,
    kind = if (reml) "reml" else "ml",
    objective = scalar(-2 * logLik(model, REML = reml)),
    theta = scalar(getME(model, "theta")),
    beta = scalar(fixef(model)),
    beta_covariance = unname(as.matrix(vcov(model))),
    sigma = scalar(sigma(model)),
    random_covariance = unname(as.matrix(VarCorr(model)$g)),
    u = scalar(getME(model, "u")),
    b = scalar(getME(model, "b")),
    fitted = scalar(fitted(model)),
    residuals = scalar(residuals(model)),
    predictions = list(
      existing_population = scalar(predict(model, newdata = existing, re.form = NA)),
      existing_conditional = scalar(predict(model, newdata = existing, re.form = NULL)),
      unseen_population = scalar(predict(model, newdata = unseen, re.form = NA, allow.new.levels = TRUE)),
      unseen_conditional = scalar(predict(model, newdata = unseen, re.form = NULL, allow.new.levels = TRUE))
    ),
    evaluations = unname(as.integer(model@optinfo$feval))
  )
}

f02_categorical_cases <- list()
f02_categorical_fits <- list()
for (contrast in c("treatment", "sum")) {
  frame <- f02_data
  if (contrast == "sum") contrasts(frame$f) <- contr.sum(3)
  for (reml in c(FALSE, TRUE)) {
    f02_categorical_cases[[length(f02_categorical_cases) + 1L]] <-
      make_f02_categorical_fixed_case(frame, contrast, reml)
    f02_categorical_fits[[length(f02_categorical_fits) + 1L]] <-
      make_f02_categorical_fit(frame, contrast, reml)
  }
}
for (reml in c(FALSE, TRUE)) {
  f02_categorical_cases[[length(f02_categorical_cases) + 1L]] <-
    make_f02_categorical_fixed_case(
      f02_data, "fixed_sum_random_treatment", reml,
      fixed_contrasts = list(f = "contr.sum"),
      fixed_contrast = "sum", random_contrast = "treatment"
    )
  f02_categorical_fits[[length(f02_categorical_fits) + 1L]] <-
    make_f02_categorical_fit(
      f02_data, "fixed_sum_random_treatment", reml,
      fixed_contrasts = list(f = "contr.sum"),
      fixed_contrast = "sum", random_contrast = "treatment"
    )
}

f02_crossed_formula <- y ~ f + (1 + f | g) + (1 | h)
f02_crossed_parsed <- lFormula(
  f02_crossed_formula, data = f02_data, REML = TRUE, na.action = na.fail
)
f02_crossed_theta <- c(0.8, 0.1, -0.15, 0.5, 0.05, 0.4, 0.6)
f02_crossed_fixed <- lapply(c(FALSE, TRUE), function(reml) {
  make_general_sparse_fixed_case(
    f02_data, f02_crossed_formula, reml, "categorical_crossed", f02_crossed_theta
  )
})
f02_crossed_existing <- data.frame(
  f = factor("b", levels = levels(f02_data$f)),
  g = factor("g2", levels = levels(f02_data$g)),
  h = factor("h3", levels = levels(f02_data$h))
)
f02_crossed_unseen <- data.frame(
  f = factor("c", levels = levels(f02_data$f)),
  g = factor("new", levels = c(levels(f02_data$g), "new")),
  h = factor("h1", levels = levels(f02_data$h))
)
f02_crossed_fits <- lapply(c(FALSE, TRUE), make_general_sparse_fit,
  frame = f02_data,
  formula = f02_crossed_formula,
  existing = f02_crossed_existing,
  unseen = f02_crossed_unseen
)

f02_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    dataset = "Kamino F02 synthetic v1",
    provenance = "embedded deterministic literals in oracle/generate_fixtures.R",
    license = "MIT"
  ),
  reference = list(
    profile = "lme4-2.0.6-unstructured-gaussian-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
  ),
  data = list(
    row_ids = unname(rownames(f02_data)),
    response = scalar(f02_data$y),
    fixed_factor = unname(as.character(f02_data$f)),
    fixed_factor_levels = unname(levels(f02_data$f)),
    groups = unname(as.character(f02_data$g)),
    group_levels = unname(levels(f02_data$g)),
    second_groups = unname(as.character(f02_data$h)),
    second_group_levels = unname(levels(f02_data$h))
  ),
  rank_cases = list(
    exact_alias = list(
      formula = "y ~ x + duplicate + (1 | g)",
      x = scalar(f02_rank_data$x),
      duplicate = scalar(f02_rank_data$duplicate),
      full_names = unname(colnames(rank_full_X)),
      full_X_sha256 = matrix_sha256(rank_full_X),
      retained_names = unname(colnames(rank_parsed$X)),
      retained_X_sha256 = matrix_sha256(rank_parsed$X),
      dropped_indices_one_based = unname(as.integer(attr(rank_parsed$X, "col.dropped"))),
      qr_pivot_one_based = unname(qr(rank_full_X, tol = 1e-7, LAPACK = FALSE)$pivot),
      fits = lapply(c(FALSE, TRUE), make_f02_rank_fit)
    ),
    near_alias = list(
      formula = "y ~ x + duplicate + (1 | g)",
      x = scalar(f02_near_alias_data$x),
      duplicate = scalar(f02_near_alias_data$duplicate),
      full_names = unname(colnames(near_alias_full_X)),
      retained_names = unname(colnames(near_alias_parsed$X)),
      dropped_indices_one_based = unname(as.integer(attr(near_alias_parsed$X, "col.dropped"))),
      qr_pivot_one_based = unname(qr(near_alias_full_X, tol = 1e-7, LAPACK = FALSE)$pivot)
    ),
    empty_interaction = list(
      formula = "y ~ a * b + (1 | g)",
      a = unname(as.character(f02_empty_data$a)),
      a_levels = unname(levels(f02_empty_data$a)),
      b = unname(as.character(f02_empty_data$b)),
      b_levels = unname(levels(f02_empty_data$b)),
      groups = unname(as.character(f02_empty_data$g)),
      response = scalar(f02_empty_data$y),
      full_names = unname(colnames(empty_full_X)),
      full_X_sha256 = matrix_sha256(empty_full_X),
      retained_names = unname(colnames(empty_parsed$X)),
      dropped_indices_one_based = unname(as.integer(attr(empty_parsed$X, "col.dropped"))),
      qr_pivot_one_based = unname(qr(empty_full_X, tol = 1e-7, LAPACK = FALSE)$pivot)
    ),
    threshold_sensitive = list(
      formula = "y ~ x + tiny + (1 | g)",
      tiny = scalar(f02_threshold_data$tiny),
      full_names = unname(colnames(threshold_full_X)),
      retained_names = unname(colnames(threshold_parsed$X)),
      qr_pivot_one_based = unname(qr(threshold_full_X, tol = 1e-7, LAPACK = FALSE)$pivot)
    )
  ),
  categorical_fixed_theta_cases = f02_categorical_cases,
  categorical_fits = f02_categorical_fits,
  crossed = list(
    formula = "y ~ f + (1 + f | g) + (1 | h)",
    random_terms = lapply(seq_along(f02_crossed_parsed$reTrms$cnms), function(index) {
      grouping_name <- names(f02_crossed_parsed$reTrms$cnms)[[index]]
      list(
        grouping = grouping_name,
        columns = unname(f02_crossed_parsed$reTrms$cnms[[index]]),
        levels = unname(levels(f02_crossed_parsed$reTrms$flist[[grouping_name]]))
      )
    }),
    X_sha256 = matrix_sha256(f02_crossed_parsed$X),
    Z_sha256 = matrix_sha256(t(as.matrix(f02_crossed_parsed$reTrms$Zt))),
    fixed_theta_cases = f02_crossed_fixed,
    fits = f02_crossed_fits,
    existing = list(f = "b", g = "g2", h = "h3"),
    unseen = list(f = "c", g = "new", h = "h1")
  )
)

i01_response_cases <- list(
  list(
    id = "dyestuff_ml_unconditional", kind = "ml", mode = "unconditional",
    root_seed = "7101", replicate_id = 0L,
    response_sha256 = "a268c8ad3fa370865fcc1ccf1e939edb0154c094ef73f11130e40a51ff83578a",
    response = c(
      1442.594843924611, 1450.4033038798518, 1527.8128747944618,
      1513.8645163907188, 1422.3562030127323, 1467.8746965518042,
      1443.5735589126725, 1576.149801059817, 1508.237330082996,
      1596.7116298144856, 1555.545141380125, 1513.0750640934095,
      1512.809053657081, 1512.9960881776835, 1466.0994733006041,
      1562.2776542857014, 1521.827293737131, 1582.4653241038052,
      1483.302119215473, 1460.446444304666, 1492.470303139595,
      1487.3828544872001, 1479.6032881999138, 1559.1888267807096,
      1566.028882775842, 1577.314557688519, 1629.9328892855574,
      1627.917067393582, 1626.153665120197, 1618.8462226824913
    )
  ),
  list(
    id = "dyestuff_ml_conditional", kind = "ml", mode = "conditional",
    root_seed = "7102", replicate_id = 0L,
    response_sha256 = "7a14be6aaa5ca1a688510c8bca291d952cf2b2e6159f3ec847e978a11b96a81c",
    response = c(
      1466.2335878581384, 1557.4707241843466, 1559.4593686427688,
      1467.3865516354886, 1595.6321349694788, 1517.3279723931778,
      1499.201808919047, 1512.939033943397, 1583.0709323069998,
      1598.4376463740414, 1541.6352046074949, 1564.1286895418282,
      1646.5778804602107, 1578.1631035813934, 1471.5746022031326,
      1585.2818719888614, 1489.6478228615324, 1557.9743413275896,
      1445.0140678811815, 1463.9665095794303, 1591.8572285391476,
      1556.8287867741292, 1523.5122045608312, 1624.1200905951612,
      1563.0819519824743, 1458.1311113253962, 1519.2545299626195,
      1417.1279313299565, 1546.3570455650836, 1453.9667452717651
    )
  ),
  list(
    id = "dyestuff_reml_unconditional", kind = "reml", mode = "unconditional",
    root_seed = "7201", replicate_id = 0L,
    response_sha256 = "cc20b87d1750440c99a2f934d0496a9432e605f7e52c9e5ea5acecf17c85cfcf",
    response = c(
      1611.832185427145, 1516.1441202965893, 1495.642798564921,
      1666.4067368214455, 1570.3665117558246, 1566.568632012124,
      1526.5780259706673, 1496.1561899865885, 1586.2776657163095,
      1590.4315741233931, 1503.4663828360149, 1451.7955918518448,
      1471.6601873506734, 1569.431941577809, 1588.3691913154403,
      1572.7162499991798, 1584.5850307952755, 1474.9530186358259,
      1533.7893500049695, 1537.2611992885672, 1352.0947505568843,
      1483.8516610622676, 1522.0327289141414, 1488.68212416446,
      1504.7345282730616, 1617.2207926088167, 1517.060912644361,
      1507.6399777017466, 1603.6213590687325, 1581.962785197196
    )
  ),
  list(
    id = "dyestuff_reml_conditional", kind = "reml", mode = "conditional",
    root_seed = "7202", replicate_id = 0L,
    response_sha256 = "88f6fca90f01e7fdbcd6b30d134a3313fe42c5018f0a8918f1ecd90c3515a6b9",
    response = c(
      1500.3790084985008, 1565.2386049034744, 1507.2184513164602,
      1501.559193604452, 1511.3854412971125, 1507.3728922326927,
      1513.3031457113757, 1620.7482269424725, 1553.3415590007367,
      1545.8382484465678, 1567.5689698800568, 1627.3437995298807,
      1531.5447258364347, 1551.9680542781375, 1528.1107316970317,
      1530.29587063117, 1441.2774884188264, 1489.9340383549093,
      1518.113956468119, 1479.651469086796, 1573.5469543562244,
      1565.2358048116248, 1538.0892888382216, 1623.8390640629548,
      1622.890832958173, 1441.7085662220381, 1472.8537968835283,
      1455.6061610298896, 1405.3641094404875, 1527.0144609085614
    )
  )
)

i01_cases <- lapply(i01_response_cases, function(case) {
  frame <- Dyestuff
  frame$Yield <- case$response
  fit <- make_dyestuff_fit(case$kind == "reml", frame, case$id)
  list(
    id = case$id,
    kind = case$kind,
    mode = case$mode,
    root_seed = case$root_seed,
    replicate_id = case$replicate_id,
    response_sha256 = case$response_sha256,
    response = scalar(case$response),
    fit = list(
      objective = fit$objective,
      log_likelihood = fit$log_likelihood,
      theta = fit$theta,
      beta = fit$beta,
      sigma = fit$sigma,
      random_variance = fit$random_variance,
      boundary = isSingular(
        lmer(
          Yield ~ 1 + (1 | Batch), data = frame,
          REML = case$kind == "reml"
        ),
        tol = 1e-4
      )
    )
  )
})

i01_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    dataset = "Dyestuff responses generated by Kamino PCG64DXSM simulation v1",
    provenance = "deterministic literals in oracle/generate_fixtures.R",
    license = "GPL (>= 2), following lme4 2.0-6 Dyestuff metadata"
  ),
  reference = list(
    profile = "lme4-2.0.6-unstructured-gaussian-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
  ),
  rng = list(
    bit_generator = "PCG64DXSM",
    numpy_version_scope = "recorded by each Kamino ledger; cross-version identity is not claimed",
    stream_key = "SeedSequence(root_seed, replicate_id, purpose)"
  ),
  group_levels = unname(levels(Dyestuff$Batch)),
  groups = unname(as.character(Dyestuff$Batch)),
  cases = i01_cases
)

make_i02_case <- function(dataset, reml) {
  if (dataset == "Dyestuff") {
    model <- lmerTest::lmer(
      Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = reml,
      control = lmerControl(
        optimizer = "nloptwrap", restart_edge = TRUE, boundary.tol = 1e-5,
        optCtrl = list(
          algorithm = "NLOPT_LN_BOBYQA", xtol_abs = 1e-8,
          ftol_abs = 1e-8, maxeval = 100000
        )
      )
    )
    one_contrast <- c(1.0)
    joint_contrast <- matrix(1.0, nrow = 1L)
  } else {
    model <- lmerTest::lmer(
      Reaction ~ Days + (1 + Days | Subject), data = sleepstudy, REML = reml,
      control = lmerControl(
        optimizer = "nloptwrap", restart_edge = TRUE, boundary.tol = 1e-5,
        optCtrl = list(
          algorithm = "NLOPT_LN_BOBYQA", xtol_abs = 1e-8,
          ftol_abs = 1e-8, maxeval = 100000
        )
      )
    )
    one_contrast <- c(0.0, 1.0)
    joint_contrast <- diag(2L)
  }
  converted <- model
  one <- lmerTest::contest1D(converted, one_contrast, rhs = 0.0)
  joint <- lmerTest::contestMD(converted, joint_contrast, rhs = 0.0)
  variance_parameter_covariance <- unname(converted@vcov_varpar)
  list(
    id = sprintf(
      "%s_%s", tolower(dataset), if (reml) "reml" else "ml"
    ),
    dataset = dataset,
    kind = if (reml) "reml" else "ml",
    fixed_names = unname(names(fixef(model))),
    eta = scalar(c(getME(model, "theta"), sigma(model))),
    beta_covariance = unname(converted@vcov_beta),
    variance_parameter_hessian = unname(2.0 * solve(variance_parameter_covariance)),
    variance_parameter_covariance = variance_parameter_covariance,
    beta_covariance_jacobian = lapply(converted@Jac_list, unname),
    one_df = list(
      contrast = scalar(one_contrast), rhs = 0.0,
      estimate = scalar(one[["Estimate"]]),
      standard_error = scalar(one[["Std. Error"]]),
      denominator_df = scalar(one[["df"]]),
      statistic = scalar(one[["t value"]]),
      p_value = scalar(one[["Pr(>|t|)"]])
    ),
    joint = list(
      contrast = unname(joint_contrast), rhs = rep(0.0, nrow(joint_contrast)),
      numerator_df = unname(as.integer(joint[["NumDF"]])),
      denominator_df = scalar(joint[["DenDF"]]),
      statistic = scalar(joint[["F value"]]),
      p_value = scalar(joint[["Pr(>F)"]])
    )
  )
}

i02_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    method = "lmerTest Satterthwaite contests and derivative slots",
    datasets = c("lme4 Dyestuff", "lme4 sleepstudy"),
    license = "GPL (>= 2), following lme4 and lmerTest package metadata"
  ),
  reference = list(
    profile = "lme4-2.0.6-lmerTest-3.1-3-satterthwaite-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e",
    lmerTest_source_commit = "35dc5885205d709cdc395b369b08ca2b7273cb78"
  ),
  cases = unlist(
    lapply(c("Dyestuff", "sleepstudy"), function(dataset) {
      lapply(c(FALSE, TRUE), function(reml) make_i02_case(dataset, reml))
    }),
    recursive = FALSE
  )
)

kr_test <- function(model, contrast, rhs) {
  beta_h <- scalar(MASS::ginv(contrast) %*% rhs)
  result <- pbkrtest::KRmodcomp(model, contrast, betaH = beta_h)
  adjusted <- result$test["Ftest", ]
  unscaled <- result$test["FtestU", ]
  list(
    contrast = unname(contrast),
    rhs = scalar(rhs),
    numerator_df = unname(as.integer(adjusted[["ndf"]])),
    denominator_df = scalar(adjusted[["ddf"]]),
    statistic = scalar(adjusted[["stat"]]),
    scaling = scalar(adjusted[["F.scaling"]]),
    p_value = scalar(adjusted[["p.value"]]),
    unscaled_statistic = scalar(unscaled[["stat"]]),
    unscaled_p_value = scalar(unscaled[["p.value"]]),
    auxiliary = unname(as.list(result$aux))
  )
}

make_i03_kr_case <- function(dataset, reml) {
  if (dataset == "Dyestuff") {
    model <- lmer(
      Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = reml,
      control = sleepstudy_control()
    )
    one <- matrix(1.0, nrow = 1L)
    joint <- one
  } else {
    model <- lmer(
      Reaction ~ Days + (1 + Days | Subject), data = sleepstudy, REML = reml,
      control = sleepstudy_control()
    )
    one <- matrix(c(0.0, 1.0), nrow = 1L)
    joint <- diag(2L)
  }
  reml_model <- if (reml) model else update(model, . ~ ., REML = TRUE)
  adjusted <- pbkrtest::vcovAdj(reml_model)
  list(
    id = sprintf("%s_%s", tolower(dataset), if (reml) "reml" else "ml"),
    dataset = dataset,
    input_kind = if (reml) "reml" else "ml",
    analysis_kind = "reml",
    reml_refit = !reml,
    fixed_names = unname(names(fixef(reml_model))),
    reml_objective = scalar(-2 * logLik(reml_model, REML = TRUE)),
    reml_theta = scalar(getME(reml_model, "theta")),
    reml_sigma = scalar(sigma(reml_model)),
    covariance = unname(as.matrix(vcov(reml_model))),
    adjusted_covariance = unname(as.matrix(adjusted)),
    covariance_parameter_information = unname(as.matrix(2.0 * solve(attr(adjusted, "W")))),
    covariance_parameter_covariance = unname(as.matrix(attr(adjusted, "W"))),
    information_minimum_absolute_eigenvalue = scalar(attr(adjusted, "condi")),
    derivative_matrices = lapply(attr(adjusted, "P"), function(value) unname(as.matrix(value))),
    one_df = kr_test(model, one, 0.0),
    joint = kr_test(model, joint, rep(0.0, nrow(joint)))
  )
}

profile_trace <- function(profiled, target, level = 0.95) {
  rows <- profiled[as.character(profiled$.par) == target, , drop = FALSE]
  parameter_names <- setdiff(colnames(rows), c(".zeta", ".par"))
  interval <- suppressWarnings(confint(profiled, parm = target, level = level))
  list(
    target = target,
    parameter_names = parameter_names,
    points = lapply(seq_len(nrow(rows)), function(index) {
      list(
        signed_root_deviance = scalar(rows$.zeta[index]),
        parameters = scalar(rows[index, parameter_names, drop = TRUE])
      )
    }),
    interval_level = level,
    interval = scalar(interval[1L, ])
  )
}

make_i03_profile_case <- function(dataset, reml) {
  if (dataset == "Dyestuff") {
    model <- lmer(
      Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = reml,
      control = sleepstudy_control()
    )
    targets <- c(".sig01", ".sigma", "(Intercept)")
  } else {
    model <- lmer(
      Reaction ~ Days + (1 + Days | Subject), data = sleepstudy, REML = reml,
      control = sleepstudy_control()
    )
    targets <- c(".sig01", ".sig02", ".sig03", ".sigma", "Days")
  }
  baseline <- if (reml) refitML(model) else model
  profiled <- suppressWarnings(profile(
    model, which = targets, alphamax = 0.01, maxpts = 30,
    delta = 0.4, devtol = 1e-8, devmatchtol = 1e-5,
    signames = TRUE, prof.scale = "sdcor"
  ))
  list(
    id = sprintf("%s_%s", tolower(dataset), if (reml) "reml" else "ml"),
    dataset = dataset,
    input_kind = if (reml) "reml" else "ml",
    baseline_kind = "ml",
    ml_refit = reml,
    fixed_names = unname(names(fixef(baseline))),
    target_order = targets,
    baseline_objective = scalar(-2 * logLik(baseline, REML = FALSE)),
    baseline_theta = scalar(getME(baseline, "theta")),
    baseline_sigma = scalar(sigma(baseline)),
    traces = lapply(targets, function(target) profile_trace(profiled, target))
  )
}

i03_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    method = "pbkrtest Kenward-Roger and lme4 likelihood profiles",
    datasets = c("lme4 Dyestuff", "lme4 sleepstudy"),
    license = "GPL (>= 2), following lme4 and pbkrtest package metadata"
  ),
  reference = list(
    profile = "lme4-2.0-6-pbkrtest-0.5.5-i03-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e",
    pbkrtest_source_commit = "4ead4ff46e90831f4a2376cdcae0602045db0458"
  ),
  kenward_roger = unlist(
    lapply(c("Dyestuff", "sleepstudy"), function(dataset) {
      lapply(c(FALSE, TRUE), function(reml) make_i03_kr_case(dataset, reml))
    }), recursive = FALSE
  ),
  profiles = unlist(
    lapply(c("Dyestuff", "sleepstudy"), function(dataset) {
      lapply(c(FALSE, TRUE), function(reml) make_i03_profile_case(dataset, reml))
    }), recursive = FALSE
  )
)

make_a01_frame <- function() {
  cells <- list(
    c("a", "u", 8L),
    c("a", "v", 4L),
    c("b", "u", 3L),
    c("b", "v", 9L)
  )
  group_effects <- c(-0.7, 0.4, 0.9, -0.2, 0.3, -0.5)
  residual_pattern <- c(-0.2, 0.1, 0.15, -0.05)
  rows <- list()
  cursor <- 0L
  for (cell in cells) {
    f_value <- cell[[1L]]
    h_value <- cell[[2L]]
    count <- as.integer(cell[[3L]])
    for (cell_index in seq_len(count)) {
      x <- -1.5 + 3.0 * ((cursor %% 7L) / 6.0)
      group_index <- cursor %% length(group_effects) + 1L
      mean <- 10.0 +
        ifelse(f_value == "b", 1.7, 0.0) +
        ifelse(h_value == "v", -0.8, 0.0) +
        ifelse(f_value == "b" && h_value == "v", 1.1, 0.0) +
        0.6 * x + group_effects[[group_index]]
      rows[[length(rows) + 1L]] <- data.frame(
        y = mean + residual_pattern[[(cell_index - 1L) %% 4L + 1L]],
        x = x,
        f = f_value,
        h = h_value,
        g = sprintf("g%d", group_index),
        stringsAsFactors = FALSE
      )
      cursor <- cursor + 1L
    }
  }
  result <- do.call(rbind, rows)
  result$f <- factor(result$f, levels = c("a", "b"))
  result$h <- factor(result$h, levels = c("u", "v"))
  result$g <- factor(result$g, levels = sprintf("g%d", seq_len(6L)))
  result
}

a01_emmeans_summary <- function(object) {
  table <- as.data.frame(summary(
    object, infer = c(TRUE, TRUE), level = 0.95, adjust = "none"
  ))
  label_names <- setdiff(
    names(object@grid), c(".wgt.", ".offset.", ".type.", ".group.")
  )
  lower_name <- intersect(c("asymp.LCL", "lower.CL"), names(table))[[1L]]
  upper_name <- intersect(c("asymp.UCL", "upper.CL"), names(table))[[1L]]
  statistic_name <- intersect(c("z.ratio", "t.ratio"), names(table))[[1L]]
  list(
    label_names = label_names,
    labels = lapply(seq_len(nrow(table)), function(index) {
      as.list(vapply(label_names, function(name) as.character(table[[name]][index]), character(1)))
    }),
    estimate = scalar(table$emmean),
    standard_error = scalar(table$SE),
    denominator_df = lapply(table$df, function(value) {
      if (is.finite(value)) scalar(value) else NULL
    }),
    lower = scalar(table[[lower_name]]),
    upper = scalar(table[[upper_name]]),
    statistic = scalar(table[[statistic_name]]),
    p_value = scalar(table$p.value),
    linear_functions = unname(as.matrix(object@linfct))
  )
}

a01_pairwise_summary <- function(object, adjustment) {
  contrasts <- contrast(object, method = "pairwise", adjust = adjustment)
  table <- as.data.frame(summary(contrasts, infer = c(TRUE, TRUE)))
  statistic_name <- intersect(c("z.ratio", "t.ratio"), names(table))[[1L]]
  list(
    adjustment = adjustment,
    labels = as.character(table$contrast),
    estimate = scalar(table$estimate),
    standard_error = scalar(table$SE),
    statistic = scalar(table[[statistic_name]]),
    p_value = scalar(table$p.value),
    linear_functions = unname(as.matrix(contrasts@linfct))
  )
}

a01_frame <- make_a01_frame()
a01_model <- lmer(
  y ~ x + f * h + (1 | g),
  data = a01_frame,
  REML = FALSE,
  control = sleepstudy_control()
)
a01_weighting_modes <- c("equal", "proportional", "outer", "cells", "flat")
a01_marginal_means <- setNames(lapply(a01_weighting_modes, function(weighting) {
  object <- emmeans(
    a01_model, specs = "f", at = list(x = 0.0),
    weights = weighting, lmer.df = "asymptotic"
  )
  c(list(weighting = weighting), a01_emmeans_summary(object))
}), a01_weighting_modes)
a01_cells <- emmeans(
  a01_model, specs = c("f", "h"), at = list(x = 0.0),
  weights = "equal", lmer.df = "asymptotic"
)
a01_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    method = "emmeans reference grids and pairwise contrasts for a pinned lme4 fit",
    dataset = "embedded deterministic unbalanced factorial",
    license = "MIT"
  ),
  reference = list(
    profile = "lme4-2.0-6-emmeans-2.0.2-a01-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e",
    emmeans_source_commit = "fbaba0c2e222a7e17bf6c4db59f3575e43607f54"
  ),
  formula = "y ~ x + f * h + (1 | g)",
  fixed_names = unname(names(fixef(a01_model))),
  beta = scalar(fixef(a01_model)),
  covariance = unname(as.matrix(vcov(a01_model))),
  marginal_means = a01_marginal_means,
  pairwise = setNames(lapply(
    c("none", "holm", "bonferroni", "sidak"),
    function(adjustment) a01_pairwise_summary(a01_cells, adjustment)
  ), c("none", "holm", "bonferroni", "sidak"))
)

make_a02_frame <- function() {
  row <- seq_len(80L)
  school_index <- rep(seq_len(16L), each = 5L)
  district_index <- rep(seq_len(8L), each = 10L)
  within <- rep(0:4, 16L)
  school_effects <- c(
    -1.20, -0.55, 0.35, 1.05, -0.80, 0.70, 1.30, -0.25,
    0.45, -1.00, 0.90, -0.40, 1.15, 0.10, -0.65, 0.60
  )
  school_slopes <- c(
    -0.35, 0.20, 0.45, -0.15, 0.30, -0.40, 0.10, 0.38,
    -0.22, 0.50, -0.28, 0.16, -0.46, 0.26, 0.34, -0.08
  )
  residual_pattern <- c(-0.42, 0.18, 0.31, -0.12, 0.05)
  district_pattern <- c(-0.25, 0.20, -0.10, 0.30, -0.18, 0.12, -0.28, 0.19)
  district_z <- c(0.22, -0.30, 0.18, -0.12, 0.35, -0.24, 0.08, -0.20)
  x <- (within - 2.0) / 2.0 + ((school_index - 1L) %% 3L - 1.0) * 0.15
  z <- ((row - 1L) %% 7L - 3.0) / 3.0
  y <- 5.0 + 0.8 * x - 0.45 * z + school_effects[school_index] +
    school_slopes[school_index] * x + district_pattern[district_index] +
    district_z[district_index] * z +
    residual_pattern[(within + school_index - 1L) %% 5L + 1L]
  data.frame(
    row_id = sprintf("a02-%03d", row),
    y = y,
    x = x,
    z = z,
    school = factor(sprintf("s%02d", school_index), levels = sprintf("s%02d", seq_len(16L))),
    district = factor(sprintf("d%02d", district_index), levels = sprintf("d%02d", seq_len(8L))),
    stringsAsFactors = FALSE
  )
}

a02_cr_case <- function(model, cluster, id, joint) {
  robust <- setNames(lapply(c("CR0", "CR1", "CR2"), function(type) {
    unname(as.matrix(vcovCR(model, cluster = cluster, type = type)))
  }), c("CR0", "CR1", "CR2"))
  cr2 <- vcovCR(model, cluster = cluster, type = "CR2")
  estimates <- coef_test(model, vcov = cr2, test = "Satterthwaite")
  joint_test <- Wald_test(
    model,
    constraints = joint,
    vcov = cr2,
    test = "HTZ"
  )
  adjusted_estimating <- Map(
    function(est, adjustment) est %*% adjustment,
    attr(cr2, "est_mats"),
    attr(cr2, "adjustments")
  )
  marginal_residuals <- getME(model, "y") - as.numeric(
    getME(model, "X") %*% fixef(model)
  )
  scores <- do.call(cbind, Map(
    function(est, residual) as.numeric(est %*% residual),
    adjusted_estimating,
    split(marginal_residuals, cluster)
  ))
  list(
    id = id,
    kind = if (isREML(model)) "reml" else "ml",
    fixed_names = unname(names(fixef(model))),
    beta = scalar(fixef(model)),
    covariance = robust,
    bread = unname(attr(cr2, "bread") / attr(cr2, "v_scale")),
    marginal_residuals = scalar(marginal_residuals),
    working_targets = lapply(attr(cr2, "target"), unname),
    adjustments = lapply(attr(cr2, "adjustments"), unname),
    estimating_matrices = lapply(adjusted_estimating, unname),
    score_contributions = unname(scores),
    coefficient_tests = list(
      estimate = scalar(estimates$beta),
      standard_error = scalar(estimates$SE),
      statistic = scalar(estimates$tstat),
      denominator_df = scalar(estimates$df_Satt),
      p_value = scalar(estimates$p_Satt)
    ),
    joint = list(
      contrast = unname(joint),
      statistic = scalar(joint_test$Fstat),
      scale = scalar(joint_test$delta),
      numerator_df = scalar(joint_test$df_num),
      denominator_df = scalar(joint_test$df_denom),
      p_value = scalar(joint_test$p_val)
    )
  )
}

a02_frame <- make_a02_frame()
a02_control <- lmerControl(
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
a02_models <- lapply(c(FALSE, TRUE), function(reml) {
  lmer(
    y ~ x + z + (1 | school),
    data = a02_frame,
    REML = reml,
    control = a02_control
  )
})
a02_sleepstudy <- lmer(
  Reaction ~ Days + (1 + Days | Subject),
  data = sleepstudy,
  REML = TRUE,
  control = a02_control
)
a02_fixture <- list(
  schema_version = "1.0.0",
  source = list(
    method = "clubSandwich CR0/CR1/CR2, Satterthwaite, and HTZ for pinned lme4 fits",
    datasets = c("embedded deterministic nested-cluster data", "lme4 sleepstudy"),
    license = "GPL (>= 2) combined fixture; synthetic input is MIT"
  ),
  reference = list(
    profile = "lme4-2.0.6-clubSandwich-0.7.0-a02-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e",
    clubSandwich_source_commit = "bc925c2c8f27cfb52ab82eab253f7f4b5253f57a"
  ),
  synthetic_data = list(
    row_ids = as.character(a02_frame$row_id),
    response = scalar(a02_frame$y),
    x = scalar(a02_frame$x),
    z = scalar(a02_frame$z),
    school = as.character(a02_frame$school),
    district = as.character(a02_frame$district)
  ),
  cases = c(
    lapply(seq_along(a02_models), function(index) {
      a02_cr_case(
        a02_models[[index]],
        a02_frame$district,
        sprintf("synthetic_higher_cluster_%s", if (isREML(a02_models[[index]])) "reml" else "ml"),
        rbind(c(0, 1, 0), c(0, 0, 1))
      )
    }),
    list(a02_cr_case(
      a02_sleepstudy,
      sleepstudy$Subject,
      "sleepstudy_correlated_reml",
      diag(2L)
    ))
  )
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
write_json(
  pastes_sparse_fixture,
  file.path(output_dir, "pastes_sparse.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  penicillin_sparse_fixture,
  file.path(output_dir, "penicillin_sparse.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  insteval_sparse_fixture,
  file.path(output_dir, "insteval_sparse.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  f02_fixture,
  file.path(output_dir, "f02_rank_categorical.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  i01_fixture,
  file.path(output_dir, "i01_bootstrap.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  i02_fixture,
  file.path(output_dir, "i02_satterthwaite.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  i03_fixture,
  file.path(output_dir, "i03_kr_profile.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  a01_fixture,
  file.path(output_dir, "a01_postfit.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
write_json(
  a02_fixture,
  file.path(output_dir, "a02_cluster_robust.json"),
  auto_unbox = TRUE,
  digits = 17,
  pretty = TRUE,
  null = "null"
)
