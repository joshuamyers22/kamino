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

make_dyestuff_fit <- function(reml) {
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
  model <- lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = reml, control = control)
  levels <- levels(Dyestuff$Batch)
  existing <- data.frame(Batch = factor(levels, levels = levels))
  unseen <- data.frame(Batch = factor("new", levels = c(levels, "new")))
  optimizer <- model@optinfo
  convergence_message <- optimizer$conv$lme4$messages
  if (is.null(convergence_message)) convergence_message <- character()

  list(
    id = sprintf("dyestuff_random_intercept_%s", if (reml) "reml" else "ml"),
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
    convergence_messages = unname(as.character(convergence_message))
  )
}

dyestuff_parsed <- lFormula(
  Yield ~ 1 + (1 | Batch),
  data = Dyestuff,
  REML = TRUE,
  na.action = na.fail
)
dyestuff_levels <- levels(dyestuff_parsed$fr$Batch)
dyestuff <- list(
  schema_version = "1.0.0",
  source = list(
    package = "lme4",
    dataset = "Dyestuff",
    citation = "O.L. Davies and P.L. Goldsmith (eds), Statistical Methods in Research and Production, 4th ed. (1972), section 6.4",
    license = "GPL (>= 2), following lme4 2.0-6 package metadata",
    modification = "Converted to JSON and augmented with lme4 fit, diagnostic, and prediction outputs"
  ),
  reference = list(
    profile = "lme4-2.0.6-unstructured-gaussian-v1",
    lme4_source_commit = "4aa26a91f9e676e9409f6cd8163ae92654ef1e7e"
  ),
  data = list(
    row_ids = sprintf("dyestuff-%02d", seq_len(nrow(Dyestuff))),
    response_name = "Yield",
    response = scalar(Dyestuff$Yield),
    group_name = "Batch",
    groups = unname(as.character(Dyestuff$Batch)),
    group_levels = unname(dyestuff_levels),
    fixed_names = I(unname(colnames(dyestuff_parsed$X))),
    random_names = unname(sprintf("Batch[%s]:(Intercept)", dyestuff_levels)),
    X = unname(as.matrix(dyestuff_parsed$X)),
    Z = unname(t(as.matrix(dyestuff_parsed$reTrms$Zt)))
  ),
  fits = lapply(c(FALSE, TRUE), make_dyestuff_fit)
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
