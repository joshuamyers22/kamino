FROM rocker/r-ver:4.5.3@sha256:c3f39b365d1077fe24f8e9ab2742e352b6d3950897f51af1624a5bb5550c21c0

ARG LME4_COMMIT=4aa26a91f9e676e9409f6cd8163ae92654ef1e7e
ARG RCPP_COMMIT=a76e1c9bab241d008940626641cf7ad48cbc7383
ARG NLOPT_APT_VERSION=2.7.1-5build2

ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    RSPM=https://p3m.dev/cran/__linux__/noble/2026-03-10

LABEL org.kamino.oracle.lme4-commit="${LME4_COMMIT}" \
      org.kamino.oracle.rcpp-commit="${RCPP_COMMIT}" \
      org.kamino.oracle.rspm-snapshot="2026-03-10"

RUN apt-get update \
    && apt-get install -y --no-install-recommends "libnlopt-dev=${NLOPT_APT_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# The base repository snapshot resolves exact package versions; Rcpp and lme4
# postdate that snapshot and therefore use immutable CRAN-mirror commits.
RUN install2.r --error --skipinstalled jsonlite remotes \
    && Rscript -e 'remotes::install_github(sprintf("cran/Rcpp@%s", Sys.getenv("RCPP_COMMIT")), dependencies = FALSE, upgrade = "never", build = FALSE)' \
    && Rscript -e 'remotes::install_github(sprintf("cran/lme4@%s", Sys.getenv("LME4_COMMIT")), dependencies = c("Depends", "Imports", "LinkingTo"), upgrade = "never")'

WORKDIR /work
ENTRYPOINT ["Rscript", "oracle/generate_fixtures.R"]
