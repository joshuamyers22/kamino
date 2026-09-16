FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG LME4_COMMIT=4aa26a91f9e676e9409f6cd8163ae92654ef1e7e
ARG LMERTEST_COMMIT=35dc5885205d709cdc395b369b08ca2b7273cb78
ARG PBKRTEST_COMMIT=4ead4ff46e90831f4a2376cdcae0602045db0458
ARG EMMEANS_COMMIT=fbaba0c2e222a7e17bf6c4db59f3575e43607f54
ARG RCPP_COMMIT=a76e1c9bab241d008940626641cf7ad48cbc7383
ARG NLOPT_APT_VERSION=2.7.1-5build2

ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    RSPM=https://p3m.dev/cran/__linux__/noble/2026-03-10

LABEL org.kamino.oracle.lme4-commit="${LME4_COMMIT}" \
      org.kamino.oracle.lmertest-commit="${LMERTEST_COMMIT}" \
      org.kamino.oracle.pbkrtest-commit="${PBKRTEST_COMMIT}" \
      org.kamino.oracle.emmeans-commit="${EMMEANS_COMMIT}" \
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

RUN Rscript -e 'remotes::install_github(sprintf("runehaubo/lmerTestR@%s", Sys.getenv("LMERTEST_COMMIT")), dependencies = c("Depends", "Imports"), upgrade = "never")'

RUN Rscript -e 'remotes::install_github(sprintf("cran/pbkrtest@%s", Sys.getenv("PBKRTEST_COMMIT")), dependencies = c("Depends", "Imports"), upgrade = "never")'

RUN Rscript -e 'remotes::install_github(sprintf("cran/emmeans@%s", Sys.getenv("EMMEANS_COMMIT")), dependencies = c("Depends", "Imports"), upgrade = "never")'

WORKDIR /work
ENTRYPOINT ["Rscript", "oracle/generate_fixtures.R"]
