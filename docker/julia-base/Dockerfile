ARG JULIA_VERSION=1.10
FROM julia:${JULIA_VERSION}

# Install Python and git (needed by pytest + submodule checkout in CI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

# Create a venv so pip installs work without --break-system-packages (PEP 668)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Store Julia depot outside $HOME so GitHub Actions' HOME=/github/home override
# doesn't hide the pre-installed packages.
ENV JULIA_DEPOT_PATH=/opt/julia-depot

# Pre-install and precompile all heavy Julia packages.
# PySD.jl is intentionally excluded: it lives in the repo and is developed
# in-place by the CI job, so only it needs recompiling on each run (fast).
RUN julia --startup-file=no -e '\
    using Pkg; \
    Pkg.add([ \
        "OrdinaryDiffEq", \
        "OrdinaryDiffEqLowOrderRK", \
        "DataInterpolations", \
        "NCDatasets", \
        "JSON3", \
    ]); \
    Pkg.precompile() \
    '
