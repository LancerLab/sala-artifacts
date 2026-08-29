# SALA artifact evaluation container (CGO 2027)
# The exact image + toolchain verified in the clean-container reproduction
# (figure2 9/9, Table-2 GEMM 5/5 + FA 3/3, all kernels Test Passed).
# Build:  docker build -t sala-ae .
# Run:    docker run --rm --gpus all -it sala-ae bash
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cmake ninja-build flex bison gcc g++ git python3 python3-pip \
        python3-dev bc libzstd-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# A newer setuptools than the distro's 59.6.0 (torch's requirements pull it;
# pip 22.0.2 then prints a scary-but-benign "Can't uninstall setuptools" and
# proceeds anyway - pre-installing avoids the noise). --ignore-installed:
# keep the apt copy untouched, the new one shadows it. pip stays 22.0.2
# (a newer pip would refuse system installs via the externally-managed guard).
RUN pip3 install --no-cache-dir --ignore-installed setuptools wheel

COPY . /ae
WORKDIR /ae

# nlohmann/json v3.11.3 include.zip for the Tawa fork build: bundled in
# the repo (benchmarks/tawa/json_include.zip, MIT) so the build never
# touches github for it. The version.txt marker (exact URL match, no
# trailing newline) makes the fork's setup.py skip the download.
# (After the COPY: the file must exist in the image first.)
RUN mkdir -p /root/.triton/json \
 && python3 -c "import zipfile; zipfile.ZipFile('/ae/benchmarks/tawa/json_include.zip').extractall('/root/.triton/json')" \
 && printf '%s' "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip" > /root/.triton/json/version.txt

# The fork build's NVIDIA redistributables are bundled in
# benchmarks/tawa/redist/ and pre-placed into ~/.triton here, so
# `pip install .` of the fork downloads only the LLVM build
# (oaitriton.blob.core.windows.net, ~1.2 GB - reliable host; retry on
# timeout). googletest is bundled in the fork's third_party/ and used
# via the cmake override; json is bundled above.
RUN for d in nvcc cuobjdump nvdisasm cudart cupti; do mkdir -p /root/.triton/nvidia/$d; done \
 && cp /ae/benchmarks/tawa/redist/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz /root/.triton/nvidia/nvcc/ \
 && cp /ae/benchmarks/tawa/redist/cuda_nvcc-linux-x86_64-12.8.61-archive.tar.xz /root/.triton/nvidia/nvcc/ \
 && cp /ae/benchmarks/tawa/redist/cuda_cuobjdump-linux-x86_64-12.8.55-archive.tar.xz /root/.triton/nvidia/cuobjdump/ \
 && cp /ae/benchmarks/tawa/redist/cuda_nvdisasm-linux-x86_64-12.8.55-archive.tar.xz /root/.triton/nvidia/nvdisasm/ \
 && cp /ae/benchmarks/tawa/redist/cuda_cudart-linux-x86_64-12.8.57-archive.tar.xz /root/.triton/nvidia/cudart/ \
 && cp /ae/benchmarks/tawa/redist/cuda_cupti-linux-x86_64-12.8.90-archive.tar.xz /root/.triton/nvidia/cupti/

# The configure step fetches CUTLASS v4.2.1 automatically (one-time,
# ~100 MB). HTTP/1.1 makes the fetch robust under flaky networks
# (Ubuntu's git-over-HTTP/2 with GnuTLS can fail with TLS -110 errors).
RUN git config --global http.version HTTP/1.1 \
 && cmake -S croqtile -B croqtile/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCHOREO_DEFAULT_TARGET=cute \
    && ninja -C croqtile/build choreo copp

# The container provides the complete prerequisite environment: the
# compiler is built and ready; the reproduction scripts resolve
# $REPO/croqtile/build/choreo themselves. Enter the shell and follow
# README.md:
#   docker run --rm --gpus all -it sala-ae
CMD ["bash"]
