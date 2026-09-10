# Development guide

Run commands from the repository root. Use the Rust toolchain pinned in
[rust-toolchain.toml](../rust-toolchain.toml); rustup selects it automatically
inside this checkout. Python bindings require Python 3.10 or later.

## Native dependencies

The workspace requires a C/C++ compiler. OpenCV development headers and
libraries, Clang, and libclang are also required by the image crate's default
`opencv-preprocess` feature, by the Rust example server, by the inference demo,
and by the Python bindings. An application that supplies its own image
preprocessor builds the image crate with `--no-default-features` and without
OpenCV.

For Debian 12 (bookworm):

```sh
sudo apt update
sudo apt install -y build-essential pkg-config clang libclang-dev libopencv-dev
```

For Python development on Debian, also install `python3-dev` and `python3-venv`.

Use OpenCV 4.x for this workspace. The pinned Rust `opencv` 0.93 bindings support
OpenCV 3.4 and 4.x, not OpenCV 5.x. For other systems or a custom installation,
follow the binding version's [installation guide][opencv-install] and
[troubleshooting guide][opencv-troubleshooting]. Configure `PKG_CONFIG_PATH`,
`OpenCV_DIR`, or `LIBCLANG_PATH` as appropriate for your installation.

To check an OpenCV 4 installation discovered through pkg-config:

```sh
pkg-config --modversion opencv4
```

## Rust source installation

To use an unreleased checkout, create an application beside this repository:

```sh
cargo new --edition 2024 ../recipe-example
```

Add these entries under `[dependencies]` in `../recipe-example/Cargo.toml`.
Adjust the paths if the checkout directory has a different name.

```toml
deepseek-recipe = { path = "../deepseek-recipe/deepseek-recipe" }
deepseek-recipe-encoding = { path = "../deepseek-recipe/deepseek-recipe-encoding" }
serde_json = "1"
```

Save the Rust quick-start example from the [README](../README.md#quick-start)
as `../recipe-example/src/main.rs`. Run it from the repository root to use the
pinned toolchain:

```sh
cargo run --manifest-path ../recipe-example/Cargo.toml
```

## Rust checks

Run the full checks when changing Rust code, public APIs, or dependencies:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
```

The last command uses POSIX shell syntax. On PowerShell, set
`$env:RUSTDOCFLAGS = '-D warnings'` before running `cargo doc`.

For protocol or prompt work without OpenCV, check the relevant packages:

```sh
cargo test -p deepseek-recipe-core -p deepseek-recipe -p deepseek-recipe-encoding -p encoding-decoding-demo --locked
```

The image crate compiles without OpenCV when its default features are disabled:

```sh
cargo check -p deepseek-recipe-image --no-default-features --locked
```

Cargo downloads dependencies on the first build.

## Python development and tests

After installing the native dependencies, create a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install 'maturin>=1.9,<2' 'pytest>=8,<10' httpx
maturin develop --manifest-path deepseek-recipe-python/Cargo.toml --locked
python3 -m pytest deepseek-recipe-python/tests server-py/tests
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Run `maturin develop` again after changing Rust bindings. The tests exercise
request conversion, prompt rendering, response processing, and the FastAPI
example with mock inference; no model weights or API credentials are needed.

For an installation without editable development, use:

```sh
python3 -m pip install ./deepseek-recipe-python
```

## Wheel packaging

The extension links OpenCV dynamically, so a wheel is only installable on a
machine with a matching OpenCV until the libraries are copied into it:
`delocate` on macOS, `auditwheel` on Linux. The libraries of the build machine
become part of the wheel, so the wheel is built against an OpenCV that is small
enough to bundle.

Build that OpenCV from source, with only the modules the extension uses. OpenCL,
TBB, OpenEXR, and TIFF are left out because each of them links libraries of its
own that the wheel would then have to carry.

```sh
git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git
cmake -S opencv -B opencv/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/target/opencv-min" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DWITH_OPENCL=OFF -DWITH_TBB=OFF -DWITH_OPENEXR=OFF -DWITH_TIFF=OFF \
    -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_WEBP=ON
cmake --build opencv/build --parallel
cmake --install opencv/build
```

Keep `CMAKE_OSX_DEPLOYMENT_TARGET` at the oldest release the wheel should support:
the installed libraries decide the platform tag of the wheel.

Then build the wheel against that OpenCV, and copy its libraries into it:

```sh
cd deepseek-recipe-python
export OPENCV_INCLUDE_PATHS="$PWD/../target/opencv-min/include/opencv4"
export OPENCV_LINK_PATHS="$PWD/../target/opencv-min/lib"
export OPENCV_LINK_LIBS=opencv_core,opencv_imgproc,opencv_imgcodecs
maturin build --release
delocate-wheel -w dist-repaired dist/*.whl      # `auditwheel repair` on Linux
twine check dist-repaired/*
```

The `OPENCV_*` variables take precedence over pkg-config, so an OpenCV installed
elsewhere on the build machine cannot take over. The linker still resolves
`-lopencv_core` in the order of its search paths, so a build machine that carries
another OpenCV in its flags has to search this one first:

```sh
export RUSTFLAGS="-L$PWD/../target/opencv-min/lib ${RUSTFLAGS:-}"
```

On macOS this produces a wheel of about 8 MB; bundling an OpenCV installed by a
package manager produces a much larger one, because that build links every
library of its own dependencies.

Upload the repaired artifacts after checking them against TestPyPI:

```sh
twine upload --repository testpypi deepseek-recipe-python/dist-repaired/*
twine upload deepseek-recipe-python/dist-repaired/*
```

An upload authenticates with an API token: `__token__` as the user name and the
token as the password, in `~/.pypirc` or in `TWINE_USERNAME` and
`TWINE_PASSWORD`.

The wheel bundles OpenCV and the codecs that its `imgcodecs` module compiles in,
so it also carries their license texts:
`deepseek-recipe-python/THIRD_PARTY_LICENSES` reproduces them and the package
declares the file as a license file.

## API documentation

Build and open the Rust API documentation locally:

```sh
cargo doc --workspace --no-deps --locked --open
```

The Python package includes a `py.typed` marker and
[type stubs](../deepseek-recipe-python/python/deepseek_recipe/_native.pyi).

## Example servers

[server-rs](../server-rs/README.md) and [server-py](../server-py/README.md) listen on
`127.0.0.1:7777` by default. Run one at a time or choose different ports. Both
default to mock inference and provide JSON and SSE examples for the three protocols.
[encoding-decoding-demo](../encoding-decoding-demo/README.md) listens on
`127.0.0.1:7778`. It encodes prompts, displays special tokens, and decodes complete
model output into Chat Completions, Responses, or Messages without executing a model.

The examples have no authentication. Keep them on loopback for local use.
Applications exposed to other users need authentication, request limits, and
appropriate network access controls. The default `ReqwestImageFetcher` follows
redirects and does not filter private, loopback, or link-local destinations.
When accepting untrusted image URLs, enforce destination restrictions on the
initial request and every redirect, using a custom fetcher or network policy.
Image byte quotas do not provide this destination filtering.

[opencv-install]: https://github.com/twistedfall/opencv-rust/blob/v0.93.7/INSTALL.md
[opencv-troubleshooting]: https://github.com/twistedfall/opencv-rust/blob/v0.93.7/TROUBLESHOOTING.md
