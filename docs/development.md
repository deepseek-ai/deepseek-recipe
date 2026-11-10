# Development guide

Run commands from the repository root. Use the Rust toolchain pinned in
[rust-toolchain.toml](../rust-toolchain.toml); rustup selects it automatically
inside this checkout. Python bindings require Python 3.10 or later.

## Native dependencies

The workspace requires a C/C++ compiler. OpenCV development headers and
libraries, Clang, and libclang are also required by the image crate's default
`opencv-preprocess` feature, by the Rust example server, and by the Python
bindings. An application that supplies its own image preprocessor builds the
image crate with `--no-default-features` and without OpenCV.
`--no-default-features` also disables the default `reqwest-fetch` feature;
add `--features reqwest-fetch` to keep the default image fetcher,
`ReqwestImageFetcher`.

For Debian 12 (bookworm):

```sh
sudo apt update
sudo apt install -y build-essential pkg-config clang libclang-dev libopencv-dev
```

For Python development on Debian, also install `python3-dev` and `python3-venv`.

Use OpenCV 4.11 or newer for this workspace. OpenCV 4.11 reads an animated WebP
as its first frame, while an earlier release returns an empty matrix for the
same image. The pinned Rust `opencv` 0.93 bindings support OpenCV 3.4 and 4.x,
not OpenCV 5.x. For other systems or a custom installation, follow the binding
version's [installation guide][opencv-install] and
[troubleshooting guide][opencv-troubleshooting]. Configure `PKG_CONFIG_PATH`,
`OpenCV_DIR`, or `LIBCLANG_PATH` as appropriate for your installation.

The `libopencv-dev` of Debian 12 predates 4.11. Build the OpenCV of the wheel as
described in [Building the Python wheel](build-python-wheel.md) to develop against
a newer release.

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

Save the Rust example in the [README](../README.md#rust) as
`../recipe-example/src/main.rs`. Run it from the repository root to use the
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
python3 -m pip install 'maturin>=1.9,<2' 'pytest>=8,<10' httpx -r server-py/requirements.txt
maturin develop --manifest-path deepseek-recipe-python/Cargo.toml --locked
python3 -m pytest deepseek-recipe-python/tests server-py/tests
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Run `maturin develop` again after changing any Rust code in the workspace. The
tests exercise request conversion, prompt rendering, response processing, and
the FastAPI example with mock inference; no model weights or API credentials are
needed.

For an installation without editable development, use:

```sh
python3 -m pip install ./deepseek-recipe-python
```

## Python wheel

[Building the Python wheel](build-python-wheel.md) describes a native build: the
requirements of the build machine, the minimal OpenCV that keeps the wheel small,
the build with `maturin`, the bundling of the native libraries, and the checks of
the artifacts. It also covers the platform tags, the licenses of the bundled
libraries, and the upload to a package index.

## API documentation

Build and open the Rust API documentation locally:

```sh
cargo doc --workspace --no-deps --locked --open
```

The Python package includes a `py.typed` marker and
[type stubs](../deepseek-recipe-python/python/deepseek_recipe/_native.pyi).

### Documentation checks

For changes to the main library's rustdoc, check its examples and documentation:

```sh
cargo test -p deepseek-recipe --doc --locked
RUSTDOCFLAGS='-D warnings' cargo doc -p deepseek-recipe --no-deps --locked
```

For other packages, run the same two commands with their `-p` names. Use the
[full Rust checks](#rust-checks) for code, API, or dependency changes.
The environment-variable command uses POSIX shell syntax; on PowerShell, set
`$env:RUSTDOCFLAGS = '-D warnings'` before running `cargo doc`.

`cargo doc` does not run examples, and `cargo test` runs only rustdoc examples.
Code blocks in the README and `docs/` are not tested: after changing one,
compile or run it with the dependencies it lists. State in the change
description which examples you could not run. For behavior changes, synchronize
the affected English and Chinese guides. Check changed links and heading
anchors, and inspect rendered documentation when changing its structure.

## Example servers

[server-rs](../server-rs/README.md) and [server-py](../server-py/README.md)
listen on `127.0.0.1:7777` by default. Run one at a time or choose different
ports. server-rs uses mock inference only. server-py uses mock inference by
default and accepts a caller-supplied backend through `create_app(infer=...)`.
Both provide JSON and SSE examples for the three protocols.
[encoding-decoding-demo](../encoding-decoding-demo/README.md) listens on
`127.0.0.1:7778`. It renders prompts, displays special tokens, and parses
complete model output into Chat Completions, Responses, or Messages responses
without executing a model.

The example servers resolve the images of a request with `ImageResolver` and
answer an `ImageError` with the error's text as the response body: a `500`
for `ImageError::Client` and `ImageError::TokenBudget`, which are server-side
failures, and a `400` for every other variant. The text names the request
location of the image, such as `.messages[0].image[1]: Failed to download image
from https://example.com/a.png`.

The examples have no authentication. Keep them on loopback for local use.
Applications exposed to other users need authentication, request limits, and
appropriate network access controls. `ReqwestImageFetcher` follows redirects
and, without a destination filter, connects to any address. When accepting
untrusted image URLs, install a `DestinationFilter` with
`ReqwestImageFetcher::with_destination_filter` (Python:
`ReqwestImageFetcher.with_destination_filter`). The filter checks the initial
request and every redirect, including each address that a host name resolves
to. The library does not include a filter implementation; the caller writes the
rule. Image quotas and image byte budgets do not filter destinations.

[opencv-install]: https://github.com/twistedfall/opencv-rust/blob/v0.93.7/INSTALL.md
[opencv-troubleshooting]: https://github.com/twistedfall/opencv-rust/blob/v0.93.7/TROUBLESHOOTING.md
