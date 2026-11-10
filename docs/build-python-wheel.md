# Building the Python wheel

The Python package is a compiled extension, so its wheel is built on the platform
it runs on. This page covers a native build: the build machine is the platform of
the wheel.

A build produces one wheel per platform and architecture, not one per Python
version: the extension uses the stable ABI (`cp310-abi3`) and runs on every
CPython from 3.10 onwards.

## Requirements

- Python 3.10 or later, with `venv`.
- The Rust toolchain pinned in [rust-toolchain.toml](../rust-toolchain.toml);
  rustup selects it inside the checkout.
- A C/C++ compiler, CMake, Clang, and libclang. The `opencv` crate generates its
  bindings by running the `clang` binary during the build.
- OpenCV 4.11 or newer with its development headers. The published wheels bundle
  OpenCV 4.12.0, which the steps below build.
- `maturin` for the build, `delocate` on macOS, `auditwheel` on Linux, or
  `delvewheel` on Windows to bundle the native libraries, and `twine` to check the
  metadata.

For Debian 12 (bookworm), `libopencv-dev` is older than 4.11, so build the
OpenCV of the wheel as described below. The compiler, CMake, and Clang come from:

```sh
sudo apt update
sudo apt install -y build-essential pkg-config cmake clang libclang-dev
```

On macOS, the Command Line Tools provide Clang and libclang. When the build does
not find libclang, point `LIBCLANG_PATH` at the directory that holds
`libclang.dylib`.

On Windows, the build needs Visual Studio Build Tools with the C++ workload, LLVM
for the `clang` binary and `libclang.dll`, and the `x86_64-pc-windows-msvc` target
of the Rust toolchain:

```sh
rustup target add x86_64-pc-windows-msvc
```

The commands of this page use the shell of Git for Windows. In PowerShell,
`export NAME=value` becomes `$env:NAME = 'value'`.

Create a virtual environment with the build tools:

```sh
python3 -m venv .venv
.venv/bin/pip install 'maturin>=1.9,<2' 'pytest>=8,<10' twine
.venv/bin/pip install delocate        # macOS
.venv/bin/pip install auditwheel      # Linux
.venv/bin/pip install delvewheel      # Windows
source .venv/bin/activate
```

## Wheel size and the bundled OpenCV

The extension links OpenCV dynamically. A wheel that is not repaired therefore
references the OpenCV of the build machine, and it runs only on a machine with
that OpenCV. `delocate`, `auditwheel`, and `delvewheel` copy the referenced
libraries into the wheel and rewrite the references, which makes the wheel
self-contained.

The libraries of the build machine become part of the wheel, so the size of the
wheel is decided by the OpenCV that the build links:

| OpenCV of the build | Bundled libraries | Wheel (macOS arm64) | Wheel (manylinux x86_64) |
| --- | --- | --- | --- |
| Minimal build from source | 3 | 8.3 MB | 11.4 MB |
| Distribution or package manager build | many | 86 MB | not measured |

A distribution build of OpenCV enables modules and backends that this project does
not use, and each of them links libraries of its own, which the wheel then has to
carry. The next section builds an OpenCV with the three modules of the extension,
without the backends that pull in further libraries.

A third option is to publish an unrepaired wheel. It is the smallest one, and it
is only usable on machines with a matching OpenCV, so it does not suit a package
index.

## Build a minimal OpenCV

The image preprocessing of the extension decodes PNG, JPEG, and WebP, resizes the
image, and encodes it as WebP through the `core`, `imgproc`, and `imgcodecs`
modules. OpenCV compiles its codecs into `imgcodecs` (`BUILD_ZLIB`, `BUILD_JPEG`,
`BUILD_PNG`, `BUILD_WEBP`), so the install holds three shared libraries and no
further codec library.

OpenCL, TBB, OpenEXR, TIFF, FFmpeg, GTK, LAPACK, OpenJPEG, Jasper, the Python
bindings, and the Java bindings are left out: each of them links libraries of its
own.

```sh
git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git
cmake -S opencv -B opencv/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/target/opencv-min" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_WEBP=ON \
    -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_WEBP=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DWITH_ADE=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JASPER=OFF \
    -DWITH_LAPACK=OFF -DWITH_OPENCL=OFF -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENEXR=OFF -DWITH_OPENJPEG=OFF \
    -DWITH_OPENMP=OFF -DWITH_QUIRC=OFF -DWITH_TBB=OFF -DWITH_TIFF=OFF \
    -DWITH_V4L=OFF -DWITH_1394=OFF
cmake --build opencv/build --parallel
cmake --install opencv/build
```

Two parameters of the build decide the platform tag of the wheel:

- `CMAKE_OSX_DEPLOYMENT_TARGET` on macOS. Keep it at the oldest release the wheel
  should support; 11.0 produces the tag `macosx_11_0_arm64` or
  `macosx_11_0_x86_64`.
- The C library of the Linux build environment on Linux. A wheel built on a
  current distribution is tagged `linux_x86_64`, which a package index rejects.
  Build inside a manylinux image of the architecture of the wheel to obtain a
  `manylinux_*` tag. The image carries CMake and a compiler, and the commands above
  run in it unchanged:

  ```sh
  docker run --rm -it -v "$PWD:/work" -w /work quay.io/pypa/manylinux_2_28_x86_64 bash
  ```

The build installs into `target/opencv-min`, which the next steps read. On a
machine with many cores it takes about a minute.

## Build the wheel

The `opencv` crate reads `OPENCV_INCLUDE_PATHS`, `OPENCV_LINK_PATHS`, and
`OPENCV_LINK_LIBS` before it probes pkg-config, so an OpenCV installed elsewhere
on the build machine cannot take over:

```sh
cd deepseek-recipe-python
export OPENCV_INCLUDE_PATHS="$PWD/../target/opencv-min/include/opencv4"
export OPENCV_LINK_PATHS="$PWD/../target/opencv-min/lib"
export OPENCV_LINK_LIBS=opencv_core,opencv_imgproc,opencv_imgcodecs
```

The linker resolves `-lopencv_core` in the order of its search paths, so a search
path of another OpenCV would win. When the linker flags of the build machine
already name the directory of another OpenCV, such as a global `RUSTFLAGS` with a
package manager's library directory, add this one first:

```sh
export RUSTFLAGS="-L$PWD/../target/opencv-min/lib -C link-arg=-Wl,-rpath,$PWD/../target/opencv-min/lib ${RUSTFLAGS:-}"
unset CARGO_ENCODED_RUSTFLAGS
```

The extension records its OpenCV as `@rpath/libopencv_*.dylib` on macOS and as
`libopencv_*.so.412` on Linux. The rpath above makes the dynamic loader and the
repair step resolve those names against this OpenCV rather than against one from
a package manager.

Build the wheel and the source distribution:

```sh
maturin build --release
maturin sdist
```

Both land in `dist/`. The wheel is `cp310-abi3` and named after the platform, for
example `deepseek_recipe-0.1.1-cp310-abi3-macosx_11_0_arm64.whl`.

## Bundle the native libraries

Copy the referenced libraries into the wheel and rewrite the references:

```sh
delocate-wheel -w dist-repaired dist/*.whl        # macOS
auditwheel repair -w dist-repaired dist/*.whl     # Linux
```

`delocate` places the libraries in `deepseek_recipe/.dylibs`, `auditwheel` places
them in `deepseek_recipe.libs`, and both rewrite the references of the extension
to load them from beside itself. Copy the source distribution, which needs no
repair, beside the repaired wheel:

```sh
cp dist/*.tar.gz dist-repaired/
```

Check that the wheel resolves every library inside itself:

```sh
delocate-listdeps --all dist-repaired/*.whl   # macOS: only /usr/lib and /System
auditwheel show dist-repaired/*.whl           # Linux: the manylinux tag it claims
```

The check that matters is that the installed extension loads its own libraries. A
virtual environment with the wheel and the tests does this on the build machine:

```sh
python3 -m venv /tmp/wheel-test
/tmp/wheel-test/bin/pip install dist-repaired/*.whl pytest
/tmp/wheel-test/bin/python -m pytest tests -q
/tmp/wheel-test/bin/python -c "import deepseek_recipe; print(deepseek_recipe.__file__)"
```

## Windows

A Windows build differs from the steps above in the generator of the OpenCV build,
the names of the import libraries, and the tool that bundles the DLLs. The build
needs Visual Studio Build Tools with the C++ workload, and LLVM for the `clang`
binary and `libclang.dll` that the `opencv` crate runs while it generates the
bindings; point `LIBCLANG_PATH` at the directory of `libclang.dll` when the build
does not find it.

### Build the minimal OpenCV

The Visual Studio generator takes the architecture of the build, and it builds and
installs a configuration at a time. The arguments are the ones of the build above;
keep the two lists in step.

```sh
git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git
cmake -S opencv -B opencv/build -A x64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/target/opencv-min" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_WEBP=ON \
    -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_WEBP=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DWITH_ADE=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JASPER=OFF \
    -DWITH_LAPACK=OFF -DWITH_OPENCL=OFF -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENEXR=OFF -DWITH_OPENJPEG=OFF \
    -DWITH_OPENMP=OFF -DWITH_QUIRC=OFF -DWITH_TBB=OFF -DWITH_TIFF=OFF \
    -DWITH_V4L=OFF -DWITH_1394=OFF
cmake --build opencv/build --config Release --parallel
cmake --install opencv/build --config Release
```

The install places the DLLs in `bin`, the import libraries in `lib`, and the
headers directly in `include`. The file names carry the version of OpenCV, because
OpenCV appends its major, minor, and patch numbers to the module name on Windows:
`opencv_core4120.dll` and `opencv_core4120.lib` for 4.12.0.

### Build the wheel

The import libraries carry that version, so `OPENCV_LINK_LIBS` holds the names
with the suffix, and the include directory is `include` rather than
`include/opencv4`:

```sh
cd deepseek-recipe-python
export OPENCV_INCLUDE_PATHS="$PWD/../target/opencv-min/include"
export OPENCV_LINK_PATHS="$PWD/../target/opencv-min/lib"
export OPENCV_LINK_LIBS=opencv_core4120,opencv_imgproc4120,opencv_imgcodecs4120
maturin build --release
maturin sdist
```

A change of the OpenCV version changes the suffix; `ls ../target/opencv-min/lib`
lists the names of the current build. Windows has no rpath, so these variables are
the whole link configuration, and the repair step rewrites the import table
afterwards.

### Bundle the DLLs

```sh
delvewheel repair --add-path "$PWD/../target/opencv-min/bin" -w dist-repaired dist/*.whl
cp dist/*.tar.gz dist-repaired/
```

`delvewheel` copies the DLLs into `deepseek_recipe.libs` and rewrites the import
table of the extension. `delvewheel show` lists the DLL dependencies that remain
outside the wheel:

```sh
delvewheel show dist-repaired/*.whl
```

The wheel is `cp310-abi3-win_amd64`. A fresh environment finds no OpenCV unless one
was installed for it separately, so an import in that environment proves that the
DLLs are bundled:

```sh
python -m venv target/wheel-test
target/wheel-test/Scripts/pip install dist-repaired/*.whl pytest
target/wheel-test/Scripts/python -m pytest tests -q
target/wheel-test/Scripts/python -c "import deepseek_recipe; print(deepseek_recipe.__file__)"
```

## Check the metadata

```sh
twine check dist-repaired/*
```

The package declares its license files through the PEP 639 `license-files` field,
which needs `twine` 7 or newer. An older release reports the field as malformed.

## Licenses of the bundled libraries

The wheel contains OpenCV, which is licensed under Apache-2.0, and the codecs that
its `imgcodecs` module compiles in, which are licensed under BSD-3-Clause, the PNG
Reference Library license, and the zlib license.
[deepseek-recipe-python/THIRD_PARTY_LICENSES](../deepseek-recipe-python/THIRD_PARTY_LICENSES)
reproduces those licenses, and the package declares the file as a license file.
Collect the texts again from the OpenCV checkout when the version of OpenCV
changes.

The CycloneDX SBOM of the wheel lists the Rust crates. OpenCV is not a Rust crate,
so that document does not cover it.

## Upload

Check the artifacts against TestPyPI before uploading them to PyPI:

```sh
twine upload --repository testpypi dist-repaired/*
twine upload dist-repaired/*
```

An upload authenticates with an API token: `__token__` as the user name and the
token as the password, in `~/.pypirc` or in `TWINE_USERNAME` and
`TWINE_PASSWORD`. A version that reaches PyPI cannot be replaced, and a file of a
released version cannot be replaced either.
