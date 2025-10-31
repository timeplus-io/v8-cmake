# Repository Guidelines

## Project Structure & Module Organization
- `CMakeLists.txt`: Single entry point; handles native and cross builds internally.
- `cmake/`: CMake helpers and modules.
- `v8/`: Upstream V8 sources (vendored). Avoid direct edits; prefer upstream patches or local CMake-only changes.
- `patches/`: Local patch files, used sparingly to adapt upstream.
- `build/`: Out-of-tree build output (create with `-B build`).
- `update_v8.py`, `update_v8.json`: Maintainer scripts for bumping V8.

## Build, Test, and Development Commands
- Configure: `cmake -S . -B build` (use `-DPYTHON_EXECUTABLE=/usr/bin/python3` if detection fails; toggle `-DV8_ENABLE_I18N=ON|OFF`).
- Build all: `cmake --build build -j` (or `cd build && make -j8`).
- Build a target: `cmake --build build --target v8_snapshot -j` (common targets include `v8_base_without_compiler`, `v8_compiler`, `v8_libplatform`, `mksnapshot`, `torque`).
- Cross-compile: tool-generating steps expect host-built binaries. Place host executables `torque`, `mksnapshot`, and `bytecode_builtins_list_generator` at the repo root, then configure with your toolchain as usual (e.g., `-DCMAKE_TOOLCHAIN_FILE=...`). You can override the host tools location with `-DV8_HOST_TOOLS_DIR=/absolute/path/to/host/tools`.

## Coding Style & Naming Conventions
- CMake: 2-space indent, lowercase commands, group related sources/defs; prefer generator expressions for platform guards (see `v8_defines`).
- C++: follow upstream V8 style; C++20; no exceptions (`-fno-exceptions`); include paths relative to `v8/` tree.
- Targets: prefix with `v8_`; provide `ch_contrib::` aliases when adding public-facing targets for consumers.
- Do not modify files under `v8/` unless absolutely required; prefer CMake-only changes or isolated patches.

## Testing Guidelines
- This repo validates via successful multi-OS builds (see `.github/workflows/ci.yml`).
- Local smoke tests: build `v8_snapshot`, `mksnapshot`, and `torque` and run `build/mksnapshot --help` or `build/torque --help`.
- If changing cross-compilation, verify both native and cross builds configure and generate outputs.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (<=72 chars), detailed body when changing build flags, platforms, or cross-compile behavior. Reference issues when relevant.
- PRs must include: problem statement, summary of changes, affected platforms, sample configure line(s), and how you validated (commands/output). Screenshots optional.

## Maintenance & Updates
- To bump V8: run `python update_v8.py`. For minor/major upgrades, update `branch` in `update_v8.json` then re-run. Commit updated files and tag the release appropriately.
