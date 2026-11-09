# Contributing

Issues and pull requests are welcome. The public API is experimental, so discuss
large API changes before investing in an implementation. For bug reports, include
a minimal reproduction, enabled Cargo features, Rust/OS versions and relevant
FFmpeg or CUDA versions. Use synthetic media where possible, or link to an input
you have permission to share.

## Development setup

Install Rust 1.88 or newer with rustfmt and Clippy. The default image/audio
backends do not need native SDKs. From the repository root:

```sh
cargo test --workspace --exclude media-backend-ffmpeg --locked
cargo check -p media --no-default-features --locked
cargo test -p media --features interop,http-tls --locked
cargo fmt --all --check
```

For FFmpeg development on Debian/Ubuntu, install the SDK and fixture generator:

```sh
sudo apt-get install clang pkg-config ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev libswresample-dev
```

On macOS, `brew install ffmpeg pkg-config llvm` provides the libraries and tools.
If discovery needs help, add Homebrew's bin directory to PATH and its
`lib/pkgconfig` directory to `PKG_CONFIG_PATH`.

With those dependencies installed, run:

```sh
MEDIA_REQUIRE_FFMPEG_TESTS=1 cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps --locked
```

GPU execution needs real NVIDIA hardware and the runtime libraries described in
the [README](README.md#building-and-testing). GPU tests are ignored by ordinary
CPU runs; see [testing](tests/README.md) and the
[GPU harness](infra/gpu/README.md) for explicit hardware runs. API docs can be
opened locally with `cargo doc -p media --no-deps --open`.

## Working on the code

Keep public contracts independent of native library types. Place FFI and unsafe
resource handling at backend or interoperability boundaries, document each
invariant, and use narrowly scoped lint allowances only after review. The
workspace denies unsafe code by default.

Keep CPU development usable without GPU SDKs. Add native dependencies only
behind the appropriate backend boundary and document supported feature/platform
combinations. Keep portable and linked-native CI checks separate.

Add tests for observable semantics and lifetimes as behavior is introduced.
Update examples or documentation when the public API or supported behavior
changes. Keep pull requests focused, and describe the user-visible change and
how you tested it. Hardware results should identify the GPU and runtime versions.

## License

Contributions are accepted under the repository's [MIT license](LICENSE).
Preserve upstream notices when incorporating third-party code and document new
dependency or media licensing requirements in [THIRD_PARTY.md](THIRD_PARTY.md).
