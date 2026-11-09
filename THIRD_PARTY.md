# Third-party licensing

The [MIT license](LICENSE) covers media-rs code, documentation and locally
generated synthetic fixtures. Dependencies and externally downloaded media keep
their own licenses. This repository does not vendor Rust dependencies, FFmpeg,
or NVIDIA runtimes.

The following direct dependency versions come from the workspace `Cargo.lock`.
The license expressions are the packages' declared licenses; `OR` means a choice
of license. Optional features determine which dependencies an application uses.

| Dependency | Locked version | Declared license |
| --- | --- | --- |
| audioadapter-buffers | 5.2.0 | MIT OR Apache-2.0 |
| cudarc | 0.19.10 | MIT OR Apache-2.0 |
| ffmpeg-next | 9.0.0 | WTFPL |
| gif | 0.14.2 | MIT OR Apache-2.0 |
| image | 0.25.10 | MIT OR Apache-2.0 |
| image-webp | 0.2.4 | MIT OR Apache-2.0 |
| moxcms | 0.8.1 | BSD-3-Clause OR Apache-2.0 |
| ndarray | 0.17.2 | MIT OR Apache-2.0 |
| png | 0.18.1 | MIT OR Apache-2.0 |
| rubato | 5.0.1 | MIT OR Apache-2.0 |
| symphonia | 0.6.1 | MPL-2.0 |
| tiff | 0.11.3 | MIT |
| ureq | 3.5.1 | MIT OR Apache-2.0 |

## Symphonia

Symphonia and its codec/format crates use MPL-2.0. Its file-level copyleft allows
separate media-rs files to remain MIT-licensed. Distributors of binaries containing
Symphonia must tell recipients how to obtain the corresponding MPL-covered
source, including any modifications, and retain its license and notices. See
[Symphonia's source and license](https://github.com/pdeljanov/Symphonia) and
[Mozilla's MPL FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).

## FFmpeg

`ffmpeg-next` and `ffmpeg-sys-next` declare WTFPL for the Rust bindings. That does
not describe the linked FFmpeg libraries. FFmpeg is normally LGPL-2.1-or-later;
enabling GPL components changes the applicable terms, and `--enable-nonfree`
can produce an unredistributable build. A system FFmpeg package may include GPL
codecs such as libx264/libx265.

Before distributing a linked binary, inspect the actual FFmpeg build, preserve
required notices and corresponding source, and meet the applicable LGPL/GPL
linking and redistribution terms. MIT on this repository does not override them.
See [FFmpeg's license](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)
and [distribution guidance](https://www.ffmpeg.org/legal.html).

## NVIDIA runtimes

The Rust CUDA wrapper is permissively licensed. NVIDIA's CUDA toolkit, driver
and video runtime libraries have separate terms and are not included here. A
release that bundles NVIDIA components must follow their redistribution terms;
enabling `cuda` does not relicense those components. See the
[CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html).

## Other dependencies and media

Transitive dependencies also carry notices. Examples in the all-feature lockfile
include `ring` (Apache-2.0 AND ISC, with additional bundled-code notices),
`unicode-ident` ((MIT OR Apache-2.0) AND Unicode-3.0), `webpki-roots`
(CDLA-Permissive-2.0), and BSD, ISC and Zlib packages. Use the resolved dependency
graph for the target and features being shipped, and include the upstream license
texts and notices with binary distributions. This overview is not a complete
binary notice bundle.

The committed [audio fixtures](crates/media/tests/fixtures/README.md) are synthetic.
The optional [external corpus manifest](infra/bench/adoption-corpus.json) records
separate creator, source and Creative Commons license information for each item.
Those downloads live outside Git; redistributing them requires honoring their
individual attribution and share-alike terms where applicable.
