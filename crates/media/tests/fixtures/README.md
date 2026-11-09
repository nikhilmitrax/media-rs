# Synthetic audio timing fixtures

These files were generated locally with FFmpeg 8.1 from its mathematical sine
source. They contain no external recordings and are covered by the repository's
[MIT license](../../../../LICENSE). Each holds 1,440 mono f32 samples
at 48 kHz in Matroska, whose packet timestamps use a 1 ms time base. Tests consume
the committed files and do not require FFmpeg to be installed.

| File | Packet timestamps | Purpose | SHA-256 |
| --- | --- | --- | --- |
| audio_quantized.mka | 0, 21 ms | 1,024 sample frames span 21 1/3 ms, demonstrating container timestamp quantization | b63c88d39ee4badc6d5ab71b937963928a255a09b887aa26a35ae5b6a36cee63 |
| audio_gap.mka | 0, 41 ms | A real 20 ms discontinuity exceeds the permitted quantization bin | ef1b176d470025705f2f42daf9163ce60f3454c228b95d9f9d09e3d4d7e2d30c |
| audio_offset.mka | 200, 221 ms | Nonzero track origin must be established before constructing the continuous resampler | b07d0819dc40bd851fd41ebf2a315c9570859444c31c8e34fab0c54bb6fff0cf |

Generate from this directory with:

```sh
ffmpeg -nostdin -hide_banner -loglevel error -y -fflags +bitexact -f lavfi -i sine=frequency=440:sample_rate=48000:duration=0.03 -map_metadata -1 -c:a pcm_f32le -flags:a +bitexact audio_quantized.mka
ffmpeg -nostdin -hide_banner -loglevel error -y -fflags +bitexact -f lavfi -i sine=frequency=440:sample_rate=48000:duration=0.03 -af 'asetpts=PTS+gte(T\,0.01)*0.02/TB' -map_metadata -1 -c:a pcm_f32le -flags:a +bitexact audio_gap.mka
ffmpeg -nostdin -hide_banner -loglevel error -y -fflags +bitexact -f lavfi -i sine=frequency=440:sample_rate=48000:duration=0.03 -af 'asetpts=PTS+0.2/TB' -map_metadata -1 -c:a pcm_f32le -flags:a +bitexact audio_offset.mka
```

Muxer metadata and therefore binary hashes may differ with a different FFmpeg
build. The committed hashes describe the inspected artifacts; the tests verify
decoded sample counts, packet/source timing, and continuous output semantics.
