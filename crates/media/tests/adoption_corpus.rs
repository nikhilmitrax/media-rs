//! Opt-in real-media conformance. Large licensed files stay outside Git.
#![cfg(feature = "ffmpeg-audio")]
use media::{
    Session,
    backend::ffmpeg::codec::{ColorAssumption, VideoDecodeOptions},
    pipeline::session::CpuUsage,
    video_index::IndexBuildLimits,
};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufReader, Read},
    path::Path,
    process::{Command, Stdio},
};

#[test]
#[ignore = "requires explicitly downloaded, provenance-recorded MEDIA_ADOPTION_CORPUS"]
fn real_screencast_edited_film_and_camera_footage_match_cli_and_indexed_reads() {
    let directory = std::env::var_os("MEDIA_ADOPTION_CORPUS")
        .expect("set MEDIA_ADOPTION_CORPUS to the licensed corpus directory");
    let mut files: Vec<_> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension().is_some_and(|extension| {
                matches!(extension.to_str(), Some("mp4" | "mov" | "webm" | "mkv"))
            })
        })
        .collect();
    files.sort();
    assert!(
        files.len() >= 2,
        "the corpus must contain multiple authentic sources"
    );
    for path in files {
        check(&path);
    }
}

fn check(path: &Path) {
    let binary = std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into());
    let session = Session::default();
    let asset = session.open(path).unwrap();
    let request = asset.video().unwrap().decode_options(VideoDecodeOptions {
        // Untagged SD content uses an explicit interpretation matching the CLI.
        // Established source metadata still takes precedence on both paths.
        color_assumption: Some(ColorAssumption::Bt601Limited),
        ..Default::default()
    });
    let mut reader = request.reader().unwrap();
    let mut child = Command::new(&binary)
        .args(["-nostdin", "-v", "error", "-noautorotate", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-threads:v",
            "1",
            "-sws_flags",
            "bilinear",
            "-pix_fmt",
            "rgb24",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut expected = BTreeMap::new();
    let mut count = 0;
    while let Some(frame) = reader
        .next_frame()
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    {
        let pixels = frame.image.packed_bytes().unwrap();
        let mut reference = vec![0; pixels.len()];
        stdout.read_exact(&mut reference).unwrap();
        assert!(
            pixels == reference,
            "CLI RGB differs: {} frame {count}",
            path.display()
        );
        if count % 97 == 0 {
            expected.insert(count as u64, (frame.timing.pts.unwrap(), reference));
        }
        assert_eq!(frame.timing.ordinal, Some(count as u64));
        count += 1;
    }
    assert_eq!(stdout.read(&mut [0]).unwrap(), 0, "CLI has extra frames");
    assert!(child.wait().unwrap().success());
    assert!(count >= 30);
    drop(reader);
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    assert_eq!(index.len(), count);
    let entries: Vec<_> = expected.iter().rev().take(16).collect();
    let actual = request
        .clone()
        .with_index(index.clone())
        .sample(entries.iter().map(|(_, (pts, _))| *pts))
        .selection(media::pipeline::sampling::TemporalSelection::Nearest)
        .read()
        .unwrap_or_else(|error| panic!("indexed {}: {error}", path.display()));
    for (frame, (&ordinal, (_, pixels))) in actual.frames.iter().zip(entries) {
        assert_eq!(frame.timing.ordinal, Some(ordinal));
        assert_eq!(frame.image.packed_bytes().unwrap(), pixels);
    }
    drop((index, actual));
    eprintln!(
        "ADOPTION_CORPUS video={} frames={count} full_pixels_match=true indexed_match=true",
        path.display()
    );
    let strict_audio = asset.audio().unwrap().ffmpeg().raw_decoded().read();
    let rebuilt = path.file_name().unwrap() == "iphone5s-tokyo.webm";
    let audio = if rebuilt {
        let error = strict_audio.unwrap_err();
        assert_eq!(error.kind(), media::core::error::ErrorKind::MissingData);
        assert_eq!(
            error.context().time,
            Some(media::core::time::Time::from_millis(6594))
        );
        let request = asset.audio().unwrap().decoded_sample_clock();
        let mut stream = request.reader().unwrap();
        while stream.next_block().unwrap().is_some() {}
        assert!(
            stream
                .ffmpeg_decode_report()
                .unwrap()
                .reconstructed_timestamps
                > 0
        );
        request.read().unwrap()
    } else {
        strict_audio.unwrap_or_else(|error| panic!("audio {}: {error}", path.display()))
    };
    let reference = Command::new(&binary)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-threads:a", "1", "-f", "f32le", "-"])
        .output()
        .unwrap();
    assert!(reference.status.success());
    let samples: Vec<_> = reference
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    assert!(
        audio.samples() == samples,
        "audio differs from CLI: {}",
        path.display()
    );
    eprintln!(
        "ADOPTION_CORPUS audio={} frames={} full_samples_match=true clock_rebuilt={rebuilt}",
        path.display(),
        audio.frames()
    );
    drop(audio);
    assert_eq!(session.live_cpu_usage(), CpuUsage::default());
    assert_eq!(session.live_batches(), 0);
    assert_eq!(session.live_host_bytes(), 0);
}
