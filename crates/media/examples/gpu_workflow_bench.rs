//! Complete required-NVDEC → fused GPU preprocessing → independent GPU consumer.
//! All validation downloads occur after the timed GPU stage. See benches/README.md.
use media::core::device::Device;
use media::core::media::Orientation;
use media::pipeline::sampling::{FrameIndex, FrameIndexEntry};
use media::{
    backend::{
        cuda::{CudaContext, CudaLimits, CudaStream, CudaTensor, PreprocessSpec, preprocess_batch},
        ffmpeg::cuda::{CudaDecodeOptions, CudaVideoFrame, CudaVideoReader},
    },
    io::source::Source,
    prelude::*,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const DEVICE_BYTES: usize = 128 * 1024 * 1024;
const NATIVE_BYTES: usize = 64 * 1024 * 1024;
const HOST_BYTES: usize = 32 * 1024 * 1024;
const RETAINED_FRAMES: usize = 20;
fn cuda_limits() -> CudaLimits {
    CudaLimits {
        memory_bytes: DEVICE_BYTES,
        streams: 8,
    }
}
fn decode_options() -> CudaDecodeOptions {
    CudaDecodeOptions {
        max_frame_bytes: 2 * 1024 * 1024,
        max_retained_frames: RETAINED_FRAMES,
        max_retained_bytes: NATIVE_BYTES,
        ..Default::default()
    }
}
fn times(dense: bool) -> Vec<Time> {
    if dense {
        let mut values = (0..16)
            .rev()
            .map(|index| Time::new(index, 3).expect("fixed times"))
            .collect::<Vec<_>>();
        values.swap(2, 11);
        values
    } else {
        vec![
            Time::from_millis(4700),
            Time::from_millis(300),
            Time::from_millis(4700),
            Time::from_millis(2100),
        ]
    }
}
struct Counted {
    file: File,
    bytes: Arc<AtomicU64>,
}
impl Read for Counted {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = self.file.read(output)?;
        self.bytes.fetch_add(count as u64, Ordering::Relaxed);
        Ok(count)
    }
}
impl Seek for Counted {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.file.seek(position)
    }
}
fn source(path: &Path) -> Result<(Source, Arc<AtomicU64>)> {
    let path = fs::canonicalize(path).map_err(Error::io)?;
    let length = fs::metadata(&path).map_err(Error::io)?.len();
    let bytes = Arc::new(AtomicU64::new(0));
    let counter = bytes.clone();
    Ok((
        Source::from_seekable_factory(
            "immutable-generated-gpu-fixture",
            Some(format!("fixture-length:{length}")),
            Some(length),
            move || {
                Ok(Counted {
                    file: File::open(&path)?,
                    bytes: counter.clone(),
                })
            },
        )?,
        bytes,
    ))
}
#[derive(Debug, Default)]
struct NativeLive {
    bytes: AtomicUsize,
    peak: AtomicUsize,
}
#[derive(Debug)]
struct NativeLease {
    live: Arc<NativeLive>,
    bytes: usize,
}
impl NativeLease {
    fn new(live: &Arc<NativeLive>, bytes: usize) -> Self {
        let total = live.bytes.fetch_add(bytes, Ordering::AcqRel) + bytes;
        live.peak.fetch_max(total, Ordering::AcqRel);
        Self {
            live: live.clone(),
            bytes,
        }
    }
}
impl Drop for NativeLease {
    fn drop(&mut self) {
        self.live.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
struct Produced {
    tensor: CudaTensor,
    ordinals: Vec<u64>,
    decoded: u64,
    native_peak: usize,
    native_live: Option<Arc<NativeLive>>,
    crossings: u64,
}
fn facade(
    session: &Session,
    source: Source,
    times: &[Time],
    context: &CudaContext,
) -> Result<Produced> {
    let batch = session
        .from_source(source)
        .video()?
        .sample(times.iter().copied())
        .duration_inference(DurationInference::NextTimestamp)
        .cuda(0, cuda_limits())
        .on_context(context)
        .decode_options(decode_options())
        .to_tensor(PreprocessSpec::new(224, 224)?)
        .read()?;
    let selection = batch
        .selection
        .as_ref()
        .ok_or_else(|| Error::invalid("facade omitted exact selection provenance"))?;
    if !batch.report.hardware_decode {
        return Err(Error::invalid("benchmark requires real NVDEC"));
    }
    Ok(Produced {
        ordinals: selection
            .selections
            .iter()
            .map(|chosen| chosen.identity.ordinal)
            .collect(),
        tensor: batch.tensor,
        decoded: batch.report.decoded_frames,
        native_peak: session.peak_reserved_cuda_native_bytes(),
        native_live: None,
        crossings: batch.report.decoded_host_transfer_bytes,
    })
}
fn direct(source: &Source, times: &[Time], context: &CudaContext) -> Result<Produced> {
    let fingerprint = source.fingerprint()?;
    let mut reader = CudaVideoReader::open_on_context(source, context, decode_options())?;
    let track = reader.track();
    let maximum = *times
        .iter()
        .max()
        .ok_or_else(|| Error::invalid("empty GPU times"))?;
    let live = Arc::new(NativeLive::default());
    let mut previous: Option<(FrameIndexEntry, CudaVideoFrame)> = None;
    let mut selected: Vec<Option<(FrameIndexEntry, CudaVideoFrame)>> = vec![None; times.len()];
    while let Some(frame) = reader.next_frame()? {
        let bytes = frame.frame().image.storage().byte_len();
        let frame = frame.retain(NativeLease::new(&live, bytes));
        let entry = FrameIndexEntry::from_timing(frame.frame().timing, false)?;
        let mut entries = Vec::with_capacity(2);
        if let Some((entry, _)) = &previous {
            entries.push(*entry);
        }
        entries.push(entry);
        let pair = FrameIndex::new(fingerprint.clone(), track.0, entries)?;
        for (index, time) in times.iter().enumerate() {
            if let Ok(chosen) = pair.select(
                *time,
                TemporalSelection::DisplayedAt,
                DurationInference::NextTimestamp,
            ) {
                if chosen.identity.ordinal == entry.ordinal {
                    selected[index] = Some((entry, frame.clone()));
                } else if let Some((last, last_frame)) = &previous
                    && chosen.identity.ordinal == last.ordinal
                {
                    selected[index] = Some((*last, last_frame.clone()));
                }
            }
        }
        previous = Some((entry, frame));
        if entry.pts > maximum {
            break;
        }
    }
    let decoded = reader.frames_decoded();
    drop(reader);
    drop(previous);
    let mut ordinals = Vec::with_capacity(times.len());
    let mut inputs = Vec::with_capacity(times.len());
    for frame in selected {
        let (entry, frame) = frame.ok_or_else(|| {
            Error::unsupported("direct scheduler could not resolve the requested display interval")
        })?;
        let metadata = frame.frame().image.metadata();
        let color = &metadata.color;
        if metadata.orientation != Orientation::Normal
            || metadata.icc_profile.is_some()
            || color.primaries.as_deref() != Some("bt709")
            || color.transfer.as_deref() != Some("bt709")
            || color.matrix.as_deref() != Some("bt709")
            || color.range.as_deref() != Some("tv")
            || color.chroma_siting.as_deref() != Some("left")
        {
            return Err(Error::incompatible(
                "fixture metadata differs from declared BT709/normal/non-ICC benchmark semantics",
            ));
        }
        ordinals.push(entry.ordinal);
        inputs.push(frame.yuv_view()?);
    }
    let stream = context.stream()?;
    let tensor = preprocess_batch(&inputs, &stream, PreprocessSpec::new(224, 224)?)?;
    drop(inputs);
    drop(stream);
    Ok(Produced {
        tensor,
        ordinals,
        decoded,
        native_peak: live.peak.load(Ordering::Acquire),
        native_live: Some(live),
        crossings: 0,
    })
}
#[derive(Clone, Copy)]
enum Export {
    Native,
    Dlpack,
}
impl Export {
    fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Dlpack => "dlpack",
        }
    }
}
struct Observation {
    ready: u128,
    consumer: u128,
    retired: u128,
    validation: u128,
    decoded: u64,
    source_bytes: u64,
    ordinals: Vec<u64>,
    values: Vec<f32>,
    digest: u64,
    host_peak: Option<usize>,
    native_peak: usize,
    owned_peak: usize,
    validation_bytes: usize,
    consumer_output_bytes: usize,
}
fn wait_retired(
    session: &Session,
    context: &CudaContext,
    live: Option<&Arc<NativeLive>>,
) -> Result<()> {
    let start = Instant::now();
    while context.memory_used() != 0
        || session.live_cuda_codec_sessions() != 0
        || session.reserved_cuda_native_bytes() != 0
        || live.is_some_and(|live| live.bytes.load(Ordering::Acquire) != 0)
    {
        if start.elapsed() > Duration::from_secs(5) {
            return Err(Error::exhausted(
                "GPU benchmark input/session retirement did not finish after completion",
            ));
        }
        std::thread::yield_now();
    }
    context.check_errors()
}
fn workflow(
    path: &Path,
    session: &Session,
    context: &CudaContext,
    consumer: &consumer::Consumer,
    times: &[Time],
    facade_variant: bool,
    export: Export,
) -> Result<Observation> {
    wait_retired(session, context, None)?;
    let start = Instant::now();
    let (source, counter) = source(path)?;
    let produced = if facade_variant {
        facade(session, source, times, context)?
    } else {
        direct(&source, times, context)?
    };
    if produced.crossings != 0 {
        return Err(Error::invalid("forbidden decoded host crossing observed"));
    }
    if produced.tensor.shape() != [times.len(), 3, 224, 224] {
        return Err(Error::incompatible("incorrect GPU batch shape"));
    }
    produced.tensor.buffer().readiness().wait()?;
    let ready = start.elapsed().as_nanos();
    let output = consumer.consume(produced.tensor, export)?;
    let completed = start.elapsed().as_nanos();
    wait_retired(session, context, produced.native_live.as_ref())?;
    let retired = start.elapsed().as_nanos();
    let validation_start = Instant::now();
    let values = consumer.download(&output)?;
    let validation = validation_start.elapsed().as_nanos();
    let validation_bytes = values.len() * 4;
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.25..=0.75).contains(value))
    {
        return Err(Error::incompatible(
            "consumer validation found invalid normalized RGB values",
        ));
    }
    let mut digest = 0xcbf29ce484222325u64;
    for bytes in produced
        .ordinals
        .iter()
        .flat_map(|ordinal| ordinal.to_le_bytes())
        .chain(
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes()),
        )
    {
        digest = (digest ^ u64::from(bytes)).wrapping_mul(0x100000001b3);
    }
    drop(output);
    context.check_errors()?;
    Ok(Observation {
        ready,
        consumer: completed,
        retired,
        validation,
        decoded: produced.decoded,
        source_bytes: counter.load(Ordering::Relaxed),
        ordinals: produced.ordinals,
        values,
        digest,
        host_peak: facade_variant.then(|| session.peak_host_bytes()),
        native_peak: produced.native_peak,
        owned_peak: context.memory_peak(),
        validation_bytes,
        consumer_output_bytes: validation_bytes,
    })
}

/// Independent foreign consumer only sees native descriptors/device addresses.
#[allow(unsafe_code)]
mod consumer {
    use super::*;
    use cudarc::{
        driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
        nvrtc::{CompileOptions, compile_ptx_with_opts},
    };
    use media::interop::{
        dlpack::{self, DLManagedTensorVersioned},
        native::export_cuda_tensor,
    };
    pub struct Consumer {
        stream: CudaStream,
        function: CudaFunction,
    }
    const KERNEL: &str = r#"extern "C" __global__ void consume(const float* input,float* output,unsigned int n) {unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=input[i]*0.5f+0.25f;}"#;
    impl Consumer {
        pub fn new(context: &CudaContext) -> Result<Self> {
            let ptx = compile_ptx_with_opts(
                KERNEL,
                CompileOptions {
                    arch: Some("compute_75"),
                    ..Default::default()
                },
            )
            .map_err(|error| Error::backend("nvrtc", error.to_string()))?;
            let module = context
                .native_context()
                .load_module(ptx)
                .map_err(cuda_error)?;
            Ok(Self {
                stream: context.stream()?,
                function: module.load_function("consume").map_err(cuda_error)?,
            })
        }
        fn launch(&self, address: u64, elements: usize) -> Result<CudaSlice<f32>> {
            let count =
                u32::try_from(elements).map_err(|_| Error::overflow("consumer launch size"))?;
            let mut output = self
                .stream
                .native_stream()
                .alloc_zeros::<f32>(elements)
                .map_err(cuda_error)?;
            let mut launch = self.stream.native_stream().launch_builder(&self.function);
            launch.arg(&address).arg(&mut output).arg(&count);
            // SAFETY: address belongs to a checked live native/DLPack owner in
            // this exact context; export ordered producer work onto this queue.
            // Both arrays cover elements and the kernel writes only its own output.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.map_err(cuda_error)?;
            drop(launch);
            Ok(output)
        }
        pub fn consume(&self, tensor: CudaTensor, export: Export) -> Result<CudaSlice<f32>> {
            let elements = tensor.shape().iter().product();
            match export {
                Export::Native => {
                    let owner = export_cuda_tensor(tensor, &self.stream, CopyPolicy::Forbid)?;
                    let descriptor = owner.descriptor();
                    if descriptor.device() != self.stream.context().device()
                        || descriptor.byte_len() != elements * 4
                    {
                        return Err(Error::incompatible("native consumer descriptor mismatch"));
                    }
                    let output = self.launch(descriptor.address(), elements)?;
                    let completion = self.stream.record_completion()?;
                    drop(owner);
                    completion.wait()?;
                    Ok(output)
                }
                Export::Dlpack => {
                    let owner = Foreign(
                        dlpack::export_cuda_versioned(tensor, &self.stream, CopyPolicy::Forbid)?
                            .into_raw(),
                    );
                    // SAFETY: original versioned descriptor is exclusively held
                    // by this guard until its matching deleter below; shape/data
                    // remain valid and all GPU uses are on the selected stream.
                    let address = unsafe {
                        let managed = &*owner.0;
                        let view = &managed.dl_tensor;
                        let Device::Cuda { ordinal, .. } = self.stream.context().device() else {
                            return Err(Error::invalid("consumer context is not CUDA"));
                        };
                        if managed.version != (dlpack::DLPackVersion { major: 1, minor: 3 })
                            || managed.deleter.is_none()
                            || managed.flags != 0
                            || view.device.device_type != 2
                            || view.device.device_id != ordinal as i32
                            || view.dtype.code != 2
                            || view.dtype.bits != 32
                            || view.dtype.lanes != 1
                            || view.ndim != 4
                            || view.byte_offset != 0
                            || view.data.is_null()
                            || view.shape.is_null()
                        {
                            return Err(Error::incompatible("DLPack consumer descriptor mismatch"));
                        }
                        let shape = std::slice::from_raw_parts(view.shape, 4);
                        if shape[1..] != [3, 224, 224]
                            || shape.iter().any(|dimension| *dimension <= 0)
                            || shape.iter().product::<i64>() as usize != elements
                        {
                            return Err(Error::incompatible("DLPack consumer shape mismatch"));
                        }
                        if !view.strides.is_null()
                            && std::slice::from_raw_parts(view.strides, 4)
                                != [3 * 224 * 224, 224 * 224, 224, 1]
                        {
                            return Err(Error::incompatible(
                                "DLPack consumer requires contiguous NCHW strides",
                            ));
                        }
                        view.data as usize as u64
                    };
                    let output = self.launch(address, elements)?;
                    let completion = self.stream.record_completion()?;
                    drop(owner);
                    completion.wait()?;
                    Ok(output)
                }
            }
        }
        pub fn download(&self, output: &CudaSlice<f32>) -> Result<Vec<f32>> {
            self.stream
                .native_stream()
                .clone_dtoh(output)
                .map_err(cuda_error)
        }
    }
    struct Foreign(*mut DLManagedTensorVersioned);
    impl Drop for Foreign {
        fn drop(&mut self) {
            // SAFETY: this guard owns one original live versioned export; all
            // submissions are queued on its selected queue before release, and
            // the matching deleter records/waits that queue and runs exactly once.
            unsafe {
                if let Some(deleter) = (*self.0).deleter {
                    deleter(self.0);
                }
            }
        }
    }
    fn cuda_error(error: cudarc::driver::DriverError) -> Error {
        Error::backend("cuda-consumer", error.to_string()).with_cause(error)
    }
}
fn environment(directory: &Path, input: &Path, iterations: usize) -> Result<()> {
    fs::create_dir_all(directory).map_err(Error::io)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("environment.txt"))
        .map_err(Error::io)?;
    writeln!(file,"unix_seconds={}\niterations={}\narch={}\nos={}\nbuild_profile=release\nowned_cuda_limit={}\nknown_native_limit={}\nretained_frames={}\nhost_limit={}\nselection=DisplayedAt\nduration_inference=NextTimestamp\nnative_duration_policy=Unknown\noutput=NCHW_RGB_f32_224x224\nconsumer=out[i]=in[i]*0.5+0.25\nvalidation_downloads=outside_GPU_stage\nopaque_driver_copies=unmeasured\npeak_scopes=context_owned_and_facade_native_host_cumulative_direct_native_execution\nconsumer_output=native_CUDA_allocation_outside_library_budget_but_known_nbytes\nOS_file_cache=not_cold_controlled",SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error|Error::backend("clock",error.to_string()))?.as_secs(),iterations,std::env::consts::ARCH,std::env::consts::OS,DEVICE_BYTES,NATIVE_BYTES,RETAINED_FRAMES,HOST_BYTES).map_err(Error::io)?;
    for (command, args) in [
        ("rustc", vec!["-Vv"]),
        ("uname", vec!["-a"]),
        (
            "nvidia-smi",
            vec![
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
        ),
    ] {
        match Command::new(command).args(args).output() {
            Ok(output) => writeln!(
                file,
                "\n{command}: {:?}\n{}{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
            .map_err(Error::io)?,
            Err(error) => writeln!(file, "{command}: unavailable: {error}").map_err(Error::io)?,
        }
    }
    fs::write(
        directory.join("runner.rs"),
        include_bytes!("gpu_workflow_bench.rs"),
    )
    .map_err(Error::io)?;
    fs::write(
        directory.join("Cargo.lock"),
        include_bytes!("../../../Cargo.lock"),
    )
    .map_err(Error::io)?;
    fs::copy(input, directory.join("fixture.mp4")).map_err(Error::io)?;
    let manifest = input
        .parent()
        .ok_or_else(|| Error::invalid("fixture parent missing"))?
        .join("manifest.json");
    fs::copy(manifest, directory.join("fixture-manifest.json")).map_err(Error::io)?;
    Ok(())
}
fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid(
            "GPU workflow timings require a release build",
        ));
    }
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().ok_or_else(|| {
        Error::invalid("usage: gpu_workflow_bench INPUT OUTPUT_DIRECTORY [ITERATIONS=7]")
    })?);
    let directory = PathBuf::from(
        args.next()
            .ok_or_else(|| Error::invalid("output directory required"))?,
    );
    let iterations = args
        .next()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| Error::invalid("invalid iteration count"))
        })
        .transpose()?
        .unwrap_or(7);
    if !(2..=100).contains(&iterations) {
        return Err(Error::invalid(
            "need an initial use and 1..99 measured warm iterations",
        ));
    }
    environment(&directory, &input, iterations)?;
    let mut setup = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("setup.csv"))
        .map_err(Error::io)?;
    writeln!(setup, "stage,ns").map_err(Error::io)?;
    let start = Instant::now();
    let session = Session::builder()
        .limits(Limits::new().host_memory(HOST_BYTES))
        .cuda_limits(media::cuda_session::CudaSessionLimits {
            max_codec_sessions: 2,
            max_native_bytes: NATIVE_BYTES,
        })
        .build()?;
    let context = session.cuda_context(0, cuda_limits())?;
    writeln!(
        setup,
        "canonical_context_bootstrap,{}",
        start.elapsed().as_nanos()
    )
    .map_err(Error::io)?;
    let start = Instant::now();
    let consumer = consumer::Consumer::new(&context)?;
    writeln!(
        setup,
        "independent_consumer_NVRTC_compile_and_module_load,{}",
        start.elapsed().as_nanos()
    )
    .map_err(Error::io)?;
    let mut csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("raw.csv"))
        .map_err(Error::io)?;
    writeln!(csv,"sampling,export,variant,iteration,phase,within_pair_order,tensor_ready_ns,consumer_complete_ns,input_retired_ns,validation_ns,outputs,decoded_frames,source_payload_bytes,context_owned_peak_cumulative_bytes,native_peak_bytes,native_peak_scope,host_peak_cumulative_bytes,consumer_output_native_bytes,validation_download_bytes,decoded_host_crossings_GPU_stage,driver_crossings,digest_fnv64,validation").map_err(Error::io)?;
    let mut first = true;
    for dense in [false, true] {
        let requested = times(dense);
        for export in [Export::Native, Export::Dlpack] {
            for iteration in 0..iterations {
                let order = if iteration % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                };
                let mut pair = Vec::new();
                for (position, facade_variant) in order.into_iter().enumerate() {
                    let observation = workflow(
                        &input,
                        &session,
                        &context,
                        &consumer,
                        &requested,
                        facade_variant,
                        export,
                    )?;
                    let phase = if first {
                        first = false;
                        "initial_includes_first_preprocess_NVRTC"
                    } else if iteration == 0 {
                        "initial_shared_module_cache"
                    } else {
                        "steady"
                    };
                    pair.push((facade_variant, position, phase, observation));
                }
                let expected = requested
                    .iter()
                    .map(|time| {
                        time.to_ticks(
                            TimeBase::new(1, 30).expect("fixed frame grid"),
                            media::core::time::Rounding::Floor,
                        )
                        .map(|ordinal| ordinal as u64)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if pair[0].3.ordinals != expected
                    || pair[0].3.decoded != pair[1].3.decoded
                    || pair[0].3.ordinals != pair[1].3.ordinals
                    || pair[0]
                        .3
                        .values
                        .iter()
                        .map(|value| value.to_bits())
                        .ne(pair[1].3.values.iter().map(|value| value.to_bits()))
                {
                    return Err(Error::incompatible(
                        "facade/direct GPU selection or consumer output mismatch; no validated timing row emitted",
                    ));
                }
                for (facade_variant, position, phase, observation) in pair {
                    let host = observation
                        .host_peak
                        .map_or_else(|| "unmeasured".into(), |bytes| bytes.to_string());
                    writeln!(csv,"{},{},{},{iteration},{phase},{position},{},{},{},{},{},{},{},{},{},{},{host},{},{},0,unmeasured,{:016x},passed",if dense{"dense16"}else{"sparse4_repeated"},export.name(),if facade_variant{"facade"}else{"direct"},observation.ready,observation.consumer,observation.retired,observation.validation,observation.ordinals.len(),observation.decoded,observation.source_bytes,observation.owned_peak,observation.native_peak,if facade_variant{"session_cumulative"}else{"execution_lease_peak"},observation.consumer_output_bytes,observation.validation_bytes,observation.digest).map_err(Error::io)?;
                }
                csv.flush().map_err(Error::io)?;
            }
        }
    }
    println!("{}", directory.join("raw.csv").display());
    Ok(())
}
