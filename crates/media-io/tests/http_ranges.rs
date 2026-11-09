//! Local-only HTTP range protocol fixtures; no external requests or cloud access.
#![cfg(feature = "http")]
use media_io::{
    http::{HttpOptions, HttpRangeSource},
    index::VersionedIndex,
    range::{RangeCancellation, RangeOptions},
};
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Good,
    Ignore,
    Short,
    Long,
    WrongRange,
    Weak,
    Encoded,
    Redirect,
    ChangedTag,
    Slow,
}
struct Request {
    method: String,
    headers: BTreeMap<String, String>,
}
struct State {
    mode: Mode,
    version: String,
    body: Vec<u8>,
    requests: Vec<Request>,
    barrier: Option<Arc<Barrier>>,
}
struct Server {
    address: SocketAddr,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn new(mode: Mode) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let state = Arc::new(Mutex::new(State {
            mode,
            version: "\"v1\"".into(),
            body: b"abcdefghij".to_vec(),
            requests: Vec::new(),
            barrier: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let state_copy = state.clone();
        let stop_copy = stop.clone();
        let worker = thread::spawn(move || {
            let mut workers = Vec::new();
            for accepted in listener.incoming() {
                let Ok(stream) = accepted else {
                    break;
                };
                if stop_copy.load(Ordering::Acquire) {
                    break;
                }
                let state = state_copy.clone();
                workers.push(thread::spawn(move || serve(stream, state)));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            address,
            state,
            stop,
            worker: Some(worker),
        }
    }
    fn url(&self) -> String {
        format!("http://{}/object?signature=PRIVATE_QUERY", self.address)
    }
    fn mode(&self, mode: Mode) {
        self.state.lock().unwrap().mode = mode;
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect(self.address);
        self.worker.take().unwrap().join().unwrap();
    }
}
fn serve(mut stream: TcpStream, state: Arc<Mutex<State>>) {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut raw = Vec::new();
    let mut byte = [0];
    while !raw.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).ok() != Some(1) {
            return;
        }
        raw.push(byte[0]);
        assert!(raw.len() < 32768);
    }
    let text = String::from_utf8(raw).unwrap();
    let mut lines = text.split("\r\n");
    let method = lines
        .next()
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned();
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.trim().to_owned()))
        .collect::<BTreeMap<_, _>>();
    let (mode, version, body, barrier) = {
        let mut state = state.lock().unwrap();
        state.requests.push(Request {
            method: method.clone(),
            headers: headers.clone(),
        });
        (
            state.mode,
            state.version.clone(),
            state.body.clone(),
            state.barrier.clone(),
        )
    };
    let tag = if mode == Mode::Weak {
        "W/\"v1\""
    } else if mode == Mode::ChangedTag && method == "GET" {
        "\"v-other\""
    } else {
        &version
    };
    if mode == Mode::Redirect {
        let _=stream.write_all(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/private\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    }
    if method == "HEAD" {
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: {tag}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        return;
    }
    if headers.get("if-match") != Some(&version) {
        let _ = stream.write_all(
            b"HTTP/1.1 412 Precondition Failed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
        return;
    }
    if let Some(barrier) = barrier {
        barrier.wait();
    }
    let range = headers
        .get("range")
        .unwrap()
        .strip_prefix("bytes=")
        .unwrap()
        .split_once('-')
        .unwrap();
    let (start, end) = (
        range.0.parse::<usize>().unwrap(),
        range.1.parse::<usize>().unwrap(),
    );
    if body.is_empty() {
        let _ = write!(
            stream,
            "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */0\r\nETag: {tag}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        return;
    }
    if mode == Mode::Ignore {
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nETag: {tag}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(&body);
        return;
    }
    let requested = end - start + 1;
    let got_start = if mode == Mode::WrongRange {
        start + 1
    } else {
        start
    };
    let mut content = format!(
        "HTTP/1.1 206 Partial Content\r\nETag: {tag}\r\nContent-Range: bytes {got_start}-{end}/{}\r\nConnection: close\r\n",
        body.len()
    );
    if mode == Mode::Encoded {
        content.push_str("Content-Encoding: gzip\r\n");
    }
    if mode == Mode::Long {
        content.push_str("Transfer-Encoding: chunked\r\n\r\n");
        let _ = stream.write_all(content.as_bytes());
        let _ = write!(stream, "{:X}\r\n", requested + 1);
        let _ = stream.write_all(&body[start..=end]);
        let _ = stream.write_all(b"X\r\n0\r\n\r\n");
        return;
    }
    content.push_str(&format!("Content-Length: {requested}\r\n\r\n"));
    let _ = stream.write_all(content.as_bytes());
    if mode == Mode::Slow {
        thread::sleep(Duration::from_millis(250));
    }
    let length = if mode == Mode::Short {
        requested - 1
    } else {
        requested
    };
    let _ = stream.write_all(&body[start..start + length]);
}
fn options() -> HttpOptions {
    HttpOptions {
        range: RangeOptions {
            max_request_bytes: 3,
            cancellation: None,
        },
        ..HttpOptions::default()
    }
}

#[test]
fn verified_ranges_use_exact_conditionals_and_private_identity() {
    let server = Server::new(Mode::Good);
    let mut opts = options();
    opts.headers
        .insert("Authorization".into(), "Bearer PRIVATE_AUTH".into());
    let source = HttpRangeSource::connect(server.url(), opts.clone()).unwrap();
    assert_eq!(source.read_range(2, 7).unwrap(), b"cdefghi");
    let stats = source.range_statistics().unwrap();
    assert_eq!(stats.range_requests, 4);
    assert_eq!(stats.payload_bytes_received, 8);
    for request in &server.state.lock().unwrap().requests {
        assert_eq!(
            request.headers.get("accept-encoding").map(String::as_str),
            Some("identity")
        );
        if request.method == "GET" {
            assert_eq!(
                request.headers.get("if-match").map(String::as_str),
                Some("\"v1\"")
            );
        }
    }
    let debug = format!("{source:?} {:?} {opts:?}", source.fingerprint().unwrap());
    assert!(!debug.contains("PRIVATE_QUERY"));
    assert!(!debug.contains("PRIVATE_AUTH"));
    let mut other_opts = opts;
    other_opts
        .headers
        .insert("Authorization".into(), "Bearer DIFFERENT".into());
    let other = HttpRangeSource::connect(server.url(), other_opts).unwrap();
    assert_ne!(source.fingerprint().unwrap(), other.fingerprint().unwrap());
}

#[test]
fn independent_http_cursors_can_issue_concurrent_ranges() {
    let server = Server::new(Mode::Good);
    let source = HttpRangeSource::connect(server.url(), options()).unwrap();
    let mut first = source.open().unwrap();
    let mut second = source.open().unwrap();
    first.seek(SeekFrom::Start(2)).unwrap();
    second.seek(SeekFrom::Start(6)).unwrap();
    server.state.lock().unwrap().barrier = Some(Arc::new(Barrier::new(2)));
    thread::scope(|scope| {
        let a = scope.spawn(move || {
            let mut out = [0; 3];
            first.read_exact(&mut out).unwrap();
            out
        });
        let b = scope.spawn(move || {
            let mut out = [0; 3];
            second.read_exact(&mut out).unwrap();
            out
        });
        assert_eq!(&a.join().unwrap(), b"cde");
        assert_eq!(&b.join().unwrap(), b"ghi");
    });
}

#[test]
fn ignored_ranges_weak_etags_redirects_and_changed_response_validators_are_rejected() {
    for mode in [Mode::Ignore, Mode::Weak, Mode::Redirect, Mode::ChangedTag] {
        let server = Server::new(mode);
        let error = HttpRangeSource::connect(server.url(), options()).unwrap_err();
        let debug = format!("{error:?} {error}");
        assert!(!debug.contains("PRIVATE_QUERY"));
    }
}

#[test]
fn short_long_wrong_ranges_and_transformed_encodings_fail_without_padding() {
    for mode in [Mode::Short, Mode::Long, Mode::WrongRange, Mode::Encoded] {
        let server = Server::new(Mode::Good);
        let source = HttpRangeSource::connect(server.url(), options()).unwrap();
        let mut cursor = source.open().unwrap();
        server.mode(mode);
        let mut out = [77; 3];
        assert!(cursor.read(&mut out).is_err());
        assert_eq!(out, [77; 3]);
        assert_eq!(cursor.stream_position().unwrap(), 0);
        if mode == Mode::Long {
            assert_eq!(
                source.range_statistics().unwrap().payload_bytes_received,
                5,
                "probe1 + bounded extra-byte detection4"
            );
        }
    }
}

#[test]
fn replacing_an_object_invalidates_cursor_and_index_while_new_reads_pin_new_version() {
    let server = Server::new(Mode::Good);
    let source = HttpRangeSource::connect(server.url(), options()).unwrap();
    let index = VersionedIndex::new(&source, "0", "timing", [1]).unwrap();
    let mut cursor = source.open().unwrap();
    let mut prefix = [0; 3];
    cursor.read_exact(&mut prefix).unwrap();
    assert_eq!(&prefix, b"abc");
    {
        let mut state = server.state.lock().unwrap();
        state.version = "\"v2\"".into();
        state.body = b"ABCDEFGHIJ".to_vec();
    }
    assert!(cursor.read(&mut prefix).is_err());
    assert_eq!(cursor.stream_position().unwrap(), 3);
    assert!(index.get(&source, "0", "timing").unwrap().is_none());
    assert_eq!(source.read_range(0, 3).unwrap(), b"ABC");
}

#[test]
fn cancellation_and_body_timeout_are_bounded_without_full_spooling() {
    let server = Server::new(Mode::Good);
    let token = RangeCancellation::default();
    let mut opts = options();
    opts.range.cancellation = Some(token.clone());
    let source = HttpRangeSource::connect(server.url(), opts).unwrap();
    let mut cursor = source.open().unwrap();
    let before = source.range_statistics().unwrap().range_requests;
    token.cancel();
    assert!(cursor.read(&mut [0; 3]).is_err());
    assert_eq!(source.range_statistics().unwrap().range_requests, before);
    let opts = HttpOptions {
        request_timeout: Duration::from_millis(100),
        read_timeout: Duration::from_millis(100),
        ..options()
    };
    let source = HttpRangeSource::connect(server.url(), opts).unwrap();
    server.mode(Mode::Slow);
    assert!(source.read_range(0, 3).is_err());
}

#[test]
fn empty_objects_verify_unsatisfied_range_and_return_empty_reads() {
    let server = Server::new(Mode::Good);
    server.state.lock().unwrap().body.clear();
    let source = HttpRangeSource::connect(server.url(), options()).unwrap();
    assert_eq!(source.capabilities().unwrap().known_length, Some(0));
    assert_eq!(source.read_range(0, 0).unwrap(), Vec::<u8>::new());
    assert!(source.read_range(0, 1).is_err());
}
