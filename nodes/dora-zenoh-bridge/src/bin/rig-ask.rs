//! rig-ask: puts one spatial query on `rig/spatial_query` and prints each revision of its spatial answer from
//! `rig/spatial_answer`, until the final one. It only asks: it publishes no nav goals and no goal statuses.
//!
//! rig-ask --connect tcp/192.0.2.10:7447 "where is the couch?"
//!
//! The answers come from the Zenoh subscriber's callback through a channel of payloads (`follow_answers`), so
//! everything after the Zenoh session runs without a network in the tests.

use std::fmt;
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use dora_node_api::arrow::array::StructArray;
use dora_zenoh_bridge::payload;
use perception_wire::navigation::{
    decode_spatial_answer, encode_spatial_query, Asker, QuestionId, Reference, Referent, Revision, SpatialAnswer,
    SpatialQuery, UnixNanos,
};
use serde::Serialize;
use zenoh::qos::CongestionControl;
use zenoh::Wait;

const QUESTION_KEY: &str = "rig/spatial_query";
const ANSWER_KEY: &str = "rig/spatial_answer";

/// The bridge's payload for a message.
fn encode_payload(message: &StructArray) -> eyre::Result<Vec<u8>> {
    payload::PayloadEncoder::new()?.encode(message)
}

/// The message in a bridge payload.
fn decode_payload(bytes: &[u8]) -> eyre::Result<StructArray> {
    payload::PayloadDecoder::new()?.decode(bytes)
}

#[derive(Parser, Debug)]
#[command(name = "rig-ask", about = "Asks the robot's spatial agent one question and prints each revision of its answer")]
struct Arguments {
    /// The question; `-` reads it from stdin.
    question: String,
    /// A Zenoh endpoint to connect to, e.g. tcp/192.0.2.10:7447. Repeatable; ZENOH_CONNECT takes a comma-separated
    /// list. One is required.
    #[arg(long = "connect", value_name = "ENDPOINT", env = "ZENOH_CONNECT", value_delimiter = ',')]
    connect: Vec<String>,
    /// A Zenoh endpoint to listen on, e.g. tcp/192.0.2.20:0. Connected to the camera machine's bridge, rig-ask must
    /// listen on its own LAN address: the server's bridge learns it through gossip and connects to it. Repeatable;
    /// ZENOH_LISTEN takes a comma-separated list.
    #[arg(long = "listen", value_name = "ENDPOINT", env = "ZENOH_LISTEN", value_delimiter = ',')]
    listen: Vec<String>,
    /// Whom the question is asked on behalf of [default: the USER environment variable].
    #[arg(long = "asker", value_name = "NAME")]
    caller: Option<String>,
    /// How long the spatial agent has for its first answer after the question goes out.
    #[arg(long = "first-answer-within", value_name = "SECONDS", default_value = "60", value_parser = seconds)]
    first_answer_within: Duration,
    /// How long the spatial agent has for its final answer after the question goes out.
    #[arg(long = "final-answer-within", value_name = "SECONDS", default_value = "180", value_parser = seconds)]
    final_answer_within: Duration,
    /// How long to wait for a subscriber on rig/spatial_query before giving up.
    #[arg(long = "subscriber-within", value_name = "SECONDS", default_value = "10", value_parser = seconds)]
    subscriber_within: Duration,
    /// One JSON object per revision, one per line, instead of text.
    #[arg(long = "json")]
    json: bool,
}

fn seconds(value: &str) -> Result<Duration, String> {
    let seconds: f64 = value.parse().map_err(|_| format!("{value:?} is not a number of seconds"))?;
    if seconds.is_nan() || seconds <= 0.0 {
        return Err(format!("{value} s is not a positive duration"));
    }
    Duration::try_from_secs_f64(seconds).map_err(|e| format!("{value} s: {e}"))
}

/// The person or agent rig-ask asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Caller(String);

impl Caller {
    fn asker(&self) -> Asker {
        Asker(format!("rig-ask for {}", self.0))
    }
}

/// When the answers are due, measured from when the question went out.
#[derive(Clone, Copy, Debug)]
struct AnswerDeadlines {
    first: Duration,
    last: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Text,
    JsonLines,
}

#[derive(Debug, thiserror::Error)]
enum AskError {
    #[error("{0}")]
    Usage(String),
    #[error("zenoh: {0}")]
    Zenoh(String),
    #[error("no spatial agent is listening on {QUESTION_KEY}: no subscriber matched within {} s", .within.as_secs_f64())]
    NoSpatialAgent { within: Duration },
    #[error("no first answer within {} s of asking {question}", .within.as_secs_f64())]
    FirstAnswerLate { question: QuestionId, within: Duration },
    #[error("no final answer within {} s of asking {question}; {received} revisions came", .within.as_secs_f64())]
    FinalAnswerLate { question: QuestionId, within: Duration, received: u32 },
    #[error("a message on {ANSWER_KEY} does not decode: {0}")]
    UndecodableAnswer(String),
    #[error("the subscriber on {ANSWER_KEY} stopped")]
    SubscriberStopped,
    #[error("writing the answer: {0}")]
    Output(#[from] io::Error),
}

impl AskError {
    /// The process exit status: 0 is the final answer; each failure has its own.
    fn exit_status(&self) -> u8 {
        match self {
            AskError::Zenoh(_) | AskError::SubscriberStopped | AskError::Output(_) => 1,
            AskError::Usage(_) => 2,
            AskError::NoSpatialAgent { .. } => 3,
            AskError::FirstAnswerLate { .. } => 4,
            AskError::FinalAnswerLate { .. } => 5,
            AskError::UndecodableAnswer(_) => 6,
        }
    }
}

fn zenoh_error(doing: &str) -> impl FnOnce(zenoh::Error) -> AskError + '_ {
    move |error| AskError::Zenoh(format!("{doing}: {error}"))
}

fn question_text(argument: &str, stdin: &mut impl Read) -> Result<String, AskError> {
    let text = if argument == "-" {
        let mut text = String::new();
        stdin.read_to_string(&mut text).map_err(|e| AskError::Usage(format!("reading the question from stdin: {e}")))?;
        text
    } else {
        argument.to_owned()
    };
    let text = text.trim();
    if text.is_empty() {
        return Err(AskError::Usage("the question is empty".to_owned()));
    }
    Ok(text.to_owned())
}

fn caller(named: Option<String>) -> Result<Caller, AskError> {
    match named {
        Some(name) if !name.trim().is_empty() => Ok(Caller(name.trim().to_owned())),
        Some(_) => Err(AskError::Usage("--asker is empty".to_owned())),
        None => std::env::var("USER")
            .ok()
            .filter(|name| !name.is_empty())
            .map(Caller)
            .ok_or_else(|| AskError::Usage("USER is not set; name whom the question is for with --asker".to_owned())),
    }
}

/// A question whose id is the unix time in nanoseconds when it was asked, so runs do not collide.
fn question(text: String, caller: &Caller, asked: SystemTime) -> SpatialQuery {
    let since_epoch = asked.duration_since(UNIX_EPOCH).expect("the clock is after 1970");
    let nanos = u64::try_from(since_epoch.as_nanos()).expect("unix nanoseconds fit in u64 until 2554");
    SpatialQuery {
        question_id: QuestionId(nanos),
        asked: UnixNanos(i64::try_from(nanos).expect("unix nanoseconds fit in i64 until 2262")),
        asker: caller.asker(),
        text,
    }
}

#[derive(Serialize)]
struct ReferenceLine {
    kind: &'static str,
    id: Option<u64>,
    x_m: f32,
    y_m: f32,
}

#[derive(Serialize)]
struct AnswerLine<'a> {
    question_id: u64,
    revision: u32,
    is_final: bool,
    model: &'a str,
    text: &'a str,
    references: Vec<ReferenceLine>,
    goal_id: Option<u64>,
}

fn referent_id(referent: Referent) -> Option<u64> {
    match referent {
        Referent::Object(id) => Some(id.0),
        Referent::Place(id) => Some(id.0),
        Referent::Room(id) => Some(id.0),
        Referent::Track(id) => Some(id.0),
        Referent::Pose => None,
    }
}

struct ReferenceText<'a>(&'a Reference);

impl fmt::Display for ReferenceText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Reference { referent, x_m, y_m } = self.0;
        match referent {
            Referent::Object(id) => write!(f, "object {id}")?,
            Referent::Place(id) => write!(f, "place {id}")?,
            Referent::Room(id) => write!(f, "room {id}")?,
            Referent::Track(id) => write!(f, "person track {id}")?,
            Referent::Pose => write!(f, "map point")?,
        }
        write!(f, " at x {x_m:.2} m, y {y_m:.2} m")
    }
}

fn write_answer(answer: &SpatialAnswer, format: OutputFormat, out: &mut impl Write) -> io::Result<()> {
    match format {
        OutputFormat::Text => {
            let stage = match (answer.revision, answer.is_final) {
                (_, true) => " (final)",
                (Revision::FIRST, false) => " (first)",
                _ => "",
            };
            writeln!(out, "revision {}{stage} from {}: {}", answer.revision.0, answer.model, answer.text)?;
            for reference in &answer.references {
                writeln!(out, "  {}", ReferenceText(reference))?;
            }
            if let Some(goal) = answer.goal_id {
                writeln!(out, "  NAV GOAL {goal} issued by this answer: the robot may be moving")?;
            }
        }
        OutputFormat::JsonLines => {
            let line = AnswerLine {
                question_id: answer.question_id.0,
                revision: answer.revision.0,
                is_final: answer.is_final,
                model: &answer.model,
                text: &answer.text,
                references: answer
                    .references
                    .iter()
                    .map(|r| ReferenceLine { kind: r.referent.kind().wire_name(), id: referent_id(r.referent), x_m: r.x_m, y_m: r.y_m })
                    .collect(),
                goal_id: answer.goal_id.map(|goal| goal.0),
            };
            serde_json::to_writer(&mut *out, &line).map_err(io::Error::other)?;
            writeln!(out)?;
        }
    }
    out.flush()
}

/// Prints each answer to `question` from `answers` (payloads from rig/spatial_answer) until the final one, and
/// returns how many revisions came. Answers to other questions are skipped; every payload must decode.
fn follow_answers(
    answers: &Receiver<Vec<u8>>,
    question: QuestionId,
    asked_at: Instant,
    deadlines: AnswerDeadlines,
    format: OutputFormat,
    out: &mut impl Write,
) -> Result<u32, AskError> {
    let mut received = 0;
    loop {
        let (due, late) = if received == 0 {
            (deadlines.first, AskError::FirstAnswerLate { question, within: deadlines.first })
        } else {
            (deadlines.last, AskError::FinalAnswerLate { question, within: deadlines.last, received })
        };
        let bytes = match answers.recv_timeout((asked_at + due).saturating_duration_since(Instant::now())) {
            Ok(bytes) => bytes,
            Err(RecvTimeoutError::Timeout) => return Err(late),
            Err(RecvTimeoutError::Disconnected) => return Err(AskError::SubscriberStopped),
        };
        let answer = decode_payload(&bytes)
            .and_then(|message| decode_spatial_answer(&message).map_err(eyre::Report::new))
            .map_err(|e| AskError::UndecodableAnswer(format!("{e:#}")))?;
        if answer.question_id != question {
            continue;
        }
        received += 1;
        write_answer(&answer, format, out)?;
        if answer.is_final {
            return Ok(received);
        }
    }
}

fn endpoint_list(endpoints: &[String]) -> String {
    let quoted: Vec<String> = endpoints.iter().map(|e| e.trim()).filter(|e| !e.is_empty()).map(|e| format!("{e:?}")).collect();
    format!("[{}]", quoted.join(","))
}

fn open_session(connect: &[String], listen: &[String]) -> Result<zenoh::Session, AskError> {
    let mut config = zenoh::Config::default();
    config.insert_json5("listen/endpoints", &endpoint_list(listen)).map_err(|e| AskError::Usage(format!("--listen: {e}")))?;
    config.insert_json5("connect/endpoints", &endpoint_list(connect)).map_err(|e| AskError::Usage(format!("--connect: {e}")))?;
    zenoh::open(config).wait().map_err(zenoh_error("opening the session"))
}

/// Waits until a subscriber matches `publisher`.
fn wait_for_subscriber(publisher: &zenoh::pubsub::Publisher<'_>, within: Duration) -> Result<(), AskError> {
    let deadline = Instant::now() + within;
    let listener = publisher.matching_listener().wait().map_err(zenoh_error("declaring the matching listener"))?;
    if publisher.matching_status().wait().map_err(zenoh_error("reading the matching status"))?.matching() {
        return Ok(());
    }
    loop {
        match listener.recv_deadline(deadline).map_err(zenoh_error("waiting for a matching subscriber"))? {
            Some(status) if status.matching() => return Ok(()),
            Some(_) => continue,
            None => return Err(AskError::NoSpatialAgent { within }),
        }
    }
}

fn ask(arguments: Arguments) -> Result<(), AskError> {
    if arguments.connect.iter().all(|e| e.trim().is_empty()) {
        return Err(AskError::Usage(
            "name a Zenoh endpoint to connect to with --connect or ZENOH_CONNECT, e.g. tcp/192.0.2.10:7447".to_owned(),
        ));
    }
    if arguments.final_answer_within < arguments.first_answer_within {
        return Err(AskError::Usage("--final-answer-within is shorter than --first-answer-within".to_owned()));
    }
    let text = question_text(&arguments.question, &mut io::stdin())?;
    let caller = caller(arguments.caller)?;
    let format = if arguments.json { OutputFormat::JsonLines } else { OutputFormat::Text };
    let deadlines = AnswerDeadlines { first: arguments.first_answer_within, last: arguments.final_answer_within };

    let session = open_session(&arguments.connect, &arguments.listen)?;
    let (sender, answers) = mpsc::channel();
    // Declared before the question goes out, so the first answer cannot be missed.
    let subscriber = session
        .declare_subscriber(ANSWER_KEY)
        .callback(move |sample| {
            let _ = sender.send(sample.payload().to_bytes().into_owned());
        })
        .wait()
        .map_err(zenoh_error("subscribing to rig/spatial_answer"))?;
    // Reliable is the default reliability; blocking waits for Zenoh's queue rather than drop the question.
    let publisher = session
        .declare_publisher(QUESTION_KEY)
        .congestion_control(CongestionControl::Block)
        .wait()
        .map_err(zenoh_error("declaring the publisher of rig/spatial_query"))?;
    wait_for_subscriber(&publisher, arguments.subscriber_within)?;

    let query = question(text, &caller, SystemTime::now());
    let bytes = encode_payload(&encode_spatial_query(&query)).map_err(|e| AskError::Zenoh(format!("encoding the question: {e:#}")))?;
    publisher.put(bytes).wait().map_err(zenoh_error("publishing the question"))?;
    let asked_at = Instant::now();
    eprintln!("rig-ask: asked {} as {:?} on {QUESTION_KEY}: {}", query.question_id, query.asker.0, query.text);

    let outcome = follow_answers(&answers, query.question_id, asked_at, deadlines, format, &mut io::stdout().lock());
    drop(subscriber);
    drop(publisher);
    session.close().wait().map_err(zenoh_error("closing the session"))?;
    outcome.map(|_| ())
}

fn main() -> ExitCode {
    match ask(Arguments::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rig-ask: {error}");
            ExitCode::from(error.exit_status())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::mpsc::Sender;

    use dora_node_api::arrow::ipc::reader::FileReader;
    use perception_wire::navigation::{decode_spatial_query, encode_spatial_answer};
    use serde_json::{json, Value};

    use super::*;

    const FIXTURE_QUESTION: QuestionId = QuestionId(31);

    fn fixture(name: &str) -> StructArray {
        let path = format!("{}/../../lib/dora-rig-messages/dora_rig_messages/fixtures/{name}.arrow", env!("CARGO_MANIFEST_DIR"));
        let mut reader = FileReader::try_new(File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")), None).unwrap();
        StructArray::from(reader.next().unwrap().unwrap())
    }

    fn send(answers: &Sender<Vec<u8>>, message: &StructArray) {
        answers.send(encode_payload(message).unwrap()).unwrap();
    }

    fn deadlines(first_ms: u64, last_ms: u64) -> AnswerDeadlines {
        AnswerDeadlines { first: Duration::from_millis(first_ms), last: Duration::from_millis(last_ms) }
    }

    fn follow(
        messages: &[StructArray],
        question: QuestionId,
        deadlines: AnswerDeadlines,
        format: OutputFormat,
    ) -> (Result<u32, AskError>, String) {
        let (sender, answers) = mpsc::channel();
        for message in messages {
            send(&sender, message);
        }
        let mut out = Vec::new();
        let outcome = follow_answers(&answers, question, Instant::now(), deadlines, format, &mut out);
        (outcome, String::from_utf8(out).unwrap())
    }

    fn answer_fixtures() -> [StructArray; 2] {
        [fixture("spatial_answer_first"), fixture("spatial_answer_final")]
    }

    #[test]
    fn a_question_decodes_as_the_world_agent_node_decodes_it() {
        let caller = Caller("robert".to_owned());
        let asked = UNIX_EPOCH + Duration::from_nanos(1_794_725_183_123_456_789);
        let query = question("Where is the couch?".to_owned(), &caller, asked);
        let bytes = encode_payload(&encode_spatial_query(&query)).unwrap();
        let decoded = decode_spatial_query(&decode_payload(&bytes).unwrap()).unwrap();
        assert_eq!(
            decoded,
            SpatialQuery {
                question_id: QuestionId(1_794_725_183_123_456_789),
                asked: UnixNanos(1_794_725_183_123_456_789),
                asker: Asker("rig-ask for robert".to_owned()),
                text: "Where is the couch?".to_owned(),
            }
        );
    }

    #[test]
    fn the_answer_fixtures_print_as_text_until_the_final_one() {
        let (outcome, printed) = follow(&answer_fixtures(), FIXTURE_QUESTION, deadlines(1000, 1000), OutputFormat::Text);
        assert_eq!(outcome.unwrap(), 2);
        let final_answer = decode_spatial_answer(&fixture("spatial_answer_final")).unwrap();
        assert_eq!(
            printed,
            [
                "revision 0 (first) from claude-haiku-5-5: The blue mug o51 is on the kitchen table o42; I am heading to it.",
                "  object o51 at x 2.25 m, y 1.50 m",
                "  object o42 at x 2.00 m, y 1.50 m",
                "  NAV GOAL g17 issued by this answer: the robot may be moving",
                &format!("revision 2 (final) from {}: {}", final_answer.model, final_answer.text),
                "  room r3 at x 1.50 m, y 2.75 m",
                "  map point at x 0.75 m, y 2.50 m",
                "",
            ]
            .join("\n")
        );
    }

    #[test]
    fn the_answer_fixtures_print_as_one_json_object_per_line() {
        let (outcome, printed) = follow(&answer_fixtures(), FIXTURE_QUESTION, deadlines(1000, 1000), OutputFormat::JsonLines);
        assert_eq!(outcome.unwrap(), 2);
        let final_answer = decode_spatial_answer(&fixture("spatial_answer_final")).unwrap();
        let lines: Vec<Value> = printed.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(
            lines,
            [
                json!({
                    "question_id": 31, "revision": 0, "is_final": false, "model": "claude-haiku-5-5",
                    "text": "The blue mug o51 is on the kitchen table o42; I am heading to it.",
                    "references": [
                        {"kind": "object", "id": 51, "x_m": 2.25, "y_m": 1.5},
                        {"kind": "object", "id": 42, "x_m": 2.0, "y_m": 1.5},
                    ],
                    "goal_id": 17,
                }),
                json!({
                    "question_id": 31, "revision": 2, "is_final": true, "model": final_answer.model,
                    "text": final_answer.text,
                    "references": [
                        {"kind": "room", "id": 3, "x_m": 1.5, "y_m": 2.75},
                        {"kind": "pose", "id": null, "x_m": 0.75, "y_m": 2.5},
                    ],
                    "goal_id": null,
                }),
            ]
        );
    }

    #[test]
    fn an_answer_to_another_question_is_skipped() {
        let mut other = decode_spatial_answer(&fixture("spatial_answer_final")).unwrap();
        other.question_id = QuestionId(32);
        other.text = "An answer to someone else.".to_owned();
        let [first, last] = answer_fixtures();
        let (outcome, printed) =
            follow(&[encode_spatial_answer(&other), first, last], FIXTURE_QUESTION, deadlines(1000, 1000), OutputFormat::Text);
        assert_eq!(outcome.unwrap(), 2);
        assert!(!printed.contains("someone else"), "{printed}");
    }

    #[test]
    fn only_answers_to_another_question_leave_the_first_answer_late() {
        let mut other = decode_spatial_answer(&fixture("spatial_answer_first")).unwrap();
        other.question_id = QuestionId(32);
        let (outcome, printed) = follow(&[encode_spatial_answer(&other)], FIXTURE_QUESTION, deadlines(50, 100), OutputFormat::Text);
        assert!(matches!(outcome, Err(AskError::FirstAnswerLate { question: FIXTURE_QUESTION, .. })), "{outcome:?}");
        assert_eq!(printed, "");
    }

    #[test]
    fn no_answer_is_a_late_first_answer() {
        let (outcome, _) = follow(&[], FIXTURE_QUESTION, deadlines(50, 100), OutputFormat::Text);
        let error = outcome.unwrap_err();
        assert_eq!(error.exit_status(), 4);
        assert_eq!(error.to_string(), "no first answer within 0.05 s of asking q31");
    }

    #[test]
    fn a_first_answer_without_a_final_one_is_a_late_final_answer() {
        let (outcome, printed) = follow(&[fixture("spatial_answer_first")], FIXTURE_QUESTION, deadlines(50, 100), OutputFormat::Text);
        let error = outcome.unwrap_err();
        assert!(matches!(error, AskError::FinalAnswerLate { received: 1, .. }), "{error:?}");
        assert_eq!(error.exit_status(), 5);
        assert!(printed.starts_with("revision 0 (first)"), "{printed}");
    }

    #[test]
    fn a_payload_that_is_not_an_answer_stops_with_a_decode_error() {
        let (outcome, _) = follow(&[fixture("spatial_query")], FIXTURE_QUESTION, deadlines(1000, 1000), OutputFormat::Text);
        let error = outcome.unwrap_err();
        assert!(matches!(error, AskError::UndecodableAnswer(_)), "{error:?}");
        assert_eq!(error.exit_status(), 6);
    }

    #[test]
    fn a_question_from_stdin_is_trimmed_and_an_empty_one_refused() {
        assert_eq!(question_text("-", &mut "  where is the couch?\n".as_bytes()).unwrap(), "where is the couch?");
        assert_eq!(question_text("where?", &mut "ignored".as_bytes()).unwrap(), "where?");
        assert!(matches!(question_text("-", &mut "\n".as_bytes()), Err(AskError::Usage(_))));
    }

    #[test]
    fn durations_must_be_positive_seconds() {
        assert_eq!(seconds("1.5"), Ok(Duration::from_millis(1500)));
        assert!(seconds("0").is_err());
        assert!(seconds("-3").is_err());
        assert!(seconds("soon").is_err());
    }
}
