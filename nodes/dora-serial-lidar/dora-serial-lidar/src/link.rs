//! The serial connection to the lidar, which comes and goes with its USB adapter.

use std::io::Read;
use std::time::{Duration, Instant};

use eyre::WrapErr;
use serialport::{ClearBuffer, SerialPort};

use crate::model::LidarModel;
use crate::turn::{Turn, TurnDecoder, UnixNanos};

const REOPEN_PERIOD: Duration = Duration::from_secs(1);
/// Reads only ask for bytes that have already arrived, so this is never waited out.
const READ_TIMEOUT: Duration = Duration::from_millis(10);

pub struct LidarLink {
    device: String,
    model: LidarModel,
    connection: Option<Connection>,
    reopen_at: Instant,
    waiting_announced: bool,
}

struct Connection {
    port: Box<dyn SerialPort>,
    turns: Box<dyn TurnDecoder>,
}

impl LidarLink {
    pub fn new(device: String, model: LidarModel, now: Instant) -> Self {
        Self {
            device,
            model,
            connection: None,
            reopen_at: now,
            waiting_announced: false,
        }
    }

    /// Reads what has arrived since the last poll and returns the turns it completes.
    /// Any failure of the port drops the connection, and later polls try to open it again.
    pub fn poll(&mut self, now: Instant) -> Vec<Turn> {
        if self.connection.is_none() && now >= self.reopen_at {
            self.reopen(now);
        }
        let Some(connection) = &mut self.connection else {
            return Vec::new();
        };
        match connection.read_turns() {
            Ok(turns) => turns,
            Err(error) => {
                println!("[lidar] lost {}: {error:#}", self.device);
                self.connection = None;
                self.reopen_at = now + REOPEN_PERIOD;
                Vec::new()
            }
        }
    }

    fn reopen(&mut self, now: Instant) {
        match Connection::open(&self.device, self.model) {
            Ok(connection) => {
                println!("[lidar] reading {:?} from {}", self.model, self.device);
                self.connection = Some(connection);
                self.waiting_announced = false;
            }
            Err(error) => {
                if !self.waiting_announced {
                    println!("[lidar] waiting for {}: {error:#}", self.device);
                    self.waiting_announced = true;
                }
                self.reopen_at = now + REOPEN_PERIOD;
            }
        }
    }
}

impl Connection {
    fn open(device: &str, model: LidarModel) -> eyre::Result<Self> {
        let port = serialport::new(device, model.baud_rate())
            .timeout(READ_TIMEOUT)
            .open()
            .wrap_err("cannot open the serial port")?;
        // Bytes that waited in the buffer arrive all at once, which would time their readings by the opening.
        port.clear(ClearBuffer::Input)
            .wrap_err("cannot clear what the serial port buffered before it was opened")?;
        Ok(Self {
            port,
            turns: model.turn_decoder(),
        })
    }

    fn read_turns(&mut self) -> eyre::Result<Vec<Turn>> {
        let arrived = self
            .port
            .bytes_to_read()
            .wrap_err("cannot ask how many bytes have arrived")? as usize;
        let read_at = UnixNanos::now();
        let mut bytes = vec![0; arrived];
        self.port
            .read_exact(&mut bytes)
            .wrap_err("cannot read the bytes that have arrived")?;
        Ok(self.turns.push(&bytes, read_at))
    }
}
