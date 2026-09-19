//! The serial connection to the lidar, which comes and goes with its USB adapter.

use std::io::Read;
use std::time::{Duration, Instant};

use delta2a::{Frame, FrameDecoder, Revolution, RevolutionAssembler, BAUD_RATE};
use eyre::WrapErr;
use serialport::SerialPort;

const REOPEN_PERIOD: Duration = Duration::from_secs(1);
/// Reads only ask for bytes that have already arrived, so this is never waited out.
const READ_TIMEOUT: Duration = Duration::from_millis(10);

pub struct LidarLink {
    device: String,
    connection: Option<Connection>,
    reopen_at: Instant,
    waiting_announced: bool,
}

struct Connection {
    port: Box<dyn SerialPort>,
    decoder: FrameDecoder,
    assembler: RevolutionAssembler,
    speed_fault: bool,
}

impl LidarLink {
    pub fn new(device: String, now: Instant) -> Self {
        Self {
            device,
            connection: None,
            reopen_at: now,
            waiting_announced: false,
        }
    }

    /// Reads what has arrived since the last poll and returns the revolutions it completes.
    /// Any failure of the port drops the connection, and later polls try to open it again.
    pub fn poll(&mut self, now: Instant) -> Vec<Revolution> {
        if self.connection.is_none() && now >= self.reopen_at {
            self.reopen(now);
        }
        let Some(connection) = &mut self.connection else {
            return Vec::new();
        };
        match connection.read_revolutions() {
            Ok(revolutions) => revolutions,
            Err(error) => {
                println!("[lidar] lost {}: {error:#}", self.device);
                self.connection = None;
                self.reopen_at = now + REOPEN_PERIOD;
                Vec::new()
            }
        }
    }

    fn reopen(&mut self, now: Instant) {
        match Connection::open(&self.device) {
            Ok(connection) => {
                println!("[lidar] reading {}", self.device);
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
    fn open(device: &str) -> eyre::Result<Self> {
        let port = serialport::new(device, BAUD_RATE)
            .timeout(READ_TIMEOUT)
            .open()
            .wrap_err("cannot open the serial port")?;
        Ok(Self {
            port,
            decoder: FrameDecoder::new(),
            assembler: RevolutionAssembler::new(),
            speed_fault: false,
        })
    }

    fn read_revolutions(&mut self) -> eyre::Result<Vec<Revolution>> {
        let arrived = self
            .port
            .bytes_to_read()
            .wrap_err("cannot ask how many bytes have arrived")? as usize;
        let mut bytes = vec![0; arrived];
        self.port
            .read_exact(&mut bytes)
            .wrap_err("cannot read the bytes that have arrived")?;

        let mut revolutions = Vec::new();
        for frame in self.decoder.push(&bytes) {
            match frame {
                Frame::Scan(sector) => {
                    self.note_speed_fault(false);
                    revolutions.extend(self.assembler.push(sector));
                }
                Frame::SpeedFault(_) => self.note_speed_fault(true),
            }
        }
        Ok(revolutions)
    }

    fn note_speed_fault(&mut self, speed_fault: bool) {
        if speed_fault != self.speed_fault {
            let report = if speed_fault {
                "the head is not up to speed"
            } else {
                "the head is up to speed"
            };
            println!("[lidar] {report}");
            self.speed_fault = speed_fault;
        }
    }
}
