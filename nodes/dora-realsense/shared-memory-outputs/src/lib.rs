//! Sends a dora node's outputs and stops the node with an error naming the output when a receiver has stopped
//! releasing what it was sent.
//!
//! dora-node-api 0.3.13 sends each sample of [`ZERO_COPY_THRESHOLD`] bytes or more in a shared-memory region of its
//! own, and the sender keeps that region mapped, with its file descriptor open, until every receiver has released
//! the sample. A released region goes into a cache of at most [`DORA_REGION_CACHE`] regions for reuse. A receiving
//! node releases a sample when it drops it, and dora applies an input's `queue_size` inside the receiving node only
//! when that node asks for its next input. A receiver that stops asking therefore holds every sample sent to it,
//! whatever its queue size, and the sender holds one more region and file descriptor per sample until it has none
//! left.
//!
//! [`SharedMemoryOutputs`] sends every output through dora and remembers the output each region was last sent on.
//! The node API does not say which samples are released, so the regions still mapped in this process stand for the
//! ones not yet released, plus at most [`DORA_REGION_CACHE`] kept for reuse. A send fails with [`ConsumerStopped`]
//! once more than the limit of one output's regions are mapped.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use dora_node_api::arrow::array::Array;
use dora_node_api::arrow_utils::{copy_array_into_sample, required_data_size};
use dora_node_api::dora_core::config::{DataId, InputMapping, NodeId, UserInputMapping};
use dora_node_api::dora_core::metadata::ArrowTypeInfoExt;
use dora_node_api::{DataSample, DoraNode, MetadataParameters, ZERO_COPY_THRESHOLD};
use eyre::{eyre, Context, Result};

/// The most released regions dora-node-api 0.3.13 keeps mapped for reuse.
pub const DORA_REGION_CACHE: usize = 20;
/// How often a send reads this process's mappings; sends in between only record their region.
const CHECK_INTERVAL: Duration = Duration::from_millis(100);
/// The directory and name prefix of the regions dora-node-api creates.
const DORA_REGION_PATH_PREFIX: &str = "/dev/shm/shmem_";

/// Where a shared-memory region starts in this process's address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionStart(pub usize);

/// The dora shared-memory regions mapped in this process.
#[derive(Debug, Default)]
pub struct MappedRegions(HashSet<RegionStart>);

impl MappedRegions {
    pub fn of_this_process() -> Result<MappedRegions> {
        let maps = std::fs::read_to_string("/proc/self/maps").wrap_err("reading /proc/self/maps")?;
        MappedRegions::from_maps(&maps)
    }

    /// Reads the regions out of the text of a `/proc/<pid>/maps` file.
    pub fn from_maps(maps: &str) -> Result<MappedRegions> {
        let mut regions = HashSet::new();
        for line in maps.lines() {
            let mut fields = line.split_whitespace();
            let range = fields.next().ok_or_else(|| eyre!("an empty line in /proc/self/maps"))?;
            if !fields.nth(4).is_some_and(|path| path.starts_with(DORA_REGION_PATH_PREFIX)) {
                continue;
            }
            let start = range.split_once('-').map(|(start, _)| start).ok_or_else(|| eyre!("a mapping without a range: {line}"))?;
            let start = usize::from_str_radix(start, 16).wrap_err_with(|| format!("a mapping whose start is not hex: {line}"))?;
            regions.insert(RegionStart(start));
        }
        Ok(MappedRegions(regions))
    }

    fn contains(&self, region: &RegionStart) -> bool {
        self.0.contains(region)
    }
}

/// An output with more samples in shared memory than its limit that no receiver has released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumerStopped {
    pub output: DataId,
    pub unreleased: usize,
    pub receivers: Vec<NodeId>,
}

impl fmt::Display for ConsumerStopped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let receivers: Vec<&str> = self.receivers.iter().map(|id| id.as_ref()).collect();
        write!(
            f,
            "{} has {} samples in shared memory that no receiver has released: one of its receivers ({}) has stopped \
             taking its inputs",
            self.output,
            self.unreleased,
            receivers.join(", ")
        )
    }
}

impl std::error::Error for ConsumerStopped {}

/// For each output, the shared-memory regions it was last sent in that are still mapped.
#[derive(Debug)]
pub struct UnreleasedSamples {
    limit: usize,
    last_sent_on: HashMap<RegionStart, DataId>,
}

impl UnreleasedSamples {
    /// `limit` is the most regions one output may hold; it has to exceed what its receivers keep at once, their
    /// queues included, plus [`DORA_REGION_CACHE`].
    pub fn new(limit: usize) -> UnreleasedSamples {
        UnreleasedSamples { limit, last_sent_on: HashMap::new() }
    }

    /// A sample of `output` went out in the region starting at `region`; dora reuses regions, so this replaces the
    /// region's earlier output.
    pub fn record(&mut self, region: RegionStart, output: &DataId) {
        self.last_sent_on.insert(region, output.clone());
    }

    /// Forgets the regions that are no longer mapped, and fails on the output holding the most regions when that is
    /// more than the limit.
    pub fn check(&mut self, mapped: &MappedRegions) -> Result<(), ConsumerStopped> {
        self.last_sent_on.retain(|region, _| mapped.contains(region));
        let mut held: BTreeMap<&DataId, usize> = BTreeMap::new();
        for output in self.last_sent_on.values() {
            *held.entry(output).or_default() += 1;
        }
        match held.into_iter().max_by_key(|(_, count)| *count) {
            Some((output, unreleased)) if unreleased > self.limit => {
                Err(ConsumerStopped { output: output.clone(), unreleased, receivers: Vec::new() })
            }
            _ => Ok(()),
        }
    }
}

/// How dora carries a sample: in a shared-memory region from [`ZERO_COPY_THRESHOLD`] bytes, inline below it.
enum SentSample {
    InSharedMemory(RegionStart),
    Inline,
}

impl SentSample {
    fn of(sample: &DataSample) -> SentSample {
        if sample.len() >= ZERO_COPY_THRESHOLD {
            SentSample::InSharedMemory(RegionStart(sample.as_ptr() as usize))
        } else {
            SentSample::Inline
        }
    }
}

/// Sends a node's outputs through dora, checking that its receivers keep releasing them.
///
/// An output the dataflow does not list is not sent, as dora's own `send_output` does.
pub struct SharedMemoryOutputs {
    unreleased: UnreleasedSamples,
    receivers: HashMap<DataId, Vec<NodeId>>,
    last_check: Option<Instant>,
}

impl SharedMemoryOutputs {
    /// `limit` is as for [`UnreleasedSamples::new`].
    pub fn new(node: &DoraNode, limit: usize) -> Result<SharedMemoryOutputs> {
        let descriptor = node.dataflow_descriptor().wrap_err("reading the dataflow")?;
        let mut receivers: HashMap<DataId, Vec<NodeId>> = HashMap::new();
        for receiver in &descriptor.nodes {
            for input in receiver.inputs.values() {
                if let InputMapping::User(UserInputMapping { source, output }) = &input.mapping {
                    if source == node.id() {
                        receivers.entry(output.clone()).or_default().push(receiver.id.clone());
                    }
                }
            }
        }
        Ok(SharedMemoryOutputs { unreleased: UnreleasedSamples::new(limit), receivers, last_check: None })
    }

    /// Sends `len` bytes that `fill` writes, as a byte array.
    pub fn send_raw(
        &mut self,
        node: &mut DoraNode,
        output: &DataId,
        parameters: MetadataParameters,
        len: usize,
        fill: impl FnOnce(&mut [u8]),
    ) -> Result<()> {
        if !node.node_config().outputs.contains(output) {
            return Ok(());
        }
        let mut sample = node.allocate_data_sample(len).wrap_err_with(|| format!("allocating {output}"))?;
        fill(&mut sample);
        let sent = SentSample::of(&sample);
        node.send_output_sample(output.clone(), ArrowTypeInfoExt::byte_array(len), parameters, Some(sample))
            .wrap_err_with(|| format!("sending {output}"))?;
        self.after_sending(output, sent)
    }

    /// Sends an Arrow array.
    pub fn send_array(&mut self, node: &mut DoraNode, output: &DataId, parameters: MetadataParameters, array: impl Array) -> Result<()> {
        if !node.node_config().outputs.contains(output) {
            return Ok(());
        }
        let array = array.to_data();
        let mut sample = node.allocate_data_sample(required_data_size(&array)).wrap_err_with(|| format!("allocating {output}"))?;
        let type_info = copy_array_into_sample(&mut sample, &array);
        let sent = SentSample::of(&sample);
        node.send_output_sample(output.clone(), type_info, parameters, Some(sample)).wrap_err_with(|| format!("sending {output}"))?;
        self.after_sending(output, sent)
    }

    fn after_sending(&mut self, output: &DataId, sent: SentSample) -> Result<()> {
        let SentSample::InSharedMemory(region) = sent else { return Ok(()) };
        self.unreleased.record(region, output);
        if self.last_check.is_some_and(|checked| checked.elapsed() < CHECK_INTERVAL) {
            return Ok(());
        }
        self.last_check = Some(Instant::now());
        self.unreleased.check(&MappedRegions::of_this_process()?).map_err(|stopped| {
            let receivers = self.receivers.get(&stopped.output).cloned().unwrap_or_default();
            eyre::Report::new(ConsumerStopped { receivers, ..stopped })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(name: &str) -> DataId {
        DataId::from(name.to_owned())
    }

    fn mapped(starts: &[usize]) -> MappedRegions {
        MappedRegions(starts.iter().copied().map(RegionStart).collect())
    }

    #[test]
    fn reads_only_dora_regions_from_the_maps() {
        let maps = "\
55d0c0a00000-55d0c0a21000 r--p 00000000 103:02 1234                      /usr/bin/dora-realsense
7f3a10000000-7f3a1012c000 rw-s 00000000 00:1a 5678                       /dev/shm/shmem_9F3A1C2B4D5E6F70
7f3a20000000-7f3a20001000 rw-s 00000000 00:1a 5679                       /dev/shm/other
7f3a30000000-7f3a30021000 rw-p 00000000 00:00 0
7ffd4c000000-7ffd4c021000 rw-p 00000000 00:00 0                          [stack]
";
        let regions = MappedRegions::from_maps(maps).unwrap();
        assert_eq!(regions.0, HashSet::from([RegionStart(0x7f3a10000000)]));
    }

    #[test]
    fn a_receiver_that_keeps_releasing_stays_within_the_limit() {
        let mut unreleased = UnreleasedSamples::new(3);
        // dora reuses the same two regions while the receiver releases them.
        for frame in 0..100 {
            unreleased.record(RegionStart(0x1000 * (1 + frame % 2)), &output("model_input"));
            assert_eq!(unreleased.check(&mapped(&[0x1000, 0x2000])), Ok(()));
        }
    }

    #[test]
    fn a_receiver_that_stops_releasing_fails_once_past_the_limit() {
        let mut unreleased = UnreleasedSamples::new(3);
        let mut held = Vec::new();
        for frame in 1..=3 {
            held.push(0x1000 * frame);
            unreleased.record(RegionStart(0x1000 * frame), &output("model_input"));
            assert_eq!(unreleased.check(&mapped(&held)), Ok(()));
        }
        held.push(0x4000);
        unreleased.record(RegionStart(0x4000), &output("model_input"));
        let stopped = unreleased.check(&mapped(&held)).unwrap_err();
        assert_eq!((stopped.output, stopped.unreleased), (output("model_input"), 4));
    }

    #[test]
    fn names_the_output_that_holds_the_regions() {
        let mut unreleased = UnreleasedSamples::new(2);
        unreleased.record(RegionStart(0x1000), &output("depth"));
        for region in [0x2000, 0x3000, 0x4000] {
            unreleased.record(RegionStart(region), &output("model_input"));
        }
        let stopped = unreleased.check(&mapped(&[0x1000, 0x2000, 0x3000, 0x4000])).unwrap_err();
        assert_eq!(stopped.output, output("model_input"));
    }

    #[test]
    fn a_reused_region_counts_for_the_output_it_was_last_sent_on() {
        let mut unreleased = UnreleasedSamples::new(1);
        unreleased.record(RegionStart(0x1000), &output("model_input"));
        unreleased.record(RegionStart(0x2000), &output("model_input"));
        unreleased.record(RegionStart(0x1000), &output("depth"));
        assert_eq!(unreleased.check(&mapped(&[0x1000, 0x2000])), Ok(()));
    }

    #[test]
    fn an_unmapped_region_is_forgotten() {
        let mut unreleased = UnreleasedSamples::new(1);
        unreleased.record(RegionStart(0x1000), &output("model_input"));
        unreleased.record(RegionStart(0x2000), &output("model_input"));
        assert_eq!(unreleased.check(&mapped(&[0x2000])), Ok(()));
        unreleased.record(RegionStart(0x3000), &output("model_input"));
        assert!(unreleased.check(&mapped(&[0x2000, 0x3000])).is_err());
    }

    #[test]
    fn the_error_names_the_output_and_its_receivers() {
        let stopped = ConsumerStopped {
            output: output("model_input"),
            unreleased: 121,
            receivers: vec![NodeId::from("pose".to_owned())],
        };
        assert_eq!(
            stopped.to_string(),
            "model_input has 121 samples in shared memory that no receiver has released: one of its receivers (pose) \
             has stopped taking its inputs"
        );
    }
}
