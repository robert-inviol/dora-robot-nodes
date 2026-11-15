//! Times the two steps a depth frame takes before registration, on one simulated 848x480 frame: the
//! image to sensor-frame points, and downsampling those points.
//!
//! `cargo run --release -p depth-inertial-odometry --example depth_frame_cost -- [repeats]`

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use depth_inertial_odometry::depth_image::DepthProjector;
use depth_inertial_odometry::simulation::{seeded, PinholeDepthCamera, Scene, Walk};
use depth_inertial_odometry::downsampling::{SensorExtrinsics, Downsampler};
use depth_inertial_odometry::OdometrySettings;

fn median_ms(mut times: Vec<Duration>) -> f64 {
    times.sort();
    times[times.len() / 2].as_secs_f64() * 1e3
}

fn main() {
    let repeats = std::env::args().nth(1).map(|text| text.parse::<usize>().expect("the argument is a whole number of repeats")).unwrap_or(200);
    let camera = PinholeDepthCamera::d455(848, 480);
    let extrinsics = SensorExtrinsics::from_depth_to_imu(PinholeDepthCamera::FACING_FORWARD, [0.0; 3]).expect("a rotation");
    let walk = Walk::on_a_robot(Duration::ZERO);
    let settings = OdometrySettings::d455();
    let mut depth_mm = Vec::new();
    camera.image(&Scene::living_room(), walk.position(5.0), walk.rotation(5.0), &extrinsics, &mut seeded(7), &mut depth_mm);

    println!("one 848x480 depth frame, {repeats} repeats, {} threads", rayon::current_num_threads());
    for stride in [1, 2, 4] {
        let projector = DepthProjector::new(camera.intrinsics, NonZeroUsize::new(stride).expect("not zero")).expect("usable intrinsics");
        let (mut points, mut downsampled, mut downsampler) = (Vec::new(), Vec::new(), Downsampler::default());
        let (mut project_times, mut downsample_times) = (Vec::with_capacity(repeats), Vec::with_capacity(repeats));
        for _ in 0..repeats {
            let started = Instant::now();
            projector.project(&depth_mm, &mut points).expect("the image fits");
            project_times.push(started.elapsed());
            let started = Instant::now();
            downsampler.downsample(&points, &extrinsics, &settings.downsampling, &settings.association.noise, &mut downsampled);
            downsample_times.push(started.elapsed());
        }
        println!(
            "stride {stride}: {} points, {} downsampled  depth to points: median {:.2} ms  downsampling: median {:.2} ms",
            points.len(),
            downsampled.len(),
            median_ms(project_times),
            median_ms(downsample_times),
        );
    }
}
