use glam::{Mat3, Vec3};

/// Pinhole intrinsics of the image the depth is aligned to.
#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub ppx: f32,
    pub ppy: f32,
}

impl CameraIntrinsics {
    /// Camera-frame point (x right, y down, z along the optical axis) for a pixel at a depth.
    #[inline]
    pub fn deproject(&self, column: f32, row: f32, depth_m: f32) -> Vec3 {
        Vec3::new((column - self.ppx) * depth_m / self.fx, (row - self.ppy) * depth_m / self.fy, depth_m)
    }
}

/// Depth in metres, row major, zero where the sensor has no reading.
#[derive(Clone, Copy)]
pub struct DepthImage<'a> {
    pub width: usize,
    pub height: usize,
    pub depth_m: &'a [f32],
}

impl DepthImage<'_> {
    pub fn new(width: usize, height: usize, depth_m: &[f32]) -> DepthImage<'_> {
        assert_eq!(depth_m.len(), width * height, "depth buffer does not match {width}x{height}");
        DepthImage { width, height, depth_m }
    }
}

/// Depth readings outside this band are not trusted.
#[derive(Clone, Copy, Debug)]
pub struct RangeLimits {
    pub min_m: f32,
    pub max_m: f32,
}

impl RangeLimits {
    /// The D455's minimum range, and the distance past which its depth error is too large to use.
    pub const D455: RangeLimits = RangeLimits { min_m: 0.4, max_m: 8.0 };

    #[inline]
    pub fn contains(&self, depth_m: f32) -> bool {
        depth_m > self.min_m && depth_m < self.max_m
    }
}

/// Rotation from camera coordinates into the floor frame, built from the accelerometer.
#[derive(Clone, Copy, Debug)]
pub struct FloorFrame {
    camera_to_floor: Mat3,
}

impl FloorFrame {
    /// The accelerometer at rest reports the reaction to gravity, which points up, in camera coordinates.
    pub fn from_gravity_reaction(accel_camera: Vec3) -> FloorFrame {
        let up = accel_camera.normalize();
        let optical_axis = Vec3::Z;
        let forward = (optical_axis - optical_axis.dot(up) * up).normalize();
        let right = forward.cross(up);
        FloorFrame { camera_to_floor: Mat3::from_cols(right, forward, up).transpose() }
    }

    #[inline]
    pub fn to_floor(&self, camera_point: Vec3) -> Vec3 {
        self.camera_to_floor * camera_point
    }

    /// The camera's optical axis and image-up direction, in the floor frame.
    pub fn camera_forward_and_up(&self) -> (Vec3, Vec3) {
        (self.to_floor(Vec3::Z), self.to_floor(Vec3::NEG_Y))
    }
}

/// Floor-frame points and the pixel each came from.
#[derive(Clone, Debug, Default)]
pub struct PointCloud {
    pub points: Vec<Vec3>,
    /// Row-major pixel index (row * width + column) of each point.
    pub pixels: Vec<u32>,
}

/// Every `stride`-th pixel in each direction with a trusted depth, as floor-frame points.
pub fn sample_point_cloud(
    depth: &DepthImage,
    intrinsics: &CameraIntrinsics,
    floor: &FloorFrame,
    range: RangeLimits,
    stride: usize,
) -> PointCloud {
    let mut cloud = PointCloud::default();
    let capacity = (depth.width / stride + 1) * (depth.height / stride + 1);
    cloud.points.reserve(capacity);
    cloud.pixels.reserve(capacity);
    for row in (0..depth.height).step_by(stride) {
        let row_values = &depth.depth_m[row * depth.width..(row + 1) * depth.width];
        for column in (0..depth.width).step_by(stride) {
            let depth_m = row_values[column];
            if range.contains(depth_m) {
                cloud.points.push(floor.to_floor(intrinsics.deproject(column as f32, row as f32, depth_m)));
                cloud.pixels.push((row * depth.width + column) as u32);
            }
        }
    }
    cloud
}
