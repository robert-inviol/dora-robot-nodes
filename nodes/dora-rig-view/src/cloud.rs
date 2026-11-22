//! A keyframe's depth as a thinned, coloured point cloud in the keyframe's floor frame, encoded once for the page.
//!
//! The floor frame is gravity aligned with its origin at the camera: x right, y forward, z up. The page places the
//! cloud with the keyframe's pose, so a later pose moves the cloud without sending it again.

use std::io::Write;

use bytes::{BufMut, Bytes, BytesMut};
use depth_perception::FloorFrame;
use flate2::write::GzEncoder;
use flate2::Compression;
use jpeg_decoder::PixelFormat;
use perception_wire::localization::FrameId;
use perception_wire::Keyframe;

/// The first byte of a binary message to the page that carries one keyframe's cloud.
pub const CLOUD_MESSAGE: u8 = 1;

/// Every position must fit in an i16 of millimetres.
const MAX_REPRESENTABLE_RANGE_M: f32 = 32.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudSettings {
    /// Every n-th pixel each way becomes a point.
    pub stride_px: usize,
    pub min_range_m: f32,
    pub max_range_m: f32,
    /// A later keyframe whose depth lies this much beyond a cloud's point saw through it, and removes it.
    pub space_carving_margin_m: f32,
    /// The side of the map voxels the drawn clouds share: one point is drawn in each.
    pub voxel_m: f32,
}

impl CloudSettings {
    pub const DEFAULT: CloudSettings = CloudSettings { stride_px: 2, min_range_m: 0.4, max_range_m: 5.0, space_carving_margin_m: 0.1, voxel_m: 0.02 };

    pub fn checked(self) -> Result<CloudSettings, CloudError> {
        if self.stride_px == 0 {
            return Err(CloudError::ZeroStride);
        }
        if !(0.0..self.max_range_m).contains(&self.min_range_m) || self.max_range_m > MAX_REPRESENTABLE_RANGE_M {
            return Err(CloudError::Range { min_m: self.min_range_m, max_m: self.max_range_m });
        }
        if !(self.space_carving_margin_m.is_finite() && self.space_carving_margin_m > 0.0) {
            return Err(CloudError::SpaceCarvingMargin(self.space_carving_margin_m));
        }
        if !(self.voxel_m.is_finite() && self.voxel_m > 0.0) {
            return Err(CloudError::Voxel(self.voxel_m));
        }
        Ok(self)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    #[error("the cloud stride must be at least one pixel")]
    ZeroStride,
    #[error("the cloud range {min_m} m to {max_m} m must start at 0 or more, end after it, and end within 32 m")]
    Range { min_m: f32, max_m: f32 },
    #[error("the carving margin {0} m must be above 0")]
    SpaceCarvingMargin(f32),
    #[error("the cloud voxel {0} m must be above 0")]
    Voxel(f32),
    #[error("keyframe {frame_id}'s colour image does not decode: {source}")]
    Jpeg { frame_id: u64, source: jpeg_decoder::Error },
    #[error("keyframe {frame_id}'s colour image is {jpeg_width}x{jpeg_height}, its depth {depth_width}x{depth_height}")]
    SizeMismatch { frame_id: u64, jpeg_width: usize, jpeg_height: usize, depth_width: usize, depth_height: usize },
    #[error("keyframe {frame_id}'s colour image is {format:?}; only RGB and greyscale are drawn")]
    PixelFormat { frame_id: u64, format: PixelFormat },
}

/// One keyframe's points in its floor frame, in millimetres, with their colours.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyframeCloud {
    pub frame_id: FrameId,
    pub positions_mm: Vec<[i16; 3]>,
    pub colors: Vec<[u8; 3]>,
}

impl KeyframeCloud {
    pub fn from_keyframe(keyframe: &Keyframe, settings: &CloudSettings) -> Result<KeyframeCloud, CloudError> {
        let colors = decode_rgb(keyframe)?;
        let floor = FloorFrame::from_specific_force(keyframe.specific_force);
        let mut positions_mm = Vec::new();
        let mut kept_colors = Vec::new();
        for row in (0..keyframe.height).step_by(settings.stride_px) {
            for column in (0..keyframe.width).step_by(settings.stride_px) {
                let pixel = row * keyframe.width + column;
                let depth_m = keyframe.depth_mm[pixel] as f32 * 0.001;
                if depth_m <= settings.min_range_m || depth_m >= settings.max_range_m {
                    continue;
                }
                let point = floor.to_floor(keyframe.intrinsics.deproject(column as f32, row as f32, depth_m));
                positions_mm.push((point * 1000.0).round().as_i16vec3().to_array());
                kept_colors.push(colors[pixel]);
            }
        }
        Ok(KeyframeCloud { frame_id: FrameId(keyframe.frame_id), positions_mm, colors: kept_colors })
    }

    /// `CLOUD_MESSAGE`, then gzip of: frame id (u64), point count (u32), every position (3 x i16 mm), every colour
    /// (3 x u8). Little endian.
    pub fn encode(&self) -> Bytes {
        let mut body = BytesMut::with_capacity(12 + self.positions_mm.len() * 9);
        body.put_u64_le(self.frame_id.0);
        body.put_u32_le(self.positions_mm.len() as u32);
        for position in &self.positions_mm {
            position.iter().for_each(|&axis| body.put_i16_le(axis));
        }
        for color in &self.colors {
            body.put_slice(color);
        }
        let mut encoder = GzEncoder::new(vec![CLOUD_MESSAGE], Compression::fast());
        encoder.write_all(&body).expect("writing to memory");
        Bytes::from(encoder.finish().expect("writing to memory"))
    }
}

fn decode_rgb(keyframe: &Keyframe) -> Result<Vec<[u8; 3]>, CloudError> {
    let frame_id = keyframe.frame_id;
    let mut decoder = jpeg_decoder::Decoder::new(keyframe.color_jpeg.as_slice());
    let pixels = decoder.decode().map_err(|source| CloudError::Jpeg { frame_id, source })?;
    let info = decoder.info().expect("decoded above");
    let (jpeg_width, jpeg_height) = (info.width as usize, info.height as usize);
    if (jpeg_width, jpeg_height) != (keyframe.width, keyframe.height) {
        return Err(CloudError::SizeMismatch {
            frame_id,
            jpeg_width,
            jpeg_height,
            depth_width: keyframe.width,
            depth_height: keyframe.height,
        });
    }
    match info.pixel_format {
        PixelFormat::RGB24 => Ok(pixels.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect()),
        PixelFormat::L8 => Ok(pixels.iter().map(|&l| [l, l, l]).collect()),
        format => Err(CloudError::PixelFormat { frame_id, format }),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Read;

    use depth_perception::CameraIntrinsics;
    use flate2::read::GzDecoder;
    use glam::Vec3;
    use perception_wire::Keyframe;

    use super::*;

    /// A level camera looking at a wall `depth_mm` away, its image one colour.
    pub(crate) fn wall_keyframe(frame_id: u64, depth_mm: u16, rgb: [u8; 3]) -> Keyframe {
        let (width, height) = (8usize, 6usize);
        let pixels: Vec<u8> = (0..width * height).flat_map(|_| rgb).collect();
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 100)
            .encode(&pixels, width as u16, height as u16, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        Keyframe {
            frame_id,
            captured_unix_ns: 1,
            width,
            height,
            intrinsics: CameraIntrinsics { fx: 4.0, fy: 4.0, ppx: 4.0, ppy: 3.0 },
            specific_force: Vec3::new(0.0, -9.8, 0.0),
            depth_mm: vec![depth_mm; width * height].into(),
            color_jpeg: jpeg,
        }
    }

    #[test]
    fn a_level_camera_puts_a_wall_ahead_along_y_with_up_along_z() {
        let settings = CloudSettings { stride_px: 2, ..CloudSettings::DEFAULT };
        let cloud = KeyframeCloud::from_keyframe(&wall_keyframe(7, 2000, [200, 40, 10]), &settings).unwrap();
        assert_eq!(cloud.positions_mm.len(), 4 * 3, "every second pixel each way");
        assert!(cloud.positions_mm.iter().all(|p| p[1] == 2000), "the wall is 2 m forward: {:?}", cloud.positions_mm);
        let top_row = cloud.positions_mm[0];
        assert!(top_row[2] > 0 && top_row[0] < 0, "the image's top left is up and to the left: {top_row:?}");
        assert!(cloud.colors.iter().all(|c| c[0].abs_diff(200) < 8 && c[1].abs_diff(40) < 8 && c[2].abs_diff(10) < 8));
    }

    #[test]
    fn depth_outside_the_range_is_left_out() {
        let near = KeyframeCloud::from_keyframe(&wall_keyframe(1, 300, [0; 3]), &CloudSettings::DEFAULT).unwrap();
        let far = KeyframeCloud::from_keyframe(&wall_keyframe(1, 6000, [0; 3]), &CloudSettings::DEFAULT).unwrap();
        assert!(near.positions_mm.is_empty() && far.positions_mm.is_empty());
    }

    #[test]
    fn a_colour_image_of_another_size_is_refused() {
        let mut keyframe = wall_keyframe(3, 2000, [0; 3]);
        keyframe.width = 4;
        keyframe.depth_mm = vec![2000u16; 4 * 6].into();
        let refused = KeyframeCloud::from_keyframe(&keyframe, &CloudSettings::DEFAULT);
        assert!(matches!(refused, Err(CloudError::SizeMismatch { frame_id: 3, jpeg_width: 8, .. })), "{refused:?}");
    }

    #[test]
    fn the_encoding_is_the_kind_byte_then_gzip_of_the_layout() {
        let cloud = KeyframeCloud { frame_id: FrameId(9), positions_mm: vec![[1, -2, 3]], colors: vec![[4, 5, 6]] };
        let encoded = cloud.encode();
        assert_eq!(encoded[0], CLOUD_MESSAGE);
        let mut body = Vec::new();
        GzDecoder::new(&encoded[1..]).read_to_end(&mut body).unwrap();
        let mut expected = 9u64.to_le_bytes().to_vec();
        expected.extend(1u32.to_le_bytes());
        [1i16, -2, 3].iter().for_each(|v| expected.extend(v.to_le_bytes()));
        expected.extend([4, 5, 6]);
        assert_eq!(body, expected);
    }

    #[test]
    fn settings_that_cannot_be_encoded_are_refused() {
        assert!(matches!(CloudSettings { stride_px: 0, ..CloudSettings::DEFAULT }.checked(), Err(CloudError::ZeroStride)));
        let far = CloudSettings { max_range_m: 40.0, ..CloudSettings::DEFAULT };
        assert!(matches!(far.checked(), Err(CloudError::Range { .. })));
        assert!(matches!(CloudSettings { voxel_m: 0.0, ..CloudSettings::DEFAULT }.checked(), Err(CloudError::Voxel(_))));
    }
}
