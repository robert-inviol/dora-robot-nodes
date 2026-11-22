//! Cutting an object's view out of a keyframe's colour image: the box around its mask, grown by a margin, with the
//! pixels outside the mask dimmed or blanked so the object stands out.

use eyre::{bail, eyre, Result};
use jpeg_decoder::PixelFormat;
use jpeg_encoder::{ColorType, Encoder};
use serde::Deserialize;

/// Dimmed surroundings keep this share of their brightness: dark enough that the object stands out, light enough
/// that what it stands on and its scale stay readable.
const DIMMED_BRIGHTNESS: f32 = 0.35;

/// The grey a blank background is filled with, neutral so it suggests no colour of its own.
const BLANK_GREY: u8 = 128;

/// What a view shows around the object's mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CropBackground {
    /// The surroundings at reduced brightness.
    Dimmed,
    /// Neutral grey in place of the surroundings.
    Blank,
}

/// A rectangle of pixels: columns `left..right`, rows `top..bottom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelBox {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

impl PixelBox {
    /// The smallest box around the set pixels of a row-major mask, if any are set.
    pub fn around(width: usize, pixels: impl Iterator<Item = usize>) -> Option<PixelBox> {
        pixels.fold(None, |found: Option<PixelBox>, index| {
            let (row, column) = (index / width, index % width);
            Some(match found {
                None => PixelBox { left: column, top: row, right: column + 1, bottom: row + 1 },
                Some(b) => PixelBox {
                    left: b.left.min(column),
                    top: b.top.min(row),
                    right: b.right.max(column + 1),
                    bottom: b.bottom.max(row + 1),
                },
            })
        })
    }

    pub fn width(&self) -> usize {
        self.right - self.left
    }

    pub fn height(&self) -> usize {
        self.bottom - self.top
    }

    /// The box grown on every side by `margin` times its own width or height, kept inside a `width` by `height`
    /// image.
    pub fn grown(&self, margin: f32, width: usize, height: usize) -> PixelBox {
        let across = (self.width() as f32 * margin).round() as usize;
        let down = (self.height() as f32 * margin).round() as usize;
        PixelBox {
            left: self.left.saturating_sub(across),
            top: self.top.saturating_sub(down),
            right: (self.right + across).min(width),
            bottom: (self.bottom + down).min(height),
        }
    }
}

/// Decoded colour, three bytes per pixel, row major.
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl RgbImage {
    /// The colour JPEG of a keyframe, which must be RGB and the size of its depth image.
    pub fn decode(jpeg: &[u8], width: usize, height: usize) -> Result<RgbImage> {
        let mut decoder = jpeg_decoder::Decoder::new(jpeg);
        let pixels = decoder.decode().map_err(|e| eyre!("decoding the keyframe colour JPEG: {e}"))?;
        let info = decoder.info().ok_or_else(|| eyre!("the keyframe colour JPEG has no header"))?;
        if info.pixel_format != PixelFormat::RGB24 {
            bail!("the keyframe colour JPEG is {:?}, not RGB", info.pixel_format);
        }
        if (info.width as usize, info.height as usize) != (width, height) {
            bail!("the keyframe colour JPEG is {}x{}, its depth {width}x{height}", info.width, info.height);
        }
        Ok(RgbImage { width, height, pixels })
    }

    /// The area as a JPEG, the pixels for which `in_object` (given a row-major pixel index of the whole image) is
    /// true as they are and the rest shown as `background` says.
    pub fn isolate_jpeg(&self, area: PixelBox, in_object: impl Fn(usize) -> bool, background: CropBackground, quality: u8) -> Result<Vec<u8>> {
        assert!(area.right <= self.width && area.bottom <= self.height, "{area:?} is outside {}x{}", self.width, self.height);
        let mut pixels = Vec::with_capacity(3 * area.width() * area.height());
        for row in area.top..area.bottom {
            for column in area.left..area.right {
                let index = row * self.width + column;
                let rgb = &self.pixels[3 * index..3 * index + 3];
                match (in_object(index), background) {
                    (true, _) => pixels.extend_from_slice(rgb),
                    (false, CropBackground::Dimmed) => pixels.extend(rgb.iter().map(|&c| (c as f32 * DIMMED_BRIGHTNESS).round() as u8)),
                    (false, CropBackground::Blank) => pixels.extend([BLANK_GREY; 3]),
                }
            }
        }
        encode_rgb_jpeg(area.width(), area.height(), &pixels, quality)
    }
}

pub fn encode_rgb_jpeg(width: usize, height: usize, pixels: &[u8], quality: u8) -> Result<Vec<u8>> {
    let mut jpeg = Vec::new();
    Encoder::new(&mut jpeg, quality)
        .encode(pixels, width as u16, height as u16, ColorType::Rgb)
        .map_err(|e| eyre!("encoding a JPEG: {e}"))?;
    Ok(jpeg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_spans_the_outermost_pixels() {
        let found = PixelBox::around(10, [12, 35, 17].into_iter()).unwrap();
        assert_eq!(found, PixelBox { left: 2, top: 1, right: 8, bottom: 4 });
    }

    /// A 32x24 image, red on the left half and blue on the right, with every pixel of the left half in the object.
    fn red_object_beside_blue() -> (RgbImage, impl Fn(usize) -> bool) {
        let (width, height) = (32, 24);
        let pixels: Vec<u8> = (0..width * height).flat_map(|i| if i % width < 16 { [200, 30, 30] } else { [30, 30, 200] }).collect();
        let image = RgbImage::decode(&encode_rgb_jpeg(width, height, &pixels, 95).unwrap(), width, height).unwrap();
        (image, move |index: usize| index % width < 16)
    }

    fn pixel(image: &RgbImage, column: usize, row: usize) -> [u8; 3] {
        let at = 3 * (row * image.width + column);
        [image.pixels[at], image.pixels[at + 1], image.pixels[at + 2]]
    }

    #[test]
    fn an_isolated_view_keeps_the_object_and_dims_its_surroundings() {
        let (image, in_object) = red_object_beside_blue();
        let view = image.isolate_jpeg(PixelBox { left: 8, top: 4, right: 24, bottom: 20 }, in_object, CropBackground::Dimmed, 95).unwrap();
        let back = RgbImage::decode(&view, 16, 16).unwrap();
        let object = pixel(&back, 3, 8);
        assert!(object[0] > 170 && object[2] < 60, "the object keeps its colour: {object:?}");
        let surroundings = pixel(&back, 12, 8);
        assert!(surroundings[2] > 40 && surroundings[2] < 100 && surroundings[0] < 30, "the surroundings stay blue, dimmed: {surroundings:?}");
    }

    #[test]
    fn a_blank_background_replaces_the_surroundings_with_grey() {
        let (image, in_object) = red_object_beside_blue();
        let view = image.isolate_jpeg(PixelBox { left: 8, top: 4, right: 24, bottom: 20 }, in_object, CropBackground::Blank, 95).unwrap();
        let back = RgbImage::decode(&view, 16, 16).unwrap();
        let surroundings = pixel(&back, 12, 8);
        assert!(surroundings.iter().all(|&c| c.abs_diff(BLANK_GREY) < 10), "{surroundings:?}");
    }

    #[test]
    fn a_grown_box_stays_inside_the_image() {
        let mask_box = PixelBox { left: 2, top: 10, right: 12, bottom: 14 };
        assert_eq!(mask_box.grown(0.5, 14, 100), PixelBox { left: 0, top: 8, right: 14, bottom: 16 });
        assert_eq!(mask_box.grown(0.0, 14, 100), mask_box);
    }

    #[test]
    fn a_colour_image_of_another_size_than_the_depth_is_rejected() {
        let jpeg = encode_rgb_jpeg(8, 8, &[0; 8 * 8 * 3], 90).unwrap();
        let error = RgbImage::decode(&jpeg, 16, 8).err().unwrap();
        assert!(error.to_string().contains("is 8x8, its depth 16x8"), "{error}");
    }
}
