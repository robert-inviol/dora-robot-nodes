//! Cutting an object's view out of a keyframe's colour image.

use eyre::{bail, eyre, Result};
use jpeg_decoder::PixelFormat;
use jpeg_encoder::{ColorType, Encoder};

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

    pub fn crop_jpeg(&self, area: PixelBox, quality: u8) -> Result<Vec<u8>> {
        assert!(area.right <= self.width && area.bottom <= self.height, "{area:?} is outside {}x{}", self.width, self.height);
        let mut pixels = Vec::with_capacity(3 * area.width() * area.height());
        for row in area.top..area.bottom {
            pixels.extend_from_slice(&self.pixels[3 * (row * self.width + area.left)..3 * (row * self.width + area.right)]);
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

    #[test]
    fn a_crop_is_a_jpeg_of_the_box_size_and_colour() {
        let (width, height) = (32, 24);
        let pixels: Vec<u8> = (0..width * height).flat_map(|i| if i % width < 16 { [200, 30, 30] } else { [30, 30, 200] }).collect();
        let image = RgbImage::decode(&encode_rgb_jpeg(width, height, &pixels, 95).unwrap(), width, height).unwrap();
        let crop = image.crop_jpeg(PixelBox { left: 0, top: 4, right: 8, bottom: 20 }, 90).unwrap();
        let back = RgbImage::decode(&crop, 8, 16).unwrap();
        let centre = &back.pixels[3 * (8 * 8 + 4)..3 * (8 * 8 + 4) + 3];
        assert!(centre[0] > 150 && centre[2] < 80, "{centre:?}");
    }

    #[test]
    fn a_colour_image_of_another_size_than_the_depth_is_rejected() {
        let jpeg = encode_rgb_jpeg(8, 8, &[0; 8 * 8 * 3], 90).unwrap();
        let error = RgbImage::decode(&jpeg, 16, 8).err().unwrap();
        assert!(error.to_string().contains("is 8x8, its depth 16x8"), "{error}");
    }
}
