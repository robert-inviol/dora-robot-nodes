//! The square input a detector takes, made from a wider colour image without stretching it: the image is shrunk by
//! a whole factor, averaging each block of pixels, and centred on a grey canvas.

use std::fmt;

/// The grey that pads a letterboxed image, as YOLO models are trained with.
pub const PAD_VALUE: u8 = 114;
const CHANNELS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Letterbox {
    pub source_width: usize,
    pub source_height: usize,
    /// The side of the square input.
    pub side: usize,
    /// Each output pixel averages a `factor` x `factor` block of the source.
    pub factor: usize,
    pub pad_x: usize,
    pub pad_y: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct UnevenShrink {
    pub source_width: usize,
    pub source_height: usize,
    pub factor: usize,
}

impl fmt::Display for UnevenShrink {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}x{} does not shrink evenly by {}", self.source_width, self.source_height, self.factor)
    }
}

impl std::error::Error for UnevenShrink {}

impl Letterbox {
    /// The smallest whole shrink that fits the source inside the square; the source has to divide by it evenly.
    pub fn new(source_width: usize, source_height: usize, side: usize) -> Result<Letterbox, UnevenShrink> {
        let factor = source_width.max(source_height).div_ceil(side).max(1);
        if source_width % factor != 0 || source_height % factor != 0 {
            return Err(UnevenShrink { source_width, source_height, factor });
        }
        let (width, height) = (source_width / factor, source_height / factor);
        Ok(Letterbox { source_width, source_height, side, factor, pad_x: (side - width) / 2, pad_y: (side - height) / 2 })
    }

    pub fn output_len(&self) -> usize {
        self.side * self.side * CHANNELS
    }

    /// Writes the letterboxed image of `rgb`, packed RGB rows of the source size, into `out`.
    pub fn fill(&self, rgb: &[u8], out: &mut [u8]) {
        assert_eq!(rgb.len(), self.source_width * self.source_height * CHANNELS, "source size");
        assert_eq!(out.len(), self.output_len(), "output size");
        let (width, height, factor) = (self.source_width / self.factor, self.source_height / self.factor, self.factor);
        let row_len = self.side * CHANNELS;
        let source_row_len = self.source_width * CHANNELS;
        let block = (factor * factor) as u32;
        out[..self.pad_y * row_len].fill(PAD_VALUE);
        out[(self.pad_y + height) * row_len..].fill(PAD_VALUE);
        let mut sums = vec![0u32; width * CHANNELS];
        for y in 0..height {
            let out_row = &mut out[(self.pad_y + y) * row_len..(self.pad_y + y + 1) * row_len];
            out_row[..self.pad_x * CHANNELS].fill(PAD_VALUE);
            out_row[(self.pad_x + width) * CHANNELS..].fill(PAD_VALUE);
            sums.fill(0);
            for source_row in rgb[y * factor * source_row_len..(y + 1) * factor * source_row_len].chunks_exact(source_row_len) {
                for (x, sum) in sums.chunks_exact_mut(CHANNELS).enumerate() {
                    for pixel in source_row[x * factor * CHANNELS..(x + 1) * factor * CHANNELS].chunks_exact(CHANNELS) {
                        sum[0] += pixel[0] as u32;
                        sum[1] += pixel[1] as u32;
                        sum[2] += pixel[2] as u32;
                    }
                }
            }
            let image = &mut out_row[self.pad_x * CHANNELS..(self.pad_x + width) * CHANNELS];
            for (value, sum) in image.iter_mut().zip(&sums) {
                *value = ((sum + block / 2) / block) as u8;
            }
        }
    }

    /// A point given as a share (0..1) of the square input, in source pixels.
    pub fn to_source(&self, x_share: f32, y_share: f32) -> (f32, f32) {
        let side = self.side as f32;
        let factor = self.factor as f32;
        ((x_share * side - self.pad_x as f32) * factor, (y_share * side - self.pad_y as f32) * factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_720p_image_halves_into_a_band_across_the_middle_of_640() {
        let letterbox = Letterbox::new(1280, 720, 640).unwrap();
        assert_eq!((letterbox.factor, letterbox.pad_x, letterbox.pad_y), (2, 0, 140));
    }

    #[test]
    fn a_source_that_does_not_divide_by_the_shrink_is_refused() {
        assert_eq!(Letterbox::new(1282, 720, 640), Err(UnevenShrink { source_width: 1282, source_height: 720, factor: 3 }));
    }

    #[test]
    fn each_output_pixel_averages_its_block_and_the_rest_is_grey() {
        // 4x2 source into a 2x2 square: shrink 2, one row of image, half a row of padding above and below rounds to 0.
        let letterbox = Letterbox::new(4, 2, 2).unwrap();
        assert_eq!((letterbox.factor, letterbox.pad_x, letterbox.pad_y), (2, 0, 0));
        let rgb: Vec<u8> = [[0, 10, 20], [2, 12, 22], [100, 0, 0], [101, 0, 0], [4, 14, 24], [6, 16, 26], [102, 0, 0], [104, 255, 0]]
            .concat();
        let mut out = vec![0; letterbox.output_len()];
        letterbox.fill(&rgb, &mut out);
        assert_eq!(&out[..6], &[3, 13, 23, 102, 64, 0]);
        assert_eq!(&out[6..], &[PAD_VALUE; 6]);
    }

    #[test]
    fn padding_surrounds_a_tall_image_on_both_sides() {
        let letterbox = Letterbox::new(2, 4, 4).unwrap();
        assert_eq!((letterbox.factor, letterbox.pad_x, letterbox.pad_y), (1, 1, 0));
        let mut out = vec![0; letterbox.output_len()];
        letterbox.fill(&[50; 2 * 4 * 3], &mut out);
        let row: Vec<u8> = [[PAD_VALUE; 3], [50; 3], [50; 3], [PAD_VALUE; 3]].concat();
        assert_eq!(out, row.repeat(4));
    }

    #[test]
    fn a_point_in_the_square_maps_back_to_source_pixels() {
        let letterbox = Letterbox::new(1280, 720, 640).unwrap();
        assert_eq!(letterbox.to_source(0.5, 0.5), (640.0, 360.0));
        assert_eq!(letterbox.to_source(0.0, 140.0 / 640.0), (0.0, 0.0));
    }
}
