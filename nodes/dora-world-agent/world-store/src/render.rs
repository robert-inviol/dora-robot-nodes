//! A top-down raster of part of the map, north (+y) up, written as a PNG.

use world_model::Point;

use crate::geometry::{bounds, polygon_contains};
use crate::types::{Cell, FreeSpace};

pub const PX_PER_M: f32 = 40.0;
const GLYPH_SCALE: usize = 2;

pub type Rgb = [u8; 3];

pub const BACKGROUND: Rgb = [255, 255, 255];
pub const OCCUPIED: Rgb = [70, 70, 70];
pub const UNKNOWN: Rgb = [205, 205, 205];
pub const WALL: Rgb = [40, 40, 40];
pub const OBJECT: Rgb = [150, 120, 90];
pub const HIGHLIGHT: Rgb = [220, 40, 40];
pub const ROBOT: Rgb = [40, 80, 220];
pub const PERSON: Rgb = [40, 160, 60];
pub const DOORWAY: Rgb = [240, 170, 30];
pub const LABEL: Rgb = [0, 0, 0];

/// 3 by 5 glyphs for the characters of ids, one row per byte, high bit on the left.
fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'o' => [0b000, 0b111, 0b101, 0b101, 0b111],
        'p' => [0b111, 0b101, 0b111, 0b100, 0b100],
        't' => [0b010, 0b111, 0b010, 0b010, 0b011],
        _ => return None,
    })
}

pub struct Canvas {
    origin: Point,
    top_m: f32,
    width: usize,
    height: usize,
    pixels: Vec<Rgb>,
}

impl Canvas {
    /// A blank canvas over the box from `lo` to `hi`.
    pub fn new(lo: Point, hi: Point) -> Self {
        let width = ((hi.x_m - lo.x_m) * PX_PER_M).ceil() as usize + 1;
        let height = ((hi.y_m - lo.y_m) * PX_PER_M).ceil() as usize + 1;
        Self { origin: lo, top_m: hi.y_m, width, height, pixels: vec![BACKGROUND; width * height] }
    }

    #[cfg(test)]
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn to_map(&self, column: usize, row: usize) -> Point {
        Point {
            x_m: self.origin.x_m + (column as f32 + 0.5) / PX_PER_M,
            y_m: self.top_m - (row as f32 + 0.5) / PX_PER_M,
        }
    }

    fn to_pixel(&self, p: Point) -> (isize, isize) {
        (((p.x_m - self.origin.x_m) * PX_PER_M).floor() as isize, ((self.top_m - p.y_m) * PX_PER_M).floor() as isize)
    }

    fn put(&mut self, column: isize, row: isize, colour: Rgb) {
        if column >= 0 && row >= 0 && (column as usize) < self.width && (row as usize) < self.height {
            self.pixels[row as usize * self.width + column as usize] = colour;
        }
    }

    /// Paints every pixel by the free-space cell under it; pixels off the grid stay as they are.
    pub fn paint_free_space(&mut self, free_space: &FreeSpace) {
        for row in 0..self.height {
            for column in 0..self.width {
                let colour = match free_space.at(self.to_map(column, row)) {
                    Some(Cell::Occupied) => OCCUPIED,
                    Some(Cell::Unknown) => UNKNOWN,
                    Some(Cell::Free { .. }) | None => continue,
                };
                self.pixels[row * self.width + column] = colour;
            }
        }
    }

    pub fn fill_polygon(&mut self, polygon: &[Point], colour: Rgb) {
        let Some((lo, hi)) = bounds(polygon.iter().copied()) else { return };
        let ((c0, r1), (c1, r0)) = (self.to_pixel(lo), self.to_pixel(hi));
        for row in r0.max(0)..=r1.min(self.height as isize - 1) {
            for column in c0.max(0)..=c1.min(self.width as isize - 1) {
                if polygon_contains(polygon, self.to_map(column as usize, row as usize)) {
                    self.put(column, row, colour);
                }
            }
        }
        // A footprint smaller than a pixel still shows as one.
        if c0 == c1 && r0 == r1 {
            self.put(c0, r0, colour);
        }
    }

    pub fn line(&mut self, a: Point, b: Point, colour: Rgb) {
        let ((x0, y0), (x1, y1)) = (self.to_pixel(a), self.to_pixel(b));
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
        for i in 0..=steps {
            let x = x0 + (x1 - x0) * i / steps;
            let y = y0 + (y1 - y0) * i / steps;
            self.put(x, y, colour);
        }
    }

    pub fn outline_polygon(&mut self, polygon: &[Point], colour: Rgb) {
        for i in 0..polygon.len() {
            self.line(polygon[i], polygon[(i + 1) % polygon.len()], colour);
        }
    }

    /// Writes an id with its top-left corner just above and right of a point.
    pub fn label(&mut self, at: Point, text: &str, colour: Rgb) {
        let (x, y) = self.to_pixel(at);
        let (mut column, row) = (x + 2, y - (5 * GLYPH_SCALE) as isize - 2);
        for c in text.chars() {
            let Some(rows) = glyph(c) else { continue };
            for (dy, bits) in rows.iter().enumerate() {
                for dx in 0..3 {
                    if bits & (0b100 >> dx) != 0 {
                        for sy in 0..GLYPH_SCALE {
                            for sx in 0..GLYPH_SCALE {
                                let px = column + (dx * GLYPH_SCALE + sx) as isize;
                                let py = row + (dy * GLYPH_SCALE + sy) as isize;
                                self.put(px, py, colour);
                            }
                        }
                    }
                }
            }
            column += (4 * GLYPH_SCALE) as isize;
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut bytes, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("writing a PNG header to memory");
        writer.write_image_data(self.pixels.as_flattened()).expect("writing PNG data to memory");
        writer.finish().expect("finishing a PNG in memory");
        bytes
    }

    #[cfg(test)]
    pub fn pixel_at(&self, p: Point) -> Rgb {
        let (column, row) = self.to_pixel(p);
        self.pixels[row as usize * self.width + column as usize]
    }
}

/// Re-encodes a JPEG crop as a PNG, the image type the tools hand the agents.
pub fn jpeg_to_png(jpeg: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg);
    let pixels = decoder.decode().map_err(|e| e.to_string())?;
    let info = decoder.info().ok_or("the JPEG has no header")?;
    let colour = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => png::ColorType::Grayscale,
        jpeg_decoder::PixelFormat::RGB24 => png::ColorType::Rgb,
        other => return Err(format!("the JPEG's pixel format {other:?} is not supported")),
    };
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, u32::from(info.width), u32::from(info.height));
    encoder.set_color(colour);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(&pixels).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x_m: f32, y_m: f32) -> Point {
        Point { x_m, y_m }
    }

    #[test]
    fn a_filled_square_lands_where_the_map_says_with_north_up() {
        let mut canvas = Canvas::new(p(0.0, 0.0), p(2.0, 1.0));
        assert_eq!(canvas.size(), (81, 41));
        canvas.fill_polygon(&[p(1.5, 0.6), p(1.9, 0.6), p(1.9, 0.9), p(1.5, 0.9)], HIGHLIGHT);
        assert_eq!(canvas.pixel_at(p(1.7, 0.75)), HIGHLIGHT);
        assert_eq!(canvas.pixel_at(p(1.7, 0.25)), BACKGROUND);
    }
}
