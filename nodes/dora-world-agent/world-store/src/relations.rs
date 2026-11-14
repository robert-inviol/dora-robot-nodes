//! How two objects stand to each other, from their footprints and heights.

use world_model::Point;

use crate::geometry::{gap_between, polygon_contains};

/// Furthest apart two objects can be and still stand next to each other.
pub const NEXT_TO_M: f32 = 0.3;
/// How far an object's underside may be from the top it rests on.
const RESTING_TOLERANCE_M: f32 = 0.08;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationKind {
    On,
    In,
    NextTo,
}

impl RelationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RelationKind::On => "on",
            RelationKind::In => "in",
            RelationKind::NextTo => "next to",
        }
    }
}

/// An object as a prism: its footprint, standing from `bottom_m` up to `top_m` above the floor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solid {
    pub footprint: [Point; 4],
    pub centre: Point,
    pub bottom_m: f32,
    pub top_m: f32,
}

/// How `subject` stands to `other`: on it, in it, next to it, or apart.
pub fn relation(subject: &Solid, other: &Solid) -> Option<RelationKind> {
    let over = polygon_contains(&other.footprint, subject.centre);
    if over && (subject.bottom_m - other.top_m).abs() <= RESTING_TOLERANCE_M {
        return Some(RelationKind::On);
    }
    let within = subject.footprint.iter().all(|p| polygon_contains(&other.footprint, *p));
    if within && subject.bottom_m >= other.bottom_m - RESTING_TOLERANCE_M && subject.top_m <= other.top_m + RESTING_TOLERANCE_M {
        return Some(RelationKind::In);
    }
    let stacked = over || polygon_contains(&subject.footprint, other.centre);
    match !stacked && gap_between(&subject.footprint, &other.footprint).0 <= NEXT_TO_M {
        true => Some(RelationKind::NextTo),
        false => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(x0: f32, y0: f32, x1: f32, y1: f32, bottom_m: f32, top_m: f32) -> Solid {
        let p = |x_m, y_m| Point { x_m, y_m };
        Solid {
            footprint: [p(x0, y0), p(x1, y0), p(x1, y1), p(x0, y1)],
            centre: p((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            bottom_m,
            top_m,
        }
    }

    #[test]
    fn a_mug_on_a_table_is_on_it_and_the_table_is_not_on_the_mug() {
        let table = solid(1.2, 1.05, 2.8, 1.95, 0.0, 0.74);
        let mug = solid(2.15, 1.55, 2.25, 1.65, 0.74, 0.84);
        assert_eq!(relation(&mug, &table), Some(RelationKind::On));
        assert_eq!(relation(&table, &mug), None);
    }

    #[test]
    fn a_box_inside_a_cupboard_is_in_it() {
        let cupboard = solid(0.0, 0.0, 1.0, 0.6, 0.0, 2.0);
        let shoebox = solid(0.2, 0.1, 0.5, 0.4, 0.3, 0.45);
        assert_eq!(relation(&shoebox, &cupboard), Some(RelationKind::In));
    }

    #[test]
    fn a_chair_by_the_table_is_next_to_it_both_ways() {
        let table = solid(1.2, 1.05, 2.8, 1.95, 0.0, 0.74);
        let chair = solid(0.95, 0.35, 1.45, 0.85, 0.0, 0.9);
        assert_eq!(relation(&chair, &table), Some(RelationKind::NextTo));
        assert_eq!(relation(&table, &chair), Some(RelationKind::NextTo));
        let far_chair = solid(5.0, 5.0, 5.5, 5.5, 0.0, 0.9);
        assert_eq!(relation(&far_chair, &table), None);
    }
}
