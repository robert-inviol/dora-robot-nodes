//! Point-to-plane ICP over the four degrees of freedom gravity leaves free: yaw and translation.

use glam::Vec3;
use rayon::prelude::*;

use crate::grid::{surface_normals, NeighbourGrid};
use crate::pose::Pose;

#[derive(Clone, Debug)]
pub struct RegistrationSettings {
    /// Correspondences farther apart than this are ignored; each stage tightens it.
    pub correspondence_schedule_m: Vec<f32>,
    pub iterations_per_stage: usize,
    /// Residuals beyond this are weighted down (Huber), so clutter and moving people pull less.
    pub huber_m: f32,
    /// A source point within this of the target surface counts towards fitness and RMSE.
    pub inlier_m: f32,
    pub normal_radius_m: f32,
    pub normal_min_neighbours: usize,
    /// Smallest over middle covariance eigenvalue above which a neighbourhood is not a surface.
    pub normal_max_flatness: f32,
    /// Keeps an under-constrained solve (a bare wall leaves motion along it free) near the seed.
    pub damping: f64,
}

impl Default for RegistrationSettings {
    fn default() -> Self {
        RegistrationSettings {
            correspondence_schedule_m: vec![0.30, 0.15, 0.08],
            iterations_per_stage: 8,
            huber_m: 0.03,
            inlier_m: 0.05,
            normal_radius_m: 0.15,
            normal_min_neighbours: 6,
            normal_max_flatness: 0.3,
            damping: 1.0,
        }
    }
}

/// Points with surface normals, ready to be registered against.
pub struct Target {
    grid: NeighbourGrid,
    normals: Vec<Option<Vec3>>,
}

impl Target {
    pub fn new(points: Vec<Vec3>, settings: &RegistrationSettings) -> Target {
        let cell = settings.correspondence_schedule_m.iter().copied().fold(settings.normal_radius_m, f32::max);
        let grid = NeighbourGrid::new(points, cell);
        let normals = surface_normals(&grid, settings.normal_radius_m, settings.normal_min_neighbours, settings.normal_max_flatness);
        Target { grid, normals }
    }

    pub fn len(&self) -> usize {
        self.grid.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grid.points.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Registration {
    pub pose: Pose,
    /// Share of source points within the inlier distance of the target surface.
    pub fitness: f32,
    pub inlier_rmse_m: f32,
    /// Smallest eigenvalue of the final normal equations: near zero when the scene leaves a direction free.
    pub weakest_constraint: f64,
}

#[derive(Default, Clone, Copy)]
struct Normal4 {
    h: [[f64; 4]; 4],
    g: [f64; 4],
    inliers: usize,
    squared: f64,
}

impl Normal4 {
    fn add(mut self, other: Normal4) -> Normal4 {
        for i in 0..4 {
            for j in 0..4 {
                self.h[i][j] += other.h[i][j];
            }
            self.g[i] += other.g[i];
        }
        self.inliers += other.inliers;
        self.squared += other.squared;
        self
    }
}

fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let pivot = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..4 {
            let factor = a[row][col] / a[col][col];
            for k in col..4 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 4];
    for row in (0..4).rev() {
        let sum: f64 = (row + 1..4).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - sum) / a[row][row];
    }
    Some(x)
}

/// Smallest eigenvalue of a symmetric 4x4 matrix, by inverse power iteration.
fn smallest_eigenvalue(a: [[f64; 4]; 4]) -> f64 {
    let mut v = [0.5; 4];
    let mut value = 0.0;
    for _ in 0..30 {
        let Some(w) = solve4(a, v) else { return 0.0 };
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        v = w.map(|x| x / norm);
        value = 1.0 / norm;
    }
    value
}

pub fn register(source: &[Vec3], target: &Target, seed: Pose, settings: &RegistrationSettings) -> Registration {
    let mut pose = seed;
    let mut last = Normal4::default();
    for &max_distance in &settings.correspondence_schedule_m {
        for _ in 0..settings.iterations_per_stage {
            last = source
                .par_iter()
                .fold(Normal4::default, |mut acc, &p| {
                    let moved = pose.apply(p);
                    let Some((index, _)) = target.grid.nearest(moved, max_distance) else { return acc };
                    let Some(normal) = target.normals[index as usize] else { return acc };
                    let r = normal.dot(moved - target.grid.points[index as usize]) as f64;
                    let around = moved - pose.translation;
                    let j = [normal.dot(Vec3::new(-around.y, around.x, 0.0)) as f64, normal.x as f64, normal.y as f64, normal.z as f64];
                    let weight = if r.abs() <= settings.huber_m as f64 { 1.0 } else { settings.huber_m as f64 / r.abs() };
                    for a in 0..4 {
                        for b in 0..4 {
                            acc.h[a][b] += weight * j[a] * j[b];
                        }
                        acc.g[a] -= weight * j[a] * r;
                    }
                    if r.abs() <= settings.inlier_m as f64 {
                        acc.inliers += 1;
                        acc.squared += r * r;
                    }
                    acc
                })
                .reduce(Normal4::default, Normal4::add);
            let mut damped = last.h;
            for (k, row) in damped.iter_mut().enumerate() {
                row[k] += settings.damping;
            }
            let Some(step) = solve4(damped, last.g) else { break };
            // The turn linearised above acts on the rotated point only, so the translation just adds.
            pose = Pose {
                yaw_rad: crate::pose::wrap_angle(pose.yaw_rad + step[0] as f32),
                translation: pose.translation + Vec3::new(step[1] as f32, step[2] as f32, step[3] as f32),
            };
            if step.iter().map(|x| x.abs()).fold(0.0, f64::max) < 1e-5 {
                break;
            }
        }
    }
    Registration {
        pose,
        fitness: if source.is_empty() { 0.0 } else { last.inliers as f32 / source.len() as f32 },
        inlier_rmse_m: if last.inliers == 0 { f32::INFINITY } else { (last.squared / last.inliers as f64).sqrt() as f32 },
        weakest_constraint: smallest_eigenvalue(last.h),
    }
}
