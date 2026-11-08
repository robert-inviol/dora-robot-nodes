use criterion::{criterion_group, criterion_main, Criterion};
use depth_perception::synthetic::SyntheticRoom;
use depth_perception::*;

fn frame_stages(c: &mut Criterion) {
    let scene = SyntheticRoom::living_room();
    let frame = scene.render();
    let floor = FloorFrame::from_gravity_reaction(frame.accel_camera);
    let depth = DepthImage::new(scene.width, scene.height, &frame.depth_m);
    let cloud = sample_point_cloud(&depth, &scene.intrinsics, &floor, RangeLimits::D455, 2);
    let fit = find_planes(&cloud.points, &PlaneSettings::default());
    let mask_pixels: Vec<bool> = frame.object_ids.iter().map(|id| *id == Some(0)).collect();
    let mask = ObjectMask { width: scene.width, height: scene.height, pixels: &mask_pixels };

    c.bench_function("point cloud, stride 2", |b| {
        b.iter(|| sample_point_cloud(&depth, &scene.intrinsics, &floor, RangeLimits::D455, 2))
    });
    c.bench_function("planes", |b| b.iter(|| find_planes(&cloud.points, &PlaneSettings::default())));
    c.bench_function("room", |b| b.iter(|| infer_room(&fit.planes, &cloud.points, &RoomSettings::default())));
    c.bench_function("object footprint", |b| {
        b.iter(|| object_footprint(&mask, &depth, &scene.intrinsics, &floor, RangeLimits::D455, &Default::default()))
    });
}

criterion_group!(benches, frame_stages);
criterion_main!(benches);
