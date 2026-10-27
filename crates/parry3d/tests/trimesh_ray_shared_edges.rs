// A ray cast onto a triangle mesh hits it wherever it crosses the surface, including a ray running
// along an edge two triangles share and a ray through a vertex several triangles share, whatever
// way the triangles are wound and from either side.
//
// Every mesh here lies in one plane, so the surface a ray must hit is known in closed form and the
// expected answer does not depend on the triangle test under examination.

use parry3d::math::{Pose, Real, Vector};
use parry3d::query::{Ray, RayCast};
use parry3d::shape::{TriMesh, Triangle};

/// A flat polygon of `n` corners on a circle of radius `r` in the plane y = 0, triangulated as a
/// fan. With `from_centre` the fan spreads from a vertex at the centre to every corner; without it
/// the fan spreads from the first corner to every other one, the way a convex polygon is
/// triangulated from one apex. The corners are computed in f64 and rounded, as a mesh built by a
/// script or a tool arrives. `reversed` winds every triangle the other way.
fn flat_fan(n: usize, r: f64, from_centre: bool, reversed: bool) -> TriMesh {
    let mut vertices = vec![];
    if from_centre {
        vertices.push(Vector::ZERO);
    }
    for i in 0..n {
        let angle = i as f64 / n as f64 * core::f64::consts::TAU;
        vertices.push(Vector::new(
            (angle.cos() * r) as Real,
            0.0,
            (angle.sin() * r) as Real,
        ));
    }
    let mut indices = vec![];
    let tri = |a: u32, b: u32, c: u32| if reversed { [a, c, b] } else { [a, b, c] };
    if from_centre {
        for i in 1..=n as u32 {
            indices.push(tri(0, i, i % n as u32 + 1));
        }
    } else {
        for i in 1..n as u32 - 1 {
            indices.push(tri(0, i, i + 1));
        }
    }
    TriMesh::new(vertices, indices).unwrap()
}

/// Points on every edge and at every vertex of `mesh`, pulled toward the polygon's centre by
/// `inset` so the outer border is left to its own test, plus each triangle's centroid.
fn edge_and_vertex_points(mesh: &TriMesh, inset: Real) -> Vec<(Real, Real)> {
    let pull = |p: Vector| p * (1.0 - inset);
    let mut points = vec![];
    for tri in mesh.triangles() {
        for (p, q) in [(tri.a, tri.b), (tri.b, tri.c), (tri.c, tri.a)] {
            for s in [0.0, 0.25, 0.5, 0.75] {
                let at = pull(p + (q - p) * s);
                points.push((at.x, at.z));
            }
        }
        let centroid = pull((tri.a + tri.b + tri.c) / 3.0);
        points.push((centroid.x, centroid.z));
    }
    points
}

/// The rays among `rays` that miss `mesh` (under `pose`) or hit it further than `tol` from the
/// plane y = `plane_y` the mesh lies in once posed.
fn misses(mesh: &TriMesh, pose: &Pose, plane_y: Real, rays: &[Ray], tol: Real) -> Vec<Ray> {
    rays.iter()
        .filter(|ray| {
            let expected = (plane_y - ray.origin.y) / ray.dir.y;
            match mesh.cast_ray(pose, ray, 1.0e4, true) {
                Some(toi) => (toi - expected).abs() > tol,
                None => true,
            }
        })
        .copied()
        .collect()
}

#[test]
fn a_vertical_ray_on_a_shared_edge_or_vertex_hits_a_flat_fan() {
    let pose = Pose::from_translation(Vector::new(20.0, 30.0, 200.0));
    for from_centre in [false, true] {
        for reversed in [false, true] {
            let mesh = flat_fan(16, 4.0, from_centre, reversed);
            let points = edge_and_vertex_points(&mesh, 1.0e-3);
            for (start_y, dir_y) in [(5.0, -1.0), (-5.0, 1.0)] {
                let rays: Vec<Ray> = points
                    .iter()
                    .map(|(x, z)| {
                        Ray::new(
                            Vector::new(20.0 + x, 30.0 + start_y, 200.0 + z),
                            Vector::new(0.0, dir_y, 0.0),
                        )
                    })
                    .collect();
                let missed = misses(&mesh, &pose, 30.0, &rays, 1.0e-4);
                assert!(
                    missed.is_empty(),
                    "fan from_centre={from_centre} reversed={reversed}, ray going {}: {} of {} \
                     rays missed, first at {:?}",
                    if dir_y < 0.0 { "down" } else { "up" },
                    missed.len(),
                    rays.len(),
                    missed.first().map(|ray| ray.origin),
                );
            }
        }
    }
}

#[test]
fn an_oblique_ray_through_a_shared_edge_hits_a_flat_fan() {
    for from_centre in [false, true] {
        for reversed in [false, true] {
            let mesh = flat_fan(16, 4.0, from_centre, reversed);
            let points = edge_and_vertex_points(&mesh, 1.0e-3);
            for dir in [
                Vector::new(0.3, -1.0, 0.2),
                Vector::new(-0.7, -0.4, 0.1),
                Vector::new(0.05, 0.9, -0.6),
            ] {
                // Each ray is aimed to cross the plane y = 0 exactly at the point.
                let rays: Vec<Ray> = points
                    .iter()
                    .map(|(x, z)| Ray::new(Vector::new(*x, 0.0, *z) - dir * 3.0, dir))
                    .collect();
                let missed = misses(&mesh, &Pose::IDENTITY, 0.0, &rays, 1.0e-4);
                assert!(
                    missed.is_empty(),
                    "fan from_centre={from_centre} reversed={reversed}, direction {dir:?}: {} of \
                     {} rays missed, first from {:?}",
                    missed.len(),
                    rays.len(),
                    missed.first().map(|ray| ray.origin),
                );
            }
        }
    }
}

/// A small deterministic generator, so the property below sweeps the same cases every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> Real {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f64 / (1u64 << 24) as f64) as Real
    }

    fn in_range(&mut self, lo: Real, hi: Real) -> Real {
        lo + (hi - lo) * self.next()
    }

    fn vector(&mut self, extent: Real) -> Vector {
        Vector::new(
            self.in_range(-extent, extent),
            self.in_range(-extent, extent),
            self.in_range(-extent, extent),
        )
    }
}

#[test]
fn a_ray_through_the_edge_two_triangles_share_hits_one_of_them() {
    // Two triangles on either side of the edge p..q, wound alike or opposite, and a ray through a
    // point of that edge from a direction off both triangles' planes. The point is rounded onto
    // the edge however the arithmetic falls, so one of the two triangles must take the ray.
    let mut rng = Lcg(0x5eed_2134_7000_0001);
    let mut missed = 0;
    let mut cases = 0;
    for _ in 0..4000 {
        let p = rng.vector(50.0);
        let q = rng.vector(50.0);
        let left = rng.vector(50.0);
        let right = rng.vector(50.0);
        let point = p + (q - p) * rng.next();
        let dir = rng.vector(1.0);
        if dir.length() < 1.0e-2 {
            continue;
        }
        let first = Triangle::new(p, q, left);
        let second_alike = Triangle::new(q, p, right);
        let second_opposite = Triangle::new(p, q, right);
        let n1 = first.scaled_normal();
        let n2 = second_alike.scaled_normal();
        if n1.length() < 1.0e-3 || n2.length() < 1.0e-3 {
            continue;
        }
        // Keep the two triangles on opposite sides of the edge as seen along the ray, so the ray
        // crosses the surface they make at the edge rather than grazing a fold.
        let across = dir.cross(q - p);
        if across.dot(left - p).signum() == across.dot(right - p).signum()
            || n1.dot(dir).abs() < 1.0e-3 * n1.length() * dir.length()
            || n2.dot(dir).abs() < 1.0e-3 * n2.length() * dir.length()
        {
            continue;
        }
        let ray = Ray::new(point - dir * 10.0, dir);
        for second in [second_alike, second_opposite] {
            cases += 1;
            let hit = first.cast_local_ray(&ray, Real::MAX, true).is_some()
                || second.cast_local_ray(&ray, Real::MAX, true).is_some();
            if !hit {
                missed += 1;
            }
        }
    }
    assert!(cases > 1000, "only {cases} cases met the conditions");
    assert_eq!(
        missed, 0,
        "{missed} of {cases} rays through a shared edge hit neither triangle"
    );
}

#[test]
fn a_ray_hits_the_same_point_with_the_same_normal_inside_a_triangle() {
    // The hit a triangle reports inside its own area: time of impact, the normal facing back
    // toward the ray's origin, the face it was hit on, and barycentric coordinates that rebuild
    // the hit point.
    let tri = Triangle::new(
        Vector::new(0.0, 0.0, 0.0),
        Vector::new(2.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 2.0),
    );
    let down = Ray::new(Vector::new(0.5, 3.0, 0.25), Vector::new(0.0, -1.0, 0.0));
    let up = Ray::new(Vector::new(0.5, -3.0, 0.25), Vector::new(0.0, 1.0, 0.0));

    let hit_down = tri
        .cast_local_ray_and_get_normal(&down, 10.0, true)
        .unwrap();
    assert_eq!(hit_down.time_of_impact, 3.0);
    assert_eq!(hit_down.normal, Vector::new(0.0, 1.0, 0.0));

    let hit_up = tri.cast_local_ray_and_get_normal(&up, 10.0, true).unwrap();
    assert_eq!(hit_up.time_of_impact, 3.0);
    assert_eq!(hit_up.normal, Vector::new(0.0, -1.0, 0.0));
    assert_ne!(hit_down.feature, hit_up.feature);

    let (_, bcoords) =
        parry3d::query::details::local_ray_intersection_with_triangle(tri.a, tri.b, tri.c, &down)
            .unwrap();
    let rebuilt = tri.a * bcoords.x + tri.b * bcoords.y + tri.c * bcoords.z;
    assert!((rebuilt - Vector::new(0.5, 0.0, 0.25)).length() < 1.0e-6);

    // Outside the triangle, past its range, and pointing away from it.
    let outside = Ray::new(Vector::new(1.5, 3.0, 1.5), Vector::new(0.0, -1.0, 0.0));
    assert!(tri.cast_local_ray(&outside, 10.0, true).is_none());
    assert!(tri.cast_local_ray(&down, 2.0, true).is_none());
    let away = Ray::new(Vector::new(0.5, 3.0, 0.25), Vector::new(0.0, 1.0, 0.0));
    assert!(tri.cast_local_ray(&away, 10.0, true).is_none());

    // A ray holding a coordinate that is not finite hits nothing.
    let nan = Ray::new(
        Vector::new(Real::NAN, 3.0, 0.25),
        Vector::new(0.0, -1.0, 0.0),
    );
    assert!(tri.cast_local_ray(&nan, 10.0, true).is_none());
}

#[test]
fn a_ray_along_an_axis_meets_a_triangle_whose_box_ends_a_hair_beyond_it() {
    // A ray that does not travel along an axis stays at its origin coordinate there for all of its
    // length, however far it goes, so a triangle whose bounding box ends just past that coordinate
    // is still reached. Each mesh is one triangle whose apex lies `reach` past the ray, which
    // crosses the triangle `distance` from its origin.
    for (reach, distance) in [(1.0e-6, 1000.0), (1.0e-5, 1000.0), (2.0e-7, 10.0)] {
        // A floor below a ray going straight down, its apex just past the ray in z.
        let floor = TriMesh::new(
            vec![
                Vector::new(-1.0, 0.0, -1.0),
                Vector::new(1.0, 0.0, -1.0),
                Vector::new(0.0, 0.0, reach),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let down = Ray::new(Vector::new(0.0, distance, 0.0), Vector::new(0.0, -1.0, 0.0));
        let toi = floor.cast_local_ray(&down, 2.0 * distance, true);
        assert!(
            toi.is_some_and(|toi| (toi - distance).abs() <= 1.0e-5 * distance),
            "a ray {distance} above a floor reaching {reach} past it hit at {toi:?}",
        );

        // A wall beside a ray going along x, its apex just past the ray in y.
        let wall = TriMesh::new(
            vec![
                Vector::new(0.0, -1.0, -1.0),
                Vector::new(0.0, -1.0, 1.0),
                Vector::new(0.0, reach, 0.0),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let along = Ray::new(Vector::new(-distance, 0.0, 0.0), Vector::new(1.0, 0.0, 0.0));
        let toi = wall.cast_local_ray(&along, 2.0 * distance, true);
        assert!(
            toi.is_some_and(|toi| (toi - distance).abs() <= 1.0e-5 * distance),
            "a ray {distance} before a wall reaching {reach} past it hit at {toi:?}",
        );
    }
}

/// A flat grid of `cells` by `cells` unit squares in the plane y = 0, each split into two
/// triangles along the same diagonal, the way a road deck, a floor or a terrain tile is
/// triangulated as strips. Every vertex sits on whole coordinates, so the grid lines lie on the
/// faces of the triangles' bounding boxes.
fn flat_strip_grid(cells: u32) -> TriMesh {
    let mut vertices = vec![];
    for j in 0..=cells {
        for i in 0..=cells {
            vertices.push(Vector::new(i as Real, 0.0, j as Real));
        }
    }
    let mut indices = vec![];
    for j in 0..cells {
        for i in 0..cells {
            let a = j * (cells + 1) + i;
            let b = a + 1;
            let c = a + cells + 1;
            let d = c + 1;
            indices.push([a, c, b]);
            indices.push([b, c, d]);
        }
    }
    TriMesh::new(vertices, indices).unwrap()
}

#[test]
fn a_tilted_ray_through_a_grid_line_or_vertex_hits_a_flat_strip_grid() {
    // A ray tilted off the vertical, as a wheel's suspension ray is on a car that rolls or pitches,
    // crossing a flat grid on one of its grid lines, diagonals or vertices meets the grid there.
    // Each triangle's box has no thickness, so such a ray enters the box's y slab at the same
    // distance at which it reaches the box face the grid line lies on, and both query paths
    // (with and without the normal) must still reach a triangle holding that point.
    let cells = 8;
    let grid = flat_strip_grid(cells);
    let mut rng = Lcg(0x2134_7f00_d1ce_0001);
    for pose in [
        Pose::IDENTITY,
        Pose::from_translation(Vector::new(100.5, 3.25, -40.0)),
    ] {
        for max_tilt in [0.02, 0.3, 1.2] {
            let mut missed = vec![];
            for k in 0..4000 {
                let line = rng.in_range(1.0, (cells - 1) as Real).floor();
                let other = rng.in_range(1.0, (cells - 1) as Real).floor();
                let along = rng.in_range(0.5, cells as Real - 0.5);
                let (x, z) = match k % 4 {
                    0 => (line, along),
                    1 => (along, line),
                    2 => {
                        let s = rng.next();
                        (line + s, other + 1.0 - s)
                    }
                    _ => (line, other),
                };
                let angle = rng.in_range(0.0, core::f64::consts::TAU as Real);
                let tilt = rng.in_range(0.0, max_tilt);
                let mut dir = Vector::new(
                    tilt.sin() * angle.cos(),
                    -tilt.cos(),
                    tilt.sin() * angle.sin(),
                );
                if k % 2 == 1 {
                    dir = -dir;
                }
                let target = pose * Vector::new(x, 0.0, z);
                let ray = Ray::new(target - dir * 5.0, dir);
                let plain = grid.cast_ray(&pose, &ray, 10.0, true);
                let with_normal = grid.cast_ray_and_get_normal(&pose, &ray, 10.0, true);
                if plain.is_none() || with_normal.is_none() {
                    missed.push((x, z, dir, plain, with_normal.map(|hit| hit.time_of_impact)));
                }
            }
            assert!(
                missed.is_empty(),
                "pose {:?}, tilt up to {max_tilt}: {} of 4000 rays missed, first {:?}",
                pose.translation,
                missed.len(),
                missed.first(),
            );
        }
    }
}
