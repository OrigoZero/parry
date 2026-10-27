// A ray cast onto a triangle mesh hits it wherever it crosses the surface, including a ray running
// along an edge two triangles share and a ray through a vertex several triangles share, whatever
// way the triangles are wound and from either side.
//
// Every mesh here lies in one plane, so the surface a ray must hit is known in closed form and the
// expected answer does not depend on the triangle test under examination.

use parry3d::math::{Pose, Real, Vector};
use parry3d::query::{Ray, RayCast};
use parry3d::shape::TriMesh;

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
