// A ray cast onto a heightfield hits it wherever it crosses the surface, including a ray running
// along a cell edge, through a grid vertex, along the field's outer border, and a ray reaching the
// field through a pose that rounds it onto such a line. A ray holding a coordinate that is not
// finite hits nothing.
//
// Every field here samples a plane, so the surface a ray must hit is known in closed form and the
// expected answer does not depend on the triangle test under examination.

use parry3d::math::{Pose, Real, Rotation, Vector};
use parry3d::query::{Ray, RayCast};
use parry3d::shape::HeightField;
use parry3d::utils::Array2;

/// A `n` x `n` heightfield over `scale`, sampling the plane `y = a * x + b * z + c` (in the
/// field's local frame, scale applied).
fn planar_field(n: usize, scale: Vector, (a, b, c): (Real, Real, Real)) -> HeightField {
    let mut heights = Array2::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let x = (-0.5 + j as Real / (n - 1) as Real) * scale.x;
            let z = (-0.5 + i as Real / (n - 1) as Real) * scale.z;
            heights[(i, j)] = (a * x + b * z + c) / scale.y;
        }
    }
    HeightField::new(heights, scale)
}

/// The time of impact of `ray` on the plane `y = a * x + b * z + c`.
fn plane_toi(ray: &Ray, (a, b, c): (Real, Real, Real)) -> Real {
    let o = ray.origin;
    let d = ray.dir;
    (a * o.x + b * o.z + c - o.y) / (d.y - a * d.x - b * d.z)
}

/// Casts every ray onto `field` and returns how many miss it or hit it further than `tol` from
/// the plane.
fn count_misses(field: &HeightField, plane: (Real, Real, Real), rays: &[Ray], tol: Real) -> usize {
    rays.iter()
        .filter(|ray| match field.cast_local_ray(ray, 1.0e4, true) {
            Some(toi) => (toi - plane_toi(ray, plane)).abs() > tol,
            None => true,
        })
        .count()
}

#[test]
fn a_downward_ray_on_a_grid_vertex_or_cell_edge_hits_a_flat_heightfield() {
    // 193 x 193 samples over 96 m: 0.5 m cells, whole metres on cell edges.
    let plane = (0.0, 0.0, 0.0);
    let field = planar_field(193, Vector::new(96.0, 1.0, 96.0), plane);
    let down = Vector::new(0.0, -1.0, 0.0);

    for (dx, dz) in [(0.0, 0.0), (0.0, 0.25), (0.25, 0.0), (0.25, 0.25)] {
        let mut rays = vec![];
        for x in (-40..=40).step_by(10) {
            for z in (-40..=40).step_by(10) {
                let origin = Vector::new(x as Real + dx, 3.0, z as Real + dz);
                rays.push(Ray::new(origin, down));
            }
        }
        assert_eq!(
            count_misses(&field, plane, &rays, 1.0e-4),
            0,
            "rays offset by ({dx}, {dz}) from the 10 m lattice"
        );
    }

    // Every grid vertex and the middle of every cell edge along one row and one column, as the
    // field's own `x_at` / `z_at` state them, the outer border included.
    let mut rays = vec![];
    for k in 0..=192 {
        for (x, z) in [
            (field.x_at(k), field.z_at(96)),
            (field.x_at(96), field.z_at(k)),
            (field.x_at(k), 0.1),
            (0.1, field.z_at(k)),
        ] {
            rays.push(Ray::new(Vector::new(x, 3.0, z), down));
        }
    }
    assert_eq!(count_misses(&field, plane, &rays, 1.0e-4), 0);
}

#[test]
fn a_downward_ray_on_a_cell_edge_hits_a_sloped_heightfield_at_its_surface() {
    // 300 x 300 samples over 598 m, standing on the texel centres of a 600 m ground of 2 m
    // texels: 2 m cells, odd whole metres on cell edges, and a slope so that a hit on the wrong
    // side of an edge reads a different height.
    let plane = (0.05, -0.03, 7.0);
    let field = planar_field(300, Vector::new(598.0, 1.0, 598.0), plane);
    let down = Vector::new(0.0, -1.0, 0.0);

    let mut rays = vec![];
    for x in (-299..=297).step_by(6) {
        for z in (-299..=297).step_by(6) {
            rays.push(Ray::new(Vector::new(x as Real, 100.0, z as Real), down));
            rays.push(Ray::new(
                Vector::new(x as Real, 100.0, z as Real + 0.37),
                down,
            ));
            rays.push(Ray::new(
                Vector::new(x as Real + 0.37, 100.0, z as Real),
                down,
            ));
        }
    }
    // The four corners and the middle of each side of the outer border.
    for (x, z) in [
        (-299.0, -299.0),
        (299.0, -299.0),
        (-299.0, 299.0),
        (299.0, 299.0),
        (-299.0, 2.0),
        (299.0, 2.0),
        (2.0, -299.0),
        (2.0, 299.0),
    ] {
        rays.push(Ray::new(Vector::new(x, 100.0, z), down));
    }

    assert_eq!(count_misses(&field, plane, &rays, 1.0e-3), 0);
}

#[test]
fn an_oblique_ray_along_a_cell_edge_or_through_grid_vertices_hits_a_heightfield() {
    // 2 m cells on odd whole metres, as on the field above, over its middle 32 m.
    let plane = (0.1, 0.2, 1.0);
    let field = planar_field(300, Vector::new(598.0, 1.0, 598.0), plane);

    let mut rays = vec![];
    for k in 142..158 {
        let edge = field.z_at(k);
        // Along a row boundary, in both directions.
        rays.push(Ray::new(
            Vector::new(-9.0, 10.0, edge),
            Vector::new(1.0, -1.0, 0.0),
        ));
        rays.push(Ray::new(
            Vector::new(9.0, 10.0, edge),
            Vector::new(-1.0, -1.0, 0.0),
        ));
        // Along a column boundary, in both directions.
        let edge = field.x_at(k);
        rays.push(Ray::new(
            Vector::new(edge, 10.0, -9.0),
            Vector::new(0.0, -1.0, 1.0),
        ));
        rays.push(Ray::new(
            Vector::new(edge, 10.0, 9.0),
            Vector::new(0.0, -1.0, -1.0),
        ));
        // Through a run of grid vertices along the diagonal the cells' own diagonals do not
        // follow, meeting the surface at the vertex `(k, 299 - k)`.
        let (x, z) = (field.x_at(k), field.z_at(299 - k));
        let vertex = Vector::new(x, plane.0 * x + plane.1 * z + plane.2, z);
        for dir in [Vector::new(1.0, -2.0, 1.0), Vector::new(-1.0, -2.0, -1.0)] {
            rays.push(Ray::new(vertex - dir * 3.0, dir));
        }
    }

    assert_eq!(count_misses(&field, plane, &rays, 1.0e-4), 0);
}

#[test]
fn a_ray_cast_down_onto_a_turned_heightfield_hits_it_at_its_cell_edges() {
    // 33 x 33 samples over 128 m (4 m cells), a quarter turn about +Y, far from the origin: the
    // world-down ray reaches the field's frame rounded onto, or a hair beside, a cell edge.
    let plane = (0.05, 0.02, 20.0);
    let field = planar_field(33, Vector::new(128.0, 1.0, 128.0), plane);
    let half = (0.5 as Real).sqrt();
    let pose = Pose::from_parts(
        Vector::new(5000.0, 0.0, 5800.0),
        Rotation::from_xyzw(0.0, half, 0.0, half),
    );
    let down = Vector::new(0.0, -1.0, 0.0);

    let mut misses = vec![];
    let mut offsets = vec![(8.0, 10.3), (10.3, -20.0)];
    for a in -5..=5 {
        for b in -5..=5 {
            offsets.push((a as Real * 11.0, b as Real * 11.0));
        }
    }
    for k in -16..=16 {
        offsets.push((k as Real * 4.0, 10.3));
        offsets.push((10.3, k as Real * 4.0));
        offsets.push((k as Real * 4.0, k as Real * 4.0));
    }

    for (x, z) in offsets {
        let ray = Ray::new(Vector::new(5000.0 + x, 300.0, 5800.0 + z), down);
        let local = ray.inverse_transform_by(&pose);
        let expected = plane_toi(&local, plane);
        match field.cast_ray(&pose, &ray, 600.0, true) {
            Some(toi) if (toi - expected).abs() <= 1.0e-3 => {}
            hit => misses.push((x, z, hit)),
        }
    }

    assert!(
        misses.is_empty(),
        "rays that missed the turned field: {misses:?}"
    );
}

#[test]
fn a_ray_holding_a_coordinate_that_is_not_finite_misses_a_heightfield() {
    // The rays would hit the field if each coordinate that is not finite were a finite number.
    let plane = (0.05, -0.03, 2.0);
    let field = planar_field(17, Vector::new(16.0, 1.0, 16.0), plane);
    let (nan, inf) = (Real::NAN, Real::INFINITY);

    let rays = [
        (Vector::new(nan, 10.0, 0.3), Vector::new(0.3, -1.0, 0.2)),
        (Vector::new(0.3, nan, 0.3), Vector::new(0.3, -1.0, 0.2)),
        (Vector::new(0.3, 10.0, nan), Vector::new(0.3, -1.0, 0.2)),
        (Vector::new(nan, 10.0, 0.3), Vector::new(0.0, -1.0, 0.0)),
        (Vector::new(0.3, 10.0, nan), Vector::new(0.0, -1.0, 0.0)),
        (Vector::new(0.3, 10.0, 0.3), Vector::new(nan, -1.0, 0.2)),
        (Vector::new(0.3, 10.0, 0.3), Vector::new(0.3, nan, 0.2)),
        (Vector::new(0.3, 10.0, 0.3), Vector::new(0.3, -1.0, nan)),
        (Vector::new(-inf, 10.0, 0.3), Vector::new(1.0, -1.0, 0.2)),
        (Vector::new(0.3, inf, 0.3), Vector::new(0.3, -1.0, 0.2)),
        (Vector::new(0.3, 10.0, 0.3), Vector::new(0.3, -inf, 0.2)),
    ];

    // Cast on a thread of its own, so a cast that never returns fails the test.
    let (sender, answers) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let hits: Vec<_> = rays
            .iter()
            .filter_map(|(origin, dir)| {
                let ray = Ray::new(*origin, *dir);
                field
                    .cast_local_ray_and_get_normal(&ray, 1.0e4, true)
                    .map(|hit| (ray, hit.time_of_impact))
            })
            .collect();
        let _ = sender.send(hits);
    });
    let hits = answers
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("a cast did not return within 10 s");
    assert!(hits.is_empty(), "rays reported as hits: {hits:?}");
}
