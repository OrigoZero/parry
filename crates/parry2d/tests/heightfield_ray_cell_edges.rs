// A ray cast onto a heightfield hits it wherever it crosses the surface, including a ray through
// a vertex shared by two segments and a ray on the field's outer border.
//
// Every field here samples a line, so the surface a ray must hit is known in closed form and the
// expected answer does not depend on the segment test under examination.

use parry2d::math::{Real, Vector};
use parry2d::query::{Ray, RayCast};
use parry2d::shape::HeightField;

/// A heightfield of `n` samples over `scale`, sampling the line `y = a * x + c`.
fn linear_field(n: usize, scale: Vector, (a, c): (Real, Real)) -> HeightField {
    let heights = (0..n)
        .map(|j| (a * (-0.5 + j as Real / (n - 1) as Real) * scale.x + c) / scale.y)
        .collect();
    HeightField::new(heights, scale)
}

/// The time of impact of `ray` on the line `y = a * x + c`.
fn line_toi(ray: &Ray, (a, c): (Real, Real)) -> Real {
    (a * ray.origin.x + c - ray.origin.y) / (ray.dir.y - a * ray.dir.x)
}

fn count_misses(field: &HeightField, line: (Real, Real), rays: &[Ray], tol: Real) -> usize {
    rays.iter()
        .filter(|ray| match field.cast_local_ray(ray, 1.0e4, true) {
            Some(toi) => (toi - line_toi(ray, line)).abs() > tol,
            None => true,
        })
        .count()
}

#[test]
fn a_downward_ray_on_a_vertex_hits_a_heightfield() {
    let line = (0.07, 3.0);
    let field = linear_field(301, Vector::new(600.0, 1.0), line);
    let down = Vector::new(0.0, -1.0);

    // Every odd whole metre is a vertex; the border at +-300 m included.
    let rays: Vec<_> = (-300..=300)
        .map(|x| Ray::new(Vector::new(x as Real, 100.0), down))
        .collect();
    assert_eq!(count_misses(&field, line, &rays, 1.0e-3), 0);
}

#[test]
fn an_oblique_ray_through_a_vertex_hits_a_heightfield() {
    let line = (0.2, 1.0);
    let field = linear_field(17, Vector::new(16.0, 1.0), line);

    let mut rays = vec![];
    for j in 1..16 {
        // Aimed at the vertex `j` from both sides.
        let x = -8.0 + j as Real;
        let y = 0.2 * x + 1.0;
        rays.push(Ray::new(
            Vector::new(x - 5.0, y + 5.0),
            Vector::new(1.0, -1.0),
        ));
        rays.push(Ray::new(
            Vector::new(x + 5.0, y + 5.0),
            Vector::new(-1.0, -1.0),
        ));
    }
    assert_eq!(count_misses(&field, line, &rays, 1.0e-4), 0);
}
