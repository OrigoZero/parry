use crate::math::Real;
#[cfg(feature = "dim2")]
use crate::math::Vector;
#[cfg(feature = "dim3")]
use crate::math::Vector;
use crate::query::{Ray, RayCast, RayIntersection};
use crate::shape::{FeatureId, Triangle};

impl RayCast for Triangle {
    #[inline]
    #[cfg(feature = "dim2")]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        solid: bool,
    ) -> Option<RayIntersection> {
        let edges = self.edges();

        if solid {
            // Check if ray starts in triangle
            let perp_sign1 = edges[0]
                .scaled_direction()
                .perp_dot(ray.origin - edges[0].a)
                > 0.0;
            let perp_sign2 = edges[1]
                .scaled_direction()
                .perp_dot(ray.origin - edges[1].a)
                > 0.0;
            let perp_sign3 = edges[2]
                .scaled_direction()
                .perp_dot(ray.origin - edges[2].a)
                > 0.0;

            if perp_sign1 == perp_sign2 && perp_sign1 == perp_sign3 {
                return Some(RayIntersection::new(0.0, Vector::Y, FeatureId::Face(0)));
            }
        }

        let mut best = None;
        let mut smallest_toi = Real::MAX;

        for edge in &edges {
            if let Some(inter) = edge.cast_local_ray_and_get_normal(ray, max_time_of_impact, solid)
            {
                if inter.time_of_impact < smallest_toi {
                    smallest_toi = inter.time_of_impact;
                    best = Some(inter);
                }
            }
        }

        best
    }

    #[inline]
    #[cfg(feature = "dim3")]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        _: bool,
    ) -> Option<RayIntersection> {
        let inter = local_ray_intersection_with_triangle(self.a, self.b, self.c, ray)?.0;

        if inter.time_of_impact <= max_time_of_impact {
            Some(inter)
        } else {
            None
        }
    }
}

/// Computes the intersection between a triangle and a ray.
///
/// If an intersection is found, the time of impact, the normal and the barycentric coordinates of
/// the intersection point are returned.
///
/// The test is watertight: a ray crossing a mesh along an edge two triangles share, or through a
/// vertex several share, hits at least one of them, whatever the triangles' winding. Each edge
/// is tested through an edge function of its two end points alone, taken in the plane across the
/// ray (Woop, Benthin and Wald, "Watertight Ray/Triangle Intersection", JCGT 2013), so the two
/// triangles on either side of an edge compute the same value for it with opposite signs, and a
/// value of exactly zero counts as inside both.
#[cfg(feature = "dim3")]
pub fn local_ray_intersection_with_triangle(
    a: Vector,
    b: Vector,
    c: Vector,
    ray: &Ray,
) -> Option<(RayIntersection, Vector)> {
    let ab = b - a;
    let ac = c - a;

    // normal
    let n = ab.cross(ac);
    let d = n.dot(ray.dir);

    // the normal and the ray direction are parallel
    if d == 0.0 {
        return None;
    }

    let ap = ray.origin - a;
    let t = ap.dot(n);

    // the ray does not intersect the halfspace defined by the triangle
    if (t < 0.0 && d < 0.0) || (t > 0.0 && d > 0.0) {
        return None;
    }

    let [u, v, w] = watertight_edge_functions(a, b, c, ray)?;
    let det = u + v + w;
    if det == 0.0 {
        return None;
    }

    let fid = if d < 0.0 { 0 } else { 1 };
    let d = d.abs();
    let (time_of_impact, normal) = if t < 0.0 {
        (-t * (1.0 / d), -n.normalize())
    } else {
        (t * (1.0 / d), n.normalize())
    };

    let inv_det = 1.0 / det;
    Some((
        RayIntersection::new(time_of_impact, normal, FeatureId::Face(fid)),
        Vector::new(u * inv_det, v * inv_det, w * inv_det),
    ))
}

/// The three edge functions of the triangle `a, b, c` seen along `ray`, each the weight of the
/// vertex opposite its edge: `[weight of a, weight of b, weight of c]`, unnormalized. `None` when
/// the ray passes outside the triangle, or when a value is not finite.
///
/// The vertices are taken relative to the ray origin and sheared so the ray runs along the axis
/// its direction is longest on. An edge function then reads the two end points of its edge and
/// nothing else, so an edge shared by two triangles yields the same magnitude in both, of
/// opposite sign when the triangles are wound alike and of the same sign when they are not. The
/// ray lies inside a triangle when its three values share one sign, zero counting as either.
#[cfg(feature = "dim3")]
fn watertight_edge_functions(a: Vector, b: Vector, c: Vector, ray: &Ray) -> Option<[Real; 3]> {
    let dir = ray.dir;
    let abs_dir = dir.abs();
    let kz = if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
        0
    } else if abs_dir.y >= abs_dir.z {
        1
    } else {
        2
    };
    let kx = (kz + 1) % 3;
    let ky = (kx + 1) % 3;

    let sx = dir[kx] / dir[kz];
    let sy = dir[ky] / dir[kz];

    let shear = |p: Vector| {
        let p = p - ray.origin;
        (p[kx] - sx * p[kz], p[ky] - sy * p[kz])
    };
    let (ax, ay) = shear(a);
    let (bx, by) = shear(b);
    let (cx, cy) = shear(c);

    let u = cx * by - cy * bx;
    let v = ax * cy - ay * cx;
    let w = bx * ay - by * ax;

    if !(u.is_finite() && v.is_finite() && w.is_finite()) {
        return None;
    }
    if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
        return None;
    }

    Some([u, v, w])
}

/// Computes the intersection between a triangle and a ray, counting a ray that passes outside the
/// triangle by at most `slack` (a share of the triangle's barycentric range) as hitting it.
///
/// Returns the intersection, the barycentric coordinates of the intersection point, and whether
/// that point lies inside the triangle itself (every barycentric coordinate in `[0, 1]`). The
/// slack lets two triangles sharing an edge answer a ray along that edge although rounding puts
/// the ray a hair outside both.
#[cfg(feature = "dim3")]
pub(crate) fn local_ray_intersection_with_triangle_within(
    a: Vector,
    b: Vector,
    c: Vector,
    ray: &Ray,
    slack: Real,
) -> Option<(RayIntersection, Vector, bool)> {
    let ab = b - a;
    let ac = c - a;

    // normal
    let n = ab.cross(ac);
    let d = n.dot(ray.dir);

    // the normal and the ray direction are parallel
    if d == 0.0 {
        return None;
    }

    let ap = ray.origin - a;
    let t = ap.dot(n);

    // the ray does not intersect the halfspace defined by the triangle
    if (t < 0.0 && d < 0.0) || (t > 0.0 && d > 0.0) {
        return None;
    }

    let fid = if d < 0.0 { 0 } else { 1 };

    let d = d.abs();
    let margin = slack * d;

    //
    // intersection: compute barycentric coordinates
    //
    let e = -ray.dir.cross(ap);

    let (mut v, mut w, time_of_impact, normal) = if t < 0.0 {
        let v = -ac.dot(e);

        if v < -margin || v > d + margin {
            return None;
        }

        let w = ab.dot(e);

        if w < -margin || v + w > d + margin {
            return None;
        }

        (v, w, -t * (1.0 / d), -n.normalize())
    } else {
        let v = ac.dot(e);

        if v < -margin || v > d + margin {
            return None;
        }

        let w = -ab.dot(e);

        if w < -margin || v + w > d + margin {
            return None;
        }

        (v, w, t * (1.0 / d), n.normalize())
    };

    let inside = v >= 0.0 && w >= 0.0 && v + w <= d;
    let invd = 1.0 / d;
    v *= invd;
    w *= invd;

    Some((
        RayIntersection::new(time_of_impact, normal, FeatureId::Face(fid)),
        Vector::new(-v - w + 1.0, v, w),
        inside,
    ))
}
