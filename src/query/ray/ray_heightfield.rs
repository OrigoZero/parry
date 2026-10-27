use crate::math::Real;
use crate::query::{Ray, RayCast, RayIntersection};
use crate::shape::HeightField;
#[cfg(feature = "dim2")]
use crate::shape::{FeatureId, Segment};
#[cfg(feature = "dim3")]
use crate::{
    query::ray::ray_triangle::local_ray_intersection_with_triangle_within, shape::Triangle,
};

/// The share of a cell's width within which a ray passing by a cell boundary is also tested
/// against the cell on the other side of it.
///
/// A ray along a cell edge or through a grid vertex lies on the boundary of every cell meeting
/// there, and which of them rounding places it in decides nothing: each of them is tested.
const NEIGHBOUR_CELL_MARGIN: Real = 1.0e-3;

/// The share of an element's barycentric range by which a ray may pass outside a triangle (or a
/// segment, in 2D) of the heightfield and still hit it.
///
/// Two elements sharing an edge are tested with their own arithmetic, and a ray exactly on that
/// edge can round a hair outside both. A hit inside an element is always preferred over a hit
/// this margin accepts, so the margin only answers rays that no element contains exactly.
const EDGE_SLACK: Real = 1.0e-4;

/// One element hit by the ray, and whether the hit point lies inside the element itself rather
/// than within its [`EDGE_SLACK`].
#[derive(Copy, Clone)]
struct ElementHit {
    inter: RayIntersection,
    inside: bool,
}

impl ElementHit {
    /// Keeps the better of `self` and `best`: a hit inside an element over one in its slack, then
    /// the earliest along the ray.
    fn keep_into(self, best: &mut Option<ElementHit>) {
        let better = match best {
            None => true,
            Some(b) if b.inside != self.inside => self.inside,
            Some(b) => self.inter.time_of_impact < b.inter.time_of_impact,
        };

        if better {
            *best = Some(self);
        }
    }
}

/// The span of the ray's coordinate `origin + dir * t` over `t` in `[t_in, t_out]`.
#[inline]
fn coordinate_span(origin: Real, dir: Real, t_in: Real, t_out: Real) -> (Real, Real) {
    let a = origin + dir * t_in;
    let b = origin + dir * t_out;
    (a.min(b), a.max(b))
}

/// The ray parameter at which the ray leaves the cell `[lo, hi]` along one axis, and which way it
/// steps to the next cell (`0` when the ray does not move along that axis).
#[inline]
fn leave_cell(origin: Real, dir: Real, lo: Real, hi: Real) -> (Real, isize) {
    if dir > 0.0 {
        ((hi - origin) / dir, 1)
    } else if dir < 0.0 {
        ((lo - origin) / dir, -1)
    } else {
        (Real::MAX, 0)
    }
}

/// The index of the cell, among `count` cells whose lower boundary is `boundary(k)`, that holds
/// `x`, taking `guess` as the starting point and settling it against the boundaries themselves.
///
/// The boundaries are the coordinates the heightfield's elements are built on, so the cell found
/// agrees with them whatever rounding the guess carried. A coordinate outside the cells gives the
/// first or the last cell.
#[inline]
fn settle_cell(x: Real, guess: Real, count: usize, boundary: impl Fn(usize) -> Real) -> usize {
    let mut k = guess.floor().clamp(0.0, (count - 1) as Real) as usize;

    while k > 0 && x < boundary(k) {
        k -= 1;
    }

    while k + 1 < count && x >= boundary(k + 1) {
        k += 1;
    }

    k
}

/// The cells adjacent to `k` (among `count`) that the span `[lo, hi]`, lying in cell
/// `[boundary(k), boundary(k + 1)]`, comes within `margin` of, as an inclusive index range.
#[inline]
fn cells_near(
    k: usize,
    count: usize,
    (lo, hi): (Real, Real),
    margin: Real,
    boundary: impl Fn(usize) -> Real,
) -> (usize, usize) {
    let first = if k > 0 && lo - boundary(k) <= margin {
        k - 1
    } else {
        k
    };
    let last = if k + 1 < count && boundary(k + 1) - hi <= margin {
        k + 1
    } else {
        k
    };
    (first, last)
}

#[cfg(feature = "dim2")]
impl HeightField {
    /// The smallest x coordinate of the `j`-th segment, the coordinate [`HeightField::segment_at`]
    /// starts that segment at.
    #[inline]
    fn segment_start_x(&self, j: usize) -> Real {
        (-0.5 + self.unit_cell_width() * (j as Real)) * self.scale().x
    }

    /// The largest x coordinate of the `j`-th segment, the coordinate [`HeightField::segment_at`]
    /// ends that segment at.
    #[inline]
    fn segment_end_x(&self, j: usize) -> Real {
        (-0.5 + self.unit_cell_width() * (j as Real) + self.unit_cell_width()) * self.scale().x
    }

    /// The ray's hit on the `j`-th segment, within [`EDGE_SLACK`] of its ends.
    fn cast_ray_on_segment(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        j: usize,
    ) -> Option<ElementHit> {
        let seg: Segment = self.segment_at(j)?;
        let seg_dir = seg.scaled_direction();
        let denom = ray.dir.perp_dot(seg_dir);

        if denom == 0.0 {
            return None;
        }

        let ao = seg.a - ray.origin;
        let s = ao.perp_dot(seg_dir) / denom;
        let t = ao.perp_dot(ray.dir) / denom;

        if s < 0.0 || s > max_time_of_impact || t < -EDGE_SLACK || t > 1.0 + EDGE_SLACK {
            return None;
        }

        let n = seg.normal()?;
        // The feature is the hit segment's own side; `subshape` says which segment.
        let fid = if n.dot(ray.dir) > 0.0 { 1 } else { 0 };

        Some(ElementHit {
            inter: RayIntersection::new(s, n, FeatureId::Face(fid)).with_subshape(j as u32),
            inside: (0.0..=1.0).contains(&t),
        })
    }
}

#[cfg(feature = "dim2")]
impl RayCast for HeightField {
    #[inline]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        _: bool,
    ) -> Option<RayIntersection> {
        let ncells = self.num_cells();
        let margin = self.cell_width().abs() * NEIGHBOUR_CELL_MARGIN;

        let mut aabb = self.local_aabb();
        aabb.mins.x -= margin;
        aabb.maxs.x += margin;

        let (min_t, max_t) = aabb.clip_ray_parameters(ray)?;
        let max_t = max_t.min(max_time_of_impact);

        if min_t > max_t {
            return None;
        }

        let entry = ray.point_at(min_t);
        let mut j = settle_cell(
            entry.x,
            (entry.x / self.scale().x + 0.5) / self.unit_cell_width(),
            ncells,
            |k| self.segment_start_x(k),
        );
        let mut t_in = min_t;
        // The segments the previous step tested, every one of them missed.
        let mut tested: Option<(usize, usize)> = None;

        loop {
            let (toi_x, step) = leave_cell(
                ray.origin.x,
                ray.dir.x,
                self.segment_start_x(j),
                self.segment_end_x(j),
            );
            let t_out = toi_x.min(max_t).max(t_in);

            let span = coordinate_span(ray.origin.x, ray.dir.x, t_in, t_out);
            let (first, last) = cells_near(j, ncells, span, margin, |k| self.segment_start_x(k));

            let mut best = None;
            for k in first..=last {
                if tested.is_some_and(|(k0, k1)| (k0..=k1).contains(&k)) {
                    continue;
                }

                if let Some(hit) = self.cast_ray_on_segment(ray, max_time_of_impact, k) {
                    hit.keep_into(&mut best);
                }
            }

            if let Some(best) = best {
                return Some(best.inter);
            }

            if t_out >= max_t || step == 0 {
                return None;
            }

            tested = Some((first, last));

            if step > 0 {
                if j + 1 >= ncells {
                    return None;
                }
                j += 1;
            } else {
                if j == 0 {
                    return None;
                }
                j -= 1;
            }

            t_in = t_out;
        }
    }
}

#[cfg(feature = "dim3")]
impl HeightField {
    /// The ray's hits on the two triangles of the cell `(i, j)`, within [`EDGE_SLACK`] of their
    /// edges, kept into `best`.
    fn cast_ray_on_cell(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        (i, j): (usize, usize),
        best: &mut Option<ElementHit>,
    ) {
        let tris = self.triangles_at(i, j);

        for (tri, left) in [(tris.0, true), (tris.1, false)] {
            let Some(Triangle { a, b, c }) = tri else {
                continue;
            };

            let Some((mut inter, _, inside)) =
                local_ray_intersection_with_triangle_within(a, b, c, ray, EDGE_SLACK)
            else {
                continue;
            };

            if inter.time_of_impact > max_time_of_impact {
                continue;
            }

            // The feature stays the one the triangle reported; `subshape` says which triangle it
            // belongs to.
            inter.subshape = self.triangle_id(i, j, left);
            ElementHit { inter, inside }.keep_into(best);
        }
    }
}

#[cfg(feature = "dim3")]
impl RayCast for HeightField {
    #[inline]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        _: bool,
    ) -> Option<RayIntersection> {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        let margin_x = self.cell_width().abs() * NEIGHBOUR_CELL_MARGIN;
        let margin_z = self.cell_height().abs() * NEIGHBOUR_CELL_MARGIN;

        let mut aabb = self.local_aabb();
        aabb.mins.x -= margin_x;
        aabb.maxs.x += margin_x;
        aabb.mins.z -= margin_z;
        aabb.maxs.z += margin_z;

        let (min_t, max_t) = aabb.clip_ray_parameters(ray)?;
        let max_t = max_t.min(max_time_of_impact);

        if min_t > max_t {
            return None;
        }

        let scale = self.scale();
        let entry = ray.point_at(min_t);
        let mut i = settle_cell(
            entry.z,
            (entry.z / scale.z + 0.5) / self.unit_cell_height(),
            nrows,
            |k| self.z_at(k),
        );
        let mut j = settle_cell(
            entry.x,
            (entry.x / scale.x + 0.5) / self.unit_cell_width(),
            ncols,
            |k| self.x_at(k),
        );
        let mut t_in = min_t;
        // The rows and columns of the cells the previous step tested, every one of them missed.
        let mut tested: Option<((usize, usize), (usize, usize))> = None;

        // Walk the cells under the ray in the order it crosses them. In each, the ray is tested
        // against the cell and against every neighbour it passes within a margin of, so a ray
        // along a cell edge or through a grid vertex meets every cell sharing it. Each step tests
        // the cells of its own the previous step left out, since the walk reaches a step only once
        // the ray has missed every cell the previous one tested.
        loop {
            let (toi_x, step_j) =
                leave_cell(ray.origin.x, ray.dir.x, self.x_at(j), self.x_at(j + 1));
            let (toi_z, step_i) =
                leave_cell(ray.origin.z, ray.dir.z, self.z_at(i), self.z_at(i + 1));
            let t_out = toi_x.min(toi_z).min(max_t).max(t_in);

            let span_x = coordinate_span(ray.origin.x, ray.dir.x, t_in, t_out);
            let span_z = coordinate_span(ray.origin.z, ray.dir.z, t_in, t_out);
            let (first_j, last_j) = cells_near(j, ncols, span_x, margin_x, |k| self.x_at(k));
            let (first_i, last_i) = cells_near(i, nrows, span_z, margin_z, |k| self.z_at(k));

            let mut best = None;
            for ci in first_i..=last_i {
                for cj in first_j..=last_j {
                    let seen = tested.is_some_and(|((i0, i1), (j0, j1))| {
                        (i0..=i1).contains(&ci) && (j0..=j1).contains(&cj)
                    });

                    if !seen {
                        self.cast_ray_on_cell(ray, max_time_of_impact, (ci, cj), &mut best);
                    }
                }
            }

            if let Some(best) = best {
                return Some(best.inter);
            }

            if t_out >= max_t {
                return None;
            }

            tested = Some(((first_i, last_i), (first_j, last_j)));

            // The ray leaves through the boundary it reaches first, or through both at a grid
            // vertex.
            if step_j != 0 && toi_x <= toi_z {
                if step_j > 0 {
                    if j + 1 >= ncols {
                        return None;
                    }
                    j += 1;
                } else {
                    if j == 0 {
                        return None;
                    }
                    j -= 1;
                }
            }

            if step_i != 0 && toi_z <= toi_x {
                if step_i > 0 {
                    if i + 1 >= nrows {
                        return None;
                    }
                    i += 1;
                } else {
                    if i == 0 {
                        return None;
                    }
                    i -= 1;
                }
            }

            t_in = t_out;
        }
    }
}
