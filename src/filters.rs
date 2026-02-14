use crate::detector::Detector;

use std::f32;

use crate::types::{Corners, Point, Quads};

const MAX_RATIO: f32 = 1.3;
const VH_MAX_RATIO: f32 = 1.5;

impl Detector {
    pub fn filter_ratios(&self, quads: Quads) -> Quads {
        quads
            .into_iter()
            .filter(|(tl, tr, bl, br)| {
                let t = self.dist(tl, tr);
                let b = self.dist(bl, br);
                let l = self.dist(tl, bl);
                let r = self.dist(tr, br);

                if t == 0.0 || b == 0.0 || l == 0.0 || r == 0.0 {
                    return false;
                }

                self.ratio(t, b) <= MAX_RATIO
                    && self.ratio(l, r) <= MAX_RATIO
                    && self.ratio((t + b) / 2.0, (l + r) / 2.0) <= VH_MAX_RATIO
            })
            .collect()
    }

    fn dist(&self, a: &Point, b: &Point) -> f32 {
        let dx = a.0 as f32 - b.0 as f32;
        let dy = a.1 as f32 - b.1 as f32;

        (dx * dx + dy * dy).sqrt()
    }

    fn ratio(&self, a: f32, b: f32) -> f32 {
        if a > b { a / b } else { b / a }
    }

    pub fn filter_angles(&self, quads: Quads) -> Quads {
        quads
            .into_iter()
            .filter(|(tl, tr, bl, br)| {
                let corners = [tl, bl, br, tr];

                for i in 0..4 {
                    let c0 = corners[(i + 3) % 4];
                    let c1 = corners[i];
                    let c2 = corners[(i + 1) % 4];

                    let a = ((c1.0 as f32 - c0.0 as f32).abs(), (c1.1 as f32 - c0.1 as f32).abs());
                    let b = ((c2.0 as f32 - c1.0 as f32).abs(), (c2.1 as f32 - c1.1 as f32).abs());

                    let a_mag = a.0.hypot(a.1);
                    let b_mag = b.0.hypot(b.1);

                    let rad = ((a.0 * b.0 + a.1 * b.1) / (a_mag * b_mag)).acos();

                    if (rad - f32::consts::FRAC_PI_2).abs() > f32::consts::FRAC_PI_6 {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    pub fn filter_enclosed(&self, quads: Quads) -> Quads {
        let bds: Vec<Bounds> = quads.iter().map(bounds).collect();

        quads
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| {
                let bd = &bds[*idx];

                !bds
                    .iter()
                    .enumerate()
                    .any(|(j, o)| {
                        if j == *idx {
                            return false;
                        }
                        o.l < bd.l && o.r > bd.r && o.t < bd.t && o.b > bd.b
                    })
            })
            .map(|(_i, pts)| pts)
            .collect()
    }
}

struct Bounds {
    l: u32,
    r: u32,
    t: u32,
    b: u32,
}

fn bounds((tl, tr, bl, br): &Corners) -> Bounds {
    let l = u32::min(tl.0, bl.0);
    let r = u32::max(tr.0, br.0);
    let t = u32::min(tl.1, tr.1);
    let b = u32::max(bl.1, br.1);

    Bounds { l, r, t, b }
}
