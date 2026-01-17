use crate::types::{Corners, Point, Quads};

use rayon::prelude::*;

const PARA_THRESH: f32 = 0.75;

pub fn filter_paras(quads: Quads) -> Quads {
    quads
        .into_par_iter()
        .filter_map(|corners| {
            let (tl, tr, bl, br) = corners;
            let t = dist(&tl, &tr);
            let b = dist(&bl, &br);
            let l = dist(&tl, &bl);
            let r = dist(&tr, &br);

            if t == 0.0 || b == 0.0 || l == 0.0 || r == 0.0 {
                return None;
            }

            let hd = (t / b).abs();
            let vd = (l / r).abs();

            if (hd - 1.0).abs() + (vd - 1.0).abs() < PARA_THRESH {
                Some(corners)
            } else {
                None
            }
        })
        .collect()
}

fn dist(a: &Point, b: &Point) -> f32 {
    let dx = a.0 as f32 - b.0 as f32;
    let dy = a.1 as f32 - b.1 as f32;

    (dx * dx + dy * dy).sqrt()
}

pub fn filter_enclosed(quads: Quads) -> Quads {
    let bds: Vec<Bounds> = quads.iter().map(bounds).collect();

    quads
        .into_par_iter()
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

struct Bounds {
    l: u32,
    r: u32,
    t: u32,
    b: u32,
}

fn bounds(corners: &Corners) -> Bounds {
    let (tl, tr, bl, br) = corners;

    let l = u32::min(tl.0, bl.0);
    let r = u32::max(tr.0, br.0);
    let t = u32::min(tl.1, tr.1);
    let b = u32::max(bl.1, br.1);

    Bounds { l, r, t, b }
}
