use crate::config::Config;
use crate::types::{Dim, Lightness, Mask};

use std::collections::VecDeque;

pub fn nms(dim: Dim, mag: &Lightness, orient: &[(i8, i8)], supp: &mut Lightness) {
    let w = dim.w;
    let h = dim.h;

    for y in 0..h {
        for x in 0..w {
            let i = x + y * w;

            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                supp[i] = 0.0;
                continue;
            }

            let cur = mag[i];
            let (dx, dy) = orient[i];

            let i1 = ((x as isize - dx as isize) + (y as isize - dy as isize) * w as isize) as usize;
            let i2 = ((x as isize + dx as isize) + (y as isize + dy as isize) * w as isize) as usize;

            let n1 = mag[i1];
            let n2 = mag[i2];

            supp[i] = if cur >= n1 && cur >= n2 {
                cur
            } else {
                0.0
            };
        }
    }
}

pub fn hysteresis(
    config: &Config,
    dim: Dim,
    edges: &Lightness,
    strong: &mut Mask,
    weak: &mut Mask,
    dq: &mut VecDeque<(usize, usize)>,
    out: &mut Mask,
) {
    let max = edges.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let high = config.hyst_high * max;
    let low = config.hyst_low * max;

    let w = dim.w;
    let h = dim.h;

    dq.clear();

    for y in 0..h {
        let r = y * w;

        for x in 0..w {
            let i = x + r;

            let v = edges[i];

            let str = v > high;
            let wk = v > low && v <= high;

            strong[i] = if str { 1 } else { 0 };
            weak[i] = if wk { 1 } else { 0 };
            out[i] = if str { 1 } else { 0 };

            if str {
                dq.push_back((y, x));
            }
        }
    }

    while let Some((y, x)) = dq.pop_front() {
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            let r = ny * w;

            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let i = nx + r;

                if weak[i] == 1 && out[i] == 0 {
                    out[i] = 1;
                    dq.push_back((ny, nx));
                }
            }
        }
    }
}
