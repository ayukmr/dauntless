use crate::types::{Dim, Lightness, Mask};

use std::collections::VecDeque;

pub fn nms(dim: Dim, gx: &Lightness, gy: &Lightness, mag: &Lightness, low: f32, supp: &mut Lightness) {
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

            if cur <= low {
                supp[i] = 0.0;
                continue;
            }

            let gx = gx[i];
            let gy = gy[i];

            let ax = gx.abs();
            let ay = gy.abs();

            let (dx, dy) = if ay <= ax * 0.4142 {
                (1, 0)
            } else if ay >= ax * 2.4142 {
                (0, 1)
            } else if gx * gy > 0.0 {
                (1, 1)
            } else {
                (1, -1)
            };

            let i1 = ((x as isize - dx as isize) + (y as isize - dy as isize) * w as isize) as usize;
            let i2 = ((x as isize + dx as isize) + (y as isize + dy as isize) * w as isize) as usize;

            let n1 = mag[i1];
            let n2 = mag[i2];

            supp[i] = if cur >= n1 && cur >= n2 { cur } else { 0.0 };
        }
    }
}

pub fn hysteresis(
    dim: Dim,
    high: f32,
    low: f32,
    edges: &Lightness,
    dq: &mut VecDeque<(usize, usize)>,
    out: &mut Mask,
) {
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

            out[i] = if str { 2 } else if wk { 1 } else { 0 };

            if str {
                dq.push_back((y, x));
            }
        }
    }

    while let Some((y, x)) = dq.pop_front() {
        for ny in y.saturating_sub(1)..(y + 2).min(h) {
            let r = ny * w;

            for nx in x.saturating_sub(1)..(x + 2).min(w) {
                let i = nx + r;

                if out[i] == 1 {
                    out[i] = 2;
                    dq.push_back((ny, nx));
                }
            }
        }
    }
}
