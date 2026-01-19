use crate::types::{Lightness, Mask};
use crate::config::cfg;

use std::collections::VecDeque;

use ndarray::{Array2, Zip};

pub fn nms(mag: &Lightness, orient: &Array2<(i32, i32)>) -> Lightness {
    let (h, w) = mag.dim();

    Zip::indexed(mag)
        .and(orient)
        .map_collect(|(y, x), &cur, (dx, dy)| {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                return 0.0;
            }

            let n1 = mag[[
                (y as i32 - dy) as usize,
                (x as i32 - dx) as usize,
            ]];

            let n2 = mag[[
                (y as i32 + dy) as usize,
                (x as i32 + dx) as usize,
            ]];

            if cur >= n1 && cur >= n2 {
                cur
            } else {
                0.0
            }
        })
}

pub fn hysteresis(edges: &Lightness) -> Mask {
    let max = edges.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let cfg = cfg();
    let high = cfg.hyst_high * max;
    let low = cfg.hyst_low * max;

    let strong = edges.mapv(|v| v > high);
    let weak = edges.mapv(|v| v > low && v <= high);

    let mut res = strong.clone();
    let mut dq = VecDeque::new();

    let (h, w) = edges.dim();

    for y in 0..h {
        for x in 0..w {
            if strong[[y, x]] {
                dq.push_back((y, x));
            }
        }
    }

    while let Some((y, x)) = dq.pop_front() {
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                if weak[[ny, nx]] && !res[[ny, nx]] {
                    res[[ny, nx]] = true;
                    dq.push_back((ny, nx));
                }
            }
        }
    }

    res
}
