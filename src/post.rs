use crate::types::{Lightness, Mask};
use crate::config::cfg;

use ndarray::{s, Array2, Zip};

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

    Array2::from_shape_fn(edges.dim(), |(y, x)| {
        let str = strong[[y, x]];
        let wk = weak[[y, x]];

        if str || !wk {
            return str;
        }

        let patch = strong.slice(s![y - 1..=y + 1, x - 1..=x + 1]);
        patch.iter().any(|&e| e)
    })
}
