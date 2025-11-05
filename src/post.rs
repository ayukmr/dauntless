use crate::types::{Lightness, Mask};

use ndarray::{s, Array2, Zip};

const HYST_LOW_T: f32 = 0.05;
const HYST_HIGH_T: f32 = 0.3;

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
    let norm = edges / max;

    let strong = norm.mapv(|v| v > HYST_HIGH_T);
    let weak = norm.mapv(|v| v > HYST_LOW_T && v <= HYST_HIGH_T);

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
