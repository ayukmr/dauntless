use crate::types::{Lightness, Mask};

use ndarray::{s, Array2, Zip};

const HYST_LOW_T: f32 = 0.05;
const HYST_HIGH_T: f32 = 0.3;

pub fn nms(mag: &Lightness, orient: &Lightness) -> Lightness {
    let (h, w) = mag.dim();

    Zip::indexed(mag)
        .and(orient)
        .map_collect(|(y, x), &cur, &orient| {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                return 0.0;
            }

            let angle = orient * 180.0 / std::f32::consts::PI;

            let (dx, dy) =
                if (-112.5..-67.5).contains(&angle) || (67.5..112.5).contains(&angle) {
                    (0, 1)
                } else if (-67.5..-22.5).contains(&angle) || (112.5..157.5).contains(&angle) {
                    (1, -1)
                } else if (22.5..67.5).contains(&angle) || (-157.5..-112.5).contains(&angle) {
                    (1, 1)
                } else {
                    (1, 0)
                };

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
