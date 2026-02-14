use crate::config::Config;
use crate::types::{Lightness, Mask};

use std::collections::VecDeque;

use ndarray::{Array2, Zip};

pub fn nms(mag: &Lightness, orient: &Array2<(i8, i8)>, supp: &mut Lightness) {
    let (h, w) = mag.dim();
    let ms = mag.as_slice_memory_order().unwrap();

    Zip::indexed(supp)
        .and(mag)
        .and(orient)
        .for_each(|(y, x), s, &cur, &(dx, dy)| {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                *s = 0.0;
                return;
            }

            let i = x + y * w;
            let d = (dx as i32 + dy as i32 * w as i32) as usize;

            let n1 = ms[i - d];
            let n2 = ms[i + d];

            *s = if cur >= n1 && cur >= n2 {
                cur
            } else {
                0.0
            };
        });
}

pub fn hysteresis(config: &Config, edges: &Lightness, strong: &mut Mask, weak: &mut Mask, out: &mut Mask) {
    let max = edges.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let high = config.hyst_high * max;
    let low = config.hyst_low * max;

    let mut dq = VecDeque::new();

    let (h, w) = edges.dim();

    let es = edges.as_slice_memory_order().unwrap();
    let ss = strong.as_slice_memory_order_mut().unwrap();
    let ws = weak.as_slice_memory_order_mut().unwrap();
    let os = out.as_slice_memory_order_mut().unwrap();

    for y in 0..h {
        let r = y * w;

        for x in 0..w {
            let i = x + r;

            let v = es[i];

            let str = v > high;
            let wk = v > low && v <= high;

            ss[i] = str;
            ws[i] = wk;
            os[i] = str;

            if ss[i] {
                dq.push_back((y, x));
            }
        }
    }

    while let Some((y, x)) = dq.pop_front() {
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            let r = ny * w;

            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let i = nx + r;

                if ws[i] && !os[i] {
                    os[i] = true;
                    dq.push_back((ny, nx));
                }
            }
        }
    }
}
