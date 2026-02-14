use crate::detector::Detector;

use crate::types::{Lightness, Mask};

use std::collections::VecDeque;

use ndarray::{Array2, Zip};

impl Detector {
    pub fn nms(&self, mag: &Lightness, orient: &Array2<(i8, i8)>) -> Lightness {
        let (h, w) = mag.dim();
        let ms = mag.as_slice_memory_order().unwrap();

        Zip::indexed(mag)
            .and(orient)
            .map_collect(|(y, x), &cur, &(dx, dy)| {
                if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                    return 0.0;
                }

                let i = x + y * w;
                let d = (dx as i32 + dy as i32 * w as i32) as usize;

                let n1 = ms[i - d];
                let n2 = ms[i + d];

                if cur >= n1 && cur >= n2 {
                    cur
                } else {
                    0.0
                }
            })
    }

    pub fn hysteresis(&self, edges: &Lightness) -> Mask {
        let max = edges.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let high = self.config.hyst_high * max;
        let low = self.config.hyst_low * max;

        let strong = edges.mapv(|v| v > high);
        let weak = edges.mapv(|v| v > low && v <= high);

        let mut res = strong.clone();
        let mut dq = VecDeque::new();

        let (h, w) = edges.dim();

        let ss = strong.as_slice_memory_order().unwrap();
        let ws = weak.as_slice_memory_order().unwrap();
        let rs = res.as_slice_memory_order_mut().unwrap();

        for y in 0..h {
            let r = y * w;

            for x in 0..w {
                let i = x + r;

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

                    if ws[i] && !rs[i] {
                        rs[i] = true;
                        dq.push_back((ny, nx));
                    }
                }
            }
        }

        res
    }
}
