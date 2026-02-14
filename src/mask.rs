use crate::detector::Detector;

use crate::types::{Lightness, Mask};

use ndarray::{Array2, Zip};

const HARRIS_NEARBY: usize = 3;

impl Detector {
    pub fn canny(&self, img: &Lightness) -> Mask {
        let blur = self.blur(img);
        let (x, y) = self.sobel(&blur);

        let mag =
            Zip::from(&x)
                .and(&y)
                .map_collect(|&gx, &gy| (gx * gx + gy * gy).sqrt());

        let orient =
            Zip::from(&x)
                .and(&y)
                .map_collect(|&gx, &gy| {
                    let ax = gx.abs();
                    let ay = gy.abs();

                    if ay <= ax * 0.4142 {
                        (1, 0)
                    } else if ay >= ax * 2.4142 {
                        (1, 1)
                    } else if gx * gy > 0.0 {
                        (0, 1)
                    } else {
                        (1, -1)
                    }
                });

        let supp = self.nms(&mag, &orient);
        self.hysteresis(&supp)
    }

    pub fn harris(&self, img: &Lightness) -> Mask {
        let (x, y) = self.sobel(img);

        let xx = x.powi(2);
        let yy = y.powi(2);
        let xy = x * y;

        let sxx = self.blur(&xx);
        let syy = self.blur(&yy);
        let sxy = self.blur(&xy);

        let resp =
            Zip::from(&sxx)
                .and(&syy)
                .and(&sxy)
                .map_collect(|a, b, c| {
                    let det = a * b - c * c;
                    let trace = a + b;
                    det - self.config.harris_k * trace * trace
                });

        let rmax = resp.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let thresh = self.config.harris_thresh * rmax;

        let mut mask = Array2::from_elem(resp.dim(), false);
        let (h, w) = resp.dim();

        let ms = mask.as_slice_memory_order_mut().unwrap();
        let rs = resp.as_slice_memory_order().unwrap();

        for y in HARRIS_NEARBY..h - HARRIS_NEARBY {
            let r = y * w;

            for x in HARRIS_NEARBY..w - HARRIS_NEARBY {
                let i = x + r;

                let v = rs[i];
                if v <= thresh {
                    continue;
                }

                let is_max =
                    !(y - HARRIS_NEARBY..=y + HARRIS_NEARBY).any(|yy| {
                        let rr = yy * w;
                        (x - HARRIS_NEARBY..=x + HARRIS_NEARBY).any(|xx| rs[xx + rr] > v)
                    });

                if is_max {
                    ms[i] = true;
                }
            }
        }

        mask
    }
}
