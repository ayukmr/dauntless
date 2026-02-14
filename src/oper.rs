use crate::detector::Detector;

use crate::types::Lightness;

use ndarray::Array2;

impl Detector {
    pub fn blur(&self, img: &Lightness) -> Lightness {
        let (h, w) = img.dim();
        let is = img.as_slice_memory_order().unwrap();

        let mut bh = Array2::zeros((h, w));
        let bs = bh.as_slice_memory_order_mut().unwrap();

        for y in 0..h {
            let r = y * w;

            for x in 2..w - 2 {
                let i = x + r;

                let p = (
                    is[i - 2],
                    is[i - 1],
                    is[i],
                    is[i + 1],
                    is[i + 2],
                );

                bs[i] = (p.0 + 4.0 * p.1 + 6.0 * p.2 + 4.0 * p.3 + p.4) / 16.0;
            }
        }

        let mut out = Array2::zeros((h, w));
        let os = out.as_slice_memory_order_mut().unwrap();

        for y in 2..h - 2 {
            let r = y * w;

            for x in 0..w {
                let i = x + r;

                let p = (
                    bs[i - 2 * w],
                    bs[i - w],
                    bs[i],
                    bs[i + w],
                    bs[i + 2 * w],
                );

                os[i] = (p.0 + 4.0 * p.1 + 6.0 * p.2 + 4.0 * p.3 + p.4) / 16.0;
            }
        }

        out
    }

    pub fn sobel(&self, img: &Lightness) -> (Lightness, Lightness) {
        let (h, w) = img.dim();
        let is = img.as_slice_memory_order().unwrap();

        let mut gx = Array2::zeros((h, w));
        let mut gy = Array2::zeros((h, w));

        let xs = gx.as_slice_memory_order_mut().unwrap();
        let ys = gy.as_slice_memory_order_mut().unwrap();

        for y in 1..h - 1 {
            let r = y * w;

            for x in 1..w - 1 {
                let i = x + r;

                let p = (
                    (is[i - 1 - w], is[i - w], is[i + 1 - w]),
                    (is[i - 1],     0.0,       is[i + 1]),
                    (is[i - 1 + w], is[i + w], is[i + 1 + w]),
                );

                xs[i] = (p.0.2 + 2.0 * p.1.2 + p.2.2) - (p.0.0 + 2.0 * p.1.0 + p.2.0);
                ys[i] = (p.2.0 + 2.0 * p.2.1 + p.2.2) - (p.0.0 + 2.0 * p.0.1 + p.0.2);
            }
        }

        (gx, gy)
    }
}
