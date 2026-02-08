use crate::{oper, post};
use crate::types::{Lightness, Mask};
use crate::config::cfg;

use ndarray::{s, Array2, Zip};

const HARRIS_NEARBY: usize = 3;

pub fn canny(img: &Lightness) -> Mask {
    let blur = oper::blur(img);
    let (x, y) = oper::sobel(&blur);

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

    let supp = post::nms(&mag, &orient);
    post::hysteresis(&supp)
}

pub fn harris(img: &Lightness) -> Mask {
    let (x, y) = oper::sobel(img);

    let xx = x.powi(2);
    let yy = y.powi(2);
    let xy = x * y;

    let sxx = oper::blur(&xx);
    let syy = oper::blur(&yy);
    let sxy = oper::blur(&xy);

    let cfg = cfg();

    let resp =
        Zip::from(&sxx)
            .and(&syy)
            .and(&sxy)
            .map_collect(|a, b, c| {
                let det = a * b - c * c;
                let trace = a + b;
                det - cfg.harris_k * trace * trace
            });

    let rmax = resp.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let thresh = cfg.harris_thresh * rmax;

    let points =
        resp
            .indexed_iter()
            .filter_map(|(i, &v)| (v > thresh).then_some(i));

    let mut mask = Array2::from_elem(resp.dim(), false);
    let (h, w) = resp.dim();

    for (y, x) in points {
        let x0 = x.saturating_sub(HARRIS_NEARBY);
        let x1 = usize::min(x + HARRIS_NEARBY, w - 1);

        let y0 = y.saturating_sub(HARRIS_NEARBY);
        let y1 = usize::min(y + HARRIS_NEARBY, h - 1);

        let patch = resp.slice(s![y0..=y1, x0..=x1]);
        let max = patch.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if resp[[y, x]] == max {
            mask[[y, x]] = true;
        }
    }

    mask
}
