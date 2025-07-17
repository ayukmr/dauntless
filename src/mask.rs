use crate::types::{Lightness, Mask};
use crate::{oper, post};

use ndarray::{s, Array2, Zip};

const HARRIS_K: f32 = 0.01;
const HARRIS_THRESH: f32 = 0.2;
const HARRIS_NEARBY: usize = 3;

pub fn canny(img: &Lightness) -> Mask {
    let blur = oper::blur(img);
    let (x, y) = oper::sobel(&blur);

    let mag = (x.powi(2) + y.powi(2)).mapv(f32::sqrt);

    let orient =
        Zip::from(&y)
            .and(&x)
            .map_collect(|&y, &x| f32::atan2(y, x));

    let supp = post::nms(&mag, &orient);
    let mask = post::hysteresis(&supp);

    mask
}

pub fn harris(img: &Lightness) -> Mask {
    let (x, y) = oper::sobel(img);

    let xx = x.powi(2);
    let yy = y.powi(2);
    let xy = x * y;

    let sxx = oper::blur(&xx);
    let syy = oper::blur(&yy);
    let sxy = oper::blur(&xy);

    let det = &sxx * &syy - sxy.powi(2);
    let trace = sxx + syy;
    let resp = det - HARRIS_K * trace.powi(2);

    let points =
        resp
            .indexed_iter()
            .filter_map(|(i, &v)| (v > HARRIS_THRESH).then_some(i));

    let mut mask = Array2::from_elem(resp.dim(), false);
    let (h, w) = resp.dim();

    for (y, x) in points {
        let x0 = x.saturating_sub(HARRIS_NEARBY);
        let x1 = usize::min(x + HARRIS_NEARBY, w - 1);

        let y0 = y.saturating_sub(HARRIS_NEARBY);
        let y1 = usize::min(y + HARRIS_NEARBY, h - 1);

        let patch = resp.slice(s![y0..=y1, x0..=x1]);
        let max = patch.iter().cloned().reduce(f32::max).unwrap();

        if resp[[y, x]] == max {
            mask[[y, x]] = true;
        }
    }

    mask
}
