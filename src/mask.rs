use crate::types::{Lightness, Mask};
use crate::{oper, post};

use ndarray::{s, Array2, Zip};

pub fn canny(img: &Lightness) -> Mask {
    let blur = oper::blur(img);
    let (x, y) = oper::sobel(&blur);

    let mag = (x.powi(2) + y.powi(2)).mapv(f32::sqrt);

    let orient =
        Zip::from(&y)
            .and(&x)
            .map_collect(|&y, &x| f32::atan2(y, x));

    let supp = post::nms(&mag, &orient);
    let mask = post::hysteresis(&supp, 0.05, 0.3);

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
    let resp = det - 0.01 * trace.powi(2);

    let points =
        resp
            .indexed_iter()
            .filter_map(|(i, &v)| (v > 0.4).then_some(i));

    let mut mask = Array2::from_elem(resp.dim(), false);
    let (h, w) = resp.dim();

    for (y, x) in points {
        let x0 = x.saturating_sub(3);
        let x1 = usize::min(x + 3, w - 1);

        let y0 = y.saturating_sub(3);
        let y1 = usize::min(y + 3, h - 1);

        let patch = resp.slice(s![y0..=y1, x0..=x1]);
        let max = patch.iter().cloned().reduce(f32::max).unwrap();

        if resp[[y, x]] == max {
            mask[[y, x]] = true;
        }
    }

    mask
}
