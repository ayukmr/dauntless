use crate::{oper, post};

use crate::config::Config;
use crate::types::{Dim, Lightness};
use crate::ws::{CannyWorkspace, HarrisWorkspace};

const HARRIS_NEARBY: usize = 3;

pub fn canny(config: &Config, dim: Dim, img: &Lightness, ws: &mut CannyWorkspace) {
    oper::blur(dim, img, &mut ws.bh, &mut ws.blur);
    oper::sobel(dim, &ws.blur, &mut ws.gx, &mut ws.gy);

    for i in 0..dim.len() {
        let gx = ws.gx[i];
        let gy = ws.gy[i];
        ws.mag[i] = (gx * gx + gy * gy).sqrt();
    }

    for i in 0..dim.len() {
        let gx = ws.gx[i];
        let gy = ws.gy[i];

        let ax = gx.abs();
        let ay = gy.abs();

        ws.orient[i] = if ay <= ax * 0.4142 {
            (1, 0)
        } else if ay >= ax * 2.4142 {
            (1, 1)
        } else if gx * gy > 0.0 {
            (0, 1)
        } else {
            (1, -1)
        };
    }

    post::nms(dim, &ws.mag, &ws.orient, &mut ws.supp);
    post::hysteresis(config, dim, &ws.supp, &mut ws.strong, &mut ws.weak, &mut ws.dq, &mut ws.edges);
}

pub fn harris(config: &Config, dim: Dim, img: &Lightness, ws: &mut HarrisWorkspace) {
    oper::sobel(dim, img, &mut ws.gx, &mut ws.gy);

    for i in 0..dim.len() {
        ws.xx[i] = ws.gx[i] * ws.gx[i];
        ws.yy[i] = ws.gy[i] * ws.gy[i];
        ws.xy[i] = ws.gx[i] * ws.gy[i];
    }

    oper::blur(dim, &ws.xx, &mut ws.bh, &mut ws.sxx);
    oper::blur(dim, &ws.yy, &mut ws.bh, &mut ws.syy);
    oper::blur(dim, &ws.xy, &mut ws.bh, &mut ws.sxy);

    for i in 0..dim.len() {
        let a = ws.sxx[i];
        let b = ws.syy[i];
        let c = ws.sxy[i];

        let det = a * b - c * c;
        let trace = a + b;
        ws.resp[i] = det - config.harris_k * trace * trace;
    }

    let rmax = ws.resp.iter_mut().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let thresh = config.harris_thresh * rmax;

    ws.corners.fill(0);

    let w = dim.w;
    let h = dim.h;

    for y in HARRIS_NEARBY..h - HARRIS_NEARBY {
        let r = y * w;

        for x in HARRIS_NEARBY..w - HARRIS_NEARBY {
            let i = x + r;

            let v = ws.resp[i];
            if v <= thresh {
                continue;
            }

            let is_max =
                !(y - HARRIS_NEARBY..=y + HARRIS_NEARBY).any(|yy| {
                    let rr = yy * w;
                    (x - HARRIS_NEARBY..=x + HARRIS_NEARBY).any(|xx| ws.resp[xx + rr] > v)
                });

            if is_max {
                ws.corners[i] = 1;
            }
        }
    }
}
