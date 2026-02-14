use crate::{oper, post};

use crate::config::Config;
use crate::types::Lightness;
use crate::ws::{CannyWorkspace, HarrisWorkspace};

use ndarray::Zip;

const HARRIS_NEARBY: usize = 3;

pub fn canny(config: &Config, img: &Lightness, ws: &mut CannyWorkspace) {
    oper::blur(img, &mut ws.bh, &mut ws.blur);
    oper::sobel(&ws.blur, &mut ws.gx, &mut ws.gy);

    Zip::from(&mut ws.mag)
        .and(&ws.gx)
        .and(&ws.gy)
        .for_each(|m, &gx, &gy| *m = (gx * gx + gy * gy).sqrt());

    Zip::from(&mut ws.orient)
        .and(&ws.gx)
        .and(&ws.gy)
        .for_each(|o, &gx, &gy| {
            let ax = gx.abs();
            let ay = gy.abs();

            *o = if ay <= ax * 0.4142 {
                (1, 0)
            } else if ay >= ax * 2.4142 {
                (1, 1)
            } else if gx * gy > 0.0 {
                (0, 1)
            } else {
                (1, -1)
            };
        });

    post::nms(&ws.mag, &ws.orient, &mut ws.supp);
    post::hysteresis(&config, &ws.supp, &mut ws.strong, &mut ws.weak, &mut ws.edges);
}

pub fn harris(config: &Config, img: &Lightness, ws: &mut HarrisWorkspace) {
    oper::sobel(img, &mut ws.gx, &mut ws.gy);

    rayon::scope(|s| {
        s.spawn(|_| {
            Zip::from(&mut ws.xx)
                .and(&ws.gx)
                .for_each(|xx, &gx| *xx = gx * gx);

            oper::blur(&ws.xx, &mut ws.bh1, &mut ws.sxx);
        });
        s.spawn(|_| {
            Zip::from(&mut ws.yy)
                .and(&ws.gy)
                .for_each(|yy, &gy| *yy = gy * gy);

            oper::blur(&ws.yy, &mut ws.bh2, &mut ws.syy);
        });
        s.spawn(|_| {
            Zip::from(&mut ws.xy)
                .and(&ws.gx)
                .and(&ws.gy)
                .for_each(|xy, &gx, &gy| *xy = gx * gy);

            oper::blur(&ws.xy, &mut ws.bh3, &mut ws.sxy);
        });
    });

    Zip::from(&mut ws.resp)
        .and(&ws.sxx)
        .and(&ws.syy)
        .and(&ws.sxy)
        .for_each(|r, &a, &b, &c| {
            let det = a * b - c * c;
            let trace = a + b;
            *r = det - config.harris_k * trace * trace;
        });

    let rmax = &ws.resp.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let thresh = config.harris_thresh * rmax;

    let (h, w) = ws.resp.dim();

    let rs = ws.resp.as_slice_memory_order().unwrap();
    let cs = ws.corners.as_slice_memory_order_mut().unwrap();

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
                cs[i] = true;
            }
        }
    }
}
