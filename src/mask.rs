use crate::{oper, post};

use crate::config::Config;
use crate::types::Lightness;
use crate::ws::{CannyWorkspace, HarrisWorkspace};

const HARRIS_NEARBY: usize = 3;

pub fn canny(config: &Config, img: &Lightness, ws: &mut CannyWorkspace) {
    oper::blur(img, &mut ws.bh, &mut ws.blur);
    oper::sobel(&ws.blur, &mut ws.gx, &mut ws.gy);

    let ml = ws.mag.len();
    let ol = ws.orient.len();

    let gxs = ws.gx.as_slice_memory_order().unwrap();
    let gys = ws.gy.as_slice_memory_order().unwrap();
    let ms = ws.mag.as_slice_memory_order_mut().unwrap();
    let os = ws.orient.as_slice_memory_order_mut().unwrap();

    for i in 0..ml {
        let gx = gxs[i];
        let gy = gys[i];
        ms[i] = (gx * gx + gy * gy).sqrt();
    }

    for i in 0..ol {
        let gx = gxs[i];
        let gy = gys[i];

        let ax = gx.abs();
        let ay = gy.abs();

        os[i] = if ay <= ax * 0.4142 {
            (1, 0)
        } else if ay >= ax * 2.4142 {
            (1, 1)
        } else if gx * gy > 0.0 {
            (0, 1)
        } else {
            (1, -1)
        };
    }

    post::nms(&ws.mag, &ws.orient, &mut ws.supp);
    post::hysteresis(&config, &ws.supp, &mut ws.strong, &mut ws.weak, &mut ws.edges);
}

pub fn harris(config: &Config, img: &Lightness, ws: &mut HarrisWorkspace) {
    oper::sobel(img, &mut ws.gx, &mut ws.gy);

    let gxs = ws.gx.as_slice_memory_order().unwrap();
    let gys = ws.gy.as_slice_memory_order().unwrap();

    let xxs = ws.xx.as_slice_memory_order_mut().unwrap();
    let yys = ws.yy.as_slice_memory_order_mut().unwrap();
    let xys = ws.xy.as_slice_memory_order_mut().unwrap();

    for i in 0..img.len() {
        xxs[i] = gxs[i] * gxs[i];
        yys[i] = gys[i] * gys[i];
        xys[i] = gxs[i] * gys[i];
    }

    oper::blur(&ws.xx, &mut ws.bh, &mut ws.sxx);
    oper::blur(&ws.yy, &mut ws.bh, &mut ws.syy);
    oper::blur(&ws.xy, &mut ws.bh, &mut ws.sxy);

    let sxxs = ws.sxx.as_slice_memory_order_mut().unwrap();
    let syys = ws.syy.as_slice_memory_order_mut().unwrap();
    let sxys = ws.sxy.as_slice_memory_order_mut().unwrap();
    let rs = ws.resp.as_slice_memory_order_mut().unwrap();

    for i in 0..img.len() {
        let a = sxxs[i];
        let b = syys[i];
        let c = sxys[i];

        let det = a * b - c * c;
        let trace = a + b;
        rs[i] = det - config.harris_k * trace * trace;
    }

    let rmax = &ws.resp.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let thresh = config.harris_thresh * rmax;

    let (h, w) = ws.resp.dim();

    ws.corners.fill(false);

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
