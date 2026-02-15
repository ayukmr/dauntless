use crate::types::{Dim, Lightness};

pub fn blur(dim: Dim, img: &Lightness, bh: &mut Lightness, out: &mut Lightness) {
    let w = dim.w;
    let h = dim.h;

    for y in 0..h {
        let r = y * w;

        for x in 2..w - 2 {
            let i = x + r;

            let p = (
                img[i - 2],
                img[i - 1],
                img[i],
                img[i + 1],
                img[i + 2],
            );

            bh[i] = (p.0 + 4.0 * p.1 + 6.0 * p.2 + 4.0 * p.3 + p.4) / 16.0;
        }
    }

    for y in 2..h - 2 {
        let r = y * w;

        for x in 0..w {
            let i = x + r;

            let p = (
                bh[i - 2 * w],
                bh[i - w],
                bh[i],
                bh[i + w],
                bh[i + 2 * w],
            );

            out[i] = (p.0 + 4.0 * p.1 + 6.0 * p.2 + 4.0 * p.3 + p.4) / 16.0;
        }
    }
}

pub fn sobel(dim: Dim, img: &Lightness, gx: &mut Lightness, gy: &mut Lightness) {
    let w = dim.w;
    let h = dim.h;

    for y in 1..h - 1 {
        let r = y * w;

        for x in 1..w - 1 {
            let i = x + r;

            let p = (
                (img[i - 1 - w], img[i - w], img[i + 1 - w]),
                (img[i - 1],     0.0,       img[i + 1]),
                (img[i - 1 + w], img[i + w], img[i + 1 + w]),
            );

            gx[i] = (p.0.2 + 2.0 * p.1.2 + p.2.2) - (p.0.0 + 2.0 * p.1.0 + p.2.0);
            gy[i] = (p.2.0 + 2.0 * p.2.1 + p.2.2) - (p.0.0 + 2.0 * p.0.1 + p.0.2);
        }
    }
}
