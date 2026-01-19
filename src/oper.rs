use crate::types::Lightness;

use ndarray::{s, Array2, Zip};

const GAUSS: [f32; 5] = [
    1.0 / 16.0,
    4.0 / 16.0,
    6.0 / 16.0,
    4.0 / 16.0,
    1.0 / 16.0,
];

const SOBEL_X: [[f32; 3]; 3] = [
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0],
];

const SOBEL_Y: [[f32; 3]; 3] = [
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0],
];

fn pad(img: &Lightness, r: usize) -> Lightness {
    let (h, w) = img.dim();

    Array2::from_shape_fn(
        (h + 2 * r, w + 2 * r),
        |(y, x)| {
            let yy = (y - r).clamp(0, h - 1);
            let xx = (x - r).clamp(0, w - 1);
            img[(yy, xx)]
        }
    )
}

pub fn blur(img: &Lightness) -> Lightness {
    let (h, w) = img.dim();
    let pad = pad(img, 2);

    let mut bh: Array2<f32> = Array2::zeros((h + 4, w));
    for (i, &k) in GAUSS.iter().enumerate() {
        let v = pad.slice(s![.., i..i + w]);
        Zip::from(&mut bh)
            .and(&v)
            .for_each(|t, &x| *t += k * x);
    }

    let mut bv: Array2<f32> = Array2::zeros((h, w));
    for (i, &k) in GAUSS.iter().enumerate() {
        let v = bh.slice(s![i..i + h, ..]);
        Zip::from(&mut bv)
            .and(&v)
            .for_each(|o, &x| *o += k * x);
    }

    bv
}

pub fn sobel(img: &Lightness) -> (Lightness, Lightness) {
    let (h, w) = img.dim();
    let pad = pad(img, 1);

    let mut gx = Array2::zeros((h, w));
    let mut gy = Array2::zeros((h, w));

    for dy in 0..3 {
        for dx in 0..3 {
            let vx = pad.slice(s![dy..dy + h, dx..dx + w]);
            let cx = SOBEL_X[dy][dx];
            if cx != 0.0 {
                Zip::from(&mut gx)
                    .and(&vx)
                    .for_each(|o, &v| *o += cx * v);
            }

            let vy = pad.slice(s![dy..dy + h, dx..dx + w]);
            let cy = SOBEL_Y[dy][dx];
            if cy != 0.0 {
                Zip::from(&mut gy)
                    .and(&vy)
                    .for_each(|o, &v| *o += cy * v);
            }
        }
    }

    (gx, gy)
}
