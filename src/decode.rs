use crate::types::{Lightness, Corners, FCorners};

use ndarray::{s, Array2};

const BIT_THRESH: f32 = 0.5;

const CODES: [u64; 11] = [
    57401312644,
    58383764297,
    59366215950,
    61331119256,
    63296022562,
    65260925868,
    1453707397,
    4401062356,
    9313320621,
    10295772274,
    14225578886,
];

pub fn decode(img: &Lightness, corners: Corners) -> Option<u32> {
    let tag = sample(img, corners)?;

    let min = tag.fold(f32::INFINITY, |a, &b| a.min(b));
    let max = tag.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut bits = (&(tag - min) / (max - min)).mapv(|l| l > BIT_THRESH);

    for _ in 0..4 {
        let bin =
            bits
                .iter()
                .fold(0, |n, &t| (n << 1) | if t { 1 } else { 0 });

        let id = CODES.iter().position(|&i| i == bin);

        if let Some(id) = id {
            return Some(id as u32);
        }

        bits = rot90(bits);
    }

    None
}

fn rot90<T: Clone>(a: Array2<T>) -> Array2<T> {
    a.slice(s![..;-1, ..]).reversed_axes().to_owned()
}

fn sample(img: &Lightness, corners: Corners) -> Option<Array2<f32>> {
    let hm = Homography::from_corners((
        (corners.0.0 as f32, corners.0.1 as f32),
        (corners.1.0 as f32, corners.1.1 as f32),
        (corners.2.0 as f32, corners.2.1 as f32),
        (corners.3.0 as f32, corners.3.1 as f32),
    ));

    let mut out = vec![0.0; 36];

    for y in 0..6 {
        for x in 0..6 {
            let u = (x as f32 + 1.5) / 8.0;
            let v = (y as f32 + 1.5) / 8.0;

            let (ix, iy) = hm.map(u, v);

            let px = ix.floor() as isize;
            let py = iy.floor() as isize;

            let val = img.get((py as usize, px as usize))?;
            out[x + y * 6] = *val;
        }
    }

    Some(Array2::from_shape_vec((6, 6), out).unwrap())
}

struct Homography {
    mat: [f32; 9],
}

impl Homography {
    fn from_corners(corners: FCorners) -> Homography {
        let ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) = corners;

        let dx1 = x1 - x3;
        let dx2 = x2 - x3;
        let dx3 = x0 - x1 + x3 - x2;

        let dy1 = y1 - y3;
        let dy2 = y2 - y3;
        let dy3 = y0 - y1 + y3 - y2;

        let denom = dx1 * dy2 - dx2 * dy1;

        let g = (dx3 * dy2 - dx2 * dy3) / denom;
        let h = (dx1 * dy3 - dx3 * dy1) / denom;

        Homography {
            mat: [
                x1 - x0 + g * x1, x2 - x0 + h * x2, x0,
                y1 - y0 + g * y1, y2 - y0 + h * y2, y0,
                g, h, 1.0,
            ],
        }
    }

    fn map(&self, u: f32, v: f32) -> (f32, f32) {
        let m = &self.mat;

        let x = m[0] * u + m[1] * v + m[2];
        let y = m[3] * u + m[4] * v + m[5];
        let w = m[6] * u + m[7] * v + m[8];

        (x / w, y / w)
    }
}
