use crate::types::{Lightness, Corners, FPoint, FCorners};

use ndarray::{s, Array2};

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
    let tag = sample(img, corners);

    let min = tag.fold(f32::INFINITY, |a, &b| a.min(b));
    let max = tag.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut bits = (&(tag - min) / (max - min)).mapv(|l| l > 0.75);

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

fn sample(img: &Lightness, corners: Corners) -> Array2<f32> {
    let (tl, tr, bl, br) = corners;

    let points = partition(
        (
            (tl.0 as f32, tl.1 as f32),
            (tr.0 as f32, tr.1 as f32),
            (bl.0 as f32, bl.1 as f32),
            (br.0 as f32, br.1 as f32),
        ),
        3,
    );

    let bits: Vec<f32> = points.iter().map(|pt| {
        img[(pt.1 as usize, pt.0 as usize)]
    }).collect();

    let dirs = [(0, 0), (1, 0), (0, 1), (1, 1)];

    let mut out = vec![0.0; 64];
    let mut idx = 0;

    for (bx, by) in &dirs {
        for (cx, cy) in &dirs {
            for (x, y) in &dirs {
                let pos = x + cx * 2 + bx * 4 + (y + cy * 2 + by * 4) * 8;
                out[pos] = bits[idx];

                idx += 1;
            }
        }
    }

    Array2::from_shape_vec((8, 8), out).unwrap().slice_move(s![1..7, 1..7])
}

fn partition(corners: FCorners, depth: u32) -> Vec<FPoint> {
    let c = center(corners);

    if depth == 0 {
        return vec![c];
    }

    let (tl, tr, bl, br) = corners;

    let left = (tl.0 + bl.0) / 2.0;
    let right = (tr.0 + br.0) / 2.0;

    let width = right - left;

    let left_h = bl.1 - tl.1;
    let right_h = br.1 - tr.1;

    let center_h = mix(
        (c.0 - left) / width,
        left_h,
        right_h,
    );

    let tm = (c.0, c.1 - center_h / 2.0);
    let bm = (c.0, c.1 + center_h / 2.0);

    let m1 = (tr.1 - tl.1) / (tr.0 - tl.0);
    let m2 = (br.1 - bl.1) / (br.0 - bl.0);

    let avg = (m1 + m2) / 2.0;

    let ml = (left,  c.1 - avg * (c.0 - left));
    let mr = (right, c.1 + avg * (right - c.0));

    vec![
        partition((tl, tm, ml, c), depth - 1),
        partition((tm, tr, c, mr), depth - 1),
        partition((ml, c, bl, bm), depth - 1),
        partition((c, mr, bm, br), depth - 1),
    ].into_iter().flatten().collect()
}

fn center(corners: FCorners) -> FPoint {
    let (tl, tr, bl, br) = corners;

    let m1 = (br.1 - tl.1) / (br.0 - tl.0);
    let m2 = (bl.1 - tr.1) / (bl.0 - tr.0);

    intersect(m1, tl, m2, tr)
}

fn intersect(m1: f32, p1: FPoint, m2: f32, p2: FPoint) -> FPoint {
    let x = (m2 * p2.0 - m1 * p1.0 + p1.1 - p2.1) / (m2 - m1);

    (x, m1 * (x - p1.0) + p1.1)
}

fn mix(x: f32, a: f32, b: f32) -> f32 {
    a + x * (b - a)
}
