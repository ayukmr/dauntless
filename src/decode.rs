use crate::types::{Lightness, Corners};

use ndarray::{s, Array1, Array2, Zip};

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

const TAG_SIZE: usize = 6;

pub fn decode(img: &Lightness, points: Corners) -> Option<u8> {
    let tag = sample(img, points);

    let max = tag.iter().cloned().reduce(f32::max).unwrap();
    let mut bits = (&tag / max).mapv(|l| l > 0.5);

    for _ in 0..4 {
        let id =
            bits
                .iter()
                .fold(0, |n, &t| (n << 1) | if t { 1 } else { 0 });

        let idx = CODES.iter().position(|&i| i == id);

        if let Some(idx) = idx {
            return Some(idx as u8);
        }

        bits = rot90(bits);
    }

    None
}

fn sample(img: &Lightness, corners: Corners) -> Lightness {
    let idx = Array1::from_shape_fn(
        TAG_SIZE,
        |i| (i as f32 + 1.5) / (TAG_SIZE + 2) as f32,
    );

    let u = Array2::from_shape_fn((TAG_SIZE, TAG_SIZE), |(_, x)| idx[x]);
    let v = Array2::from_shape_fn((TAG_SIZE, TAG_SIZE), |(y, _)| idx[y]);

    let inv_u = u.mapv(|x| 1.0 - x);
    let inv_v = v.mapv(|x| 1.0 - x);

    let (tl, tr, bl, br) = corners;

    let ix = (
        &inv_u * &inv_v * tl.0 as f32 +
        &u * &inv_v * tr.0 as f32 +
        &u * &v * br.0 as f32 +
        &inv_u * &v * bl.0 as f32
    ).mapv(|x| x as usize);

    let iy = (
        &inv_u * &inv_v * tl.1 as f32 +
        &u * &inv_v * tr.1 as f32 +
        &u * &v * br.1 as f32 +
        &inv_u * &v * bl.1 as f32
    ).mapv(|x| x as usize);

    Zip::from(&ix)
        .and(&iy)
        .map_collect(|&x, &y| img[[y, x]])
}

fn rot90<T: Clone>(a: Array2<T>) -> Array2<T> {
    a.slice(s![..;-1, ..]).reversed_axes().to_owned()
}
