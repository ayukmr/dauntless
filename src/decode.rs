use crate::hm::Homography;
use crate::types::{Bits, Corners, Dim, Lightness, Point2D};

use std::cmp::Ordering;

const BIT_THRESH: f32 = 0.5;
const ERR_THRESH: u32 = 2;
const N_MEANS: usize = 5;

const CODES: [u64; 33] = [
    57401312644, 58383764297, 59366215950, 61331119256,
    63296022562, 65260925868,  1453707397,  4401062356,
     9313320621, 10295772274, 14225578886, 17172933845,
    18155385498, 19137837151, 21102740457, 22085192110,
    24050095416, 27979902028, 28962353681, 33874611946,
    34857063599, 35839515252, 37804418558, 42716676823,
    43699128476, 46646483435, 47628935088, 49593838394,
    56470999965, 57453451618, 61383258230, 12312814554,
    18207524472,
];

pub fn decode(dim: Dim, img: &Lightness, corners: Corners) -> Option<u32> {
    let tag = sample(dim, img, corners)?;

    let mut vals = tag.clone().to_vec();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let min = vals[..N_MEANS].iter().sum::<f32>() / N_MEANS as f32;
    let max = vals[vals.len() - N_MEANS..].iter().sum::<f32>() / N_MEANS as f32;

    let mut bits =
        tag
            .into_iter()
            .map(|x| (x - min) / (max - min) > BIT_THRESH)
            .collect::<Bits>();

    let mut best: Option<(usize, u32)> = None;

    for _ in 0..4 {
        let bin: u64 =
            bits
                .iter()
                .fold(0, |n, &t| (n << 1) | if t { 1 } else { 0 });

        for (i, code) in CODES.iter().enumerate() {
            let dist = (bin ^ code).count_ones();

            if dist == 0 {
                return Some(i as u32);
            }

            if dist <= ERR_THRESH && best.is_none_or(|(_, best_dist)| dist < best_dist) {
                best = Some((i, dist));
            }
        }

        bits = rot90(bits, 6);
    }

    best.map(|(i, _)| i as u32)
}

fn rot90(a: Bits, n: usize) -> Bits {
    let mut out = vec![false; n * n];

    for y in 0..n {
        for x in 0..n {
            out[x * n + (n - 1 - y)] = a[y * n + x];
        }
    }

    out
}

fn sample(dim: Dim, img: &Lightness, corners: Corners) -> Option<Lightness> {
    let hm = Homography::from_corners((
        Point2D(corners.0.0, corners.0.1),
        Point2D(corners.1.0, corners.1.1),
        Point2D(corners.2.0, corners.2.1),
        Point2D(corners.3.0, corners.3.1),
    ));

    let mut out = vec![0.0; 36];

    let w = dim.w;

    for y in 0..6 {
        for x in 0..6 {
            let u = (x as f64 + 1.5) / 8.0;
            let v = (y as f64 + 1.5) / 8.0;

            let Point2D(ix, iy) = hm.map(u, v);

            let i = ix.floor() as usize + iy.floor() as usize * w;

            let dx = corners.0.0 - corners.1.0;
            let dy = corners.0.1 - corners.1.1;
            let neighbors = dx * dx + dy * dy >= 256.0;

            let val =
                if neighbors {
                      img[i - 1 - w] + img[i - w] + img[i + 1 - w]
                    + img[i - 1]     + img[i]     + img[i + 1]
                    + img[i - 1 + w] + img[i + w] + img[i + 1 + w]
                } else {
                    img[i]
                };

            out[x + y * 6] = val / 9_f32;
        }
    }

    Some(out)
}
