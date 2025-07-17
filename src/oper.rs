use crate::types::Lightness;
use crate::fft;

use std::cmp;

use ndarray::{s, Array2};

const BLUR_R: f32 = 0.15;

pub fn blur(img: &Lightness) -> Lightness {
    let (h, w) = img.dim();

    let x = Array2::from_shape_vec(
        (h, w),
        (0..h).flat_map(|_| (0..w).map(|x| x as f32)).collect(),
    ).unwrap();

    let y = Array2::from_shape_vec(
        (h, w),
        (0..h).flat_map(|y| (0..w).map(move |_| y as f32)).collect(),
    ).unwrap();

    let mw = w as f32 / 2.0;
    let mh = h as f32 / 2.0;

    let dist = ((x - mw).powi(2) + (y - mh).powi(2)).mapv(f32::sqrt);
    let sigma = BLUR_R * cmp::min(h, w) as f32;

    let freq = fft::to_freq(img);
    let mask = (-dist.powi(2) / (2.0 * sigma.powi(2))).exp();

    let res = freq * mask;

    fft::from_freq(&res)
}

pub fn sobel(img: &Lightness) -> (Lightness, Lightness) {
    let sobel_x = Array2::from_shape_vec(
        (3, 3),
        vec![
            -1., 0., 1.,
            -2., 0., 2.,
            -1., 0., 1.,
        ],
    ).unwrap();

    let sobel_y = Array2::from_shape_vec(
        (3, 3),
        vec![
            -1., -2., -1.,
             0.,  0.,  0.,
             1.,  2.,  1.,
        ],
    ).unwrap();

    let x = apply(&img, &sobel_x);
    let y = apply(&img, &sobel_y);

    (x, y)
}

fn apply(img: &Lightness, kernel: &Lightness) -> Lightness {
    let mut padded = Array2::zeros(img.dim());

    let (kh, kw) = kernel.dim();
    padded.slice_mut(s![..kh, ..kw]).assign(&kernel);

    let f_img = fft::to_freq(img);
    let f_knl = fft::to_freq(&padded);

    let res = f_img * f_knl;

    fft::from_freq(&res)
}
