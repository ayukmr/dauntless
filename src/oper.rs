use crate::types::Frequency;
use crate::fft;

use std::cmp;

use ndarray::{s, Array1, Array2, Axis};
use once_cell::sync::OnceCell;

const BLUR_R: f32 = 0.15;

static SX: OnceCell<Frequency> = OnceCell::new();
static SY: OnceCell<Frequency> = OnceCell::new();

pub fn blur(freq: &Frequency) -> Frequency {
    let (h, w) = freq.dim();

    let mw = w as f32 / 2.0;
    let mh = h as f32 / 2.0;

    let xs = Array1::from_iter((0..w).map(|x| (x as f32 - mw).powi(2)));
    let ys = Array1::from_iter((0..h).map(|y| (y as f32 - mh).powi(2)));

    let sigma = BLUR_R * cmp::min(h, w) as f32;

    let dist = &ys.insert_axis(Axis(1)) + &xs.insert_axis(Axis(0));
    let mask = (-dist / (2.0 * sigma.powi(2))).exp();

    freq * &mask
}

pub fn sobel(freq: &Frequency) -> (Frequency, Frequency) {
    let sobel_x = SX.get_or_init(|| {
        create_kernel(
            vec![
                -1., 0., 1.,
                -2., 0., 2.,
                -1., 0., 1.,
            ],
            freq.dim()
        )
    });

    let sobel_y = SY.get_or_init(|| {
        create_kernel(
            vec![
                -1., -2., -1.,
                 0.,  0.,  0.,
                 1.,  2.,  1.,
            ],
            freq.dim()
        )
    });

    let x = freq * sobel_x;
    let y = freq * sobel_y;

    (x, y)
}

fn create_kernel(data: Vec<f32>, dim: (usize, usize)) -> Frequency {
    let kernel = Array2::from_shape_vec((3, 3), data).unwrap();

    let mut padded = Array2::zeros(dim);

    let (kh, kw) = kernel.dim();
    padded.slice_mut(s![..kh, ..kw]).assign(&kernel);

    fft::to_freq(&padded)
}
