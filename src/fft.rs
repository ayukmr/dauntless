use crate::types::{Lightness, Frequency};

use rayon::prelude::*;

use ndarray::{Array1, Array2, Axis};

use num_complex::Complex;
use rustfft::FftPlanner;

pub fn to_freq(data: &Lightness) -> Frequency {
    let complex = data.mapv(Complex::from);

    let freq = fft2(&complex, false);
    let shifted = shift(&freq, false);

    shifted
}

pub fn from_freq(freq: &Frequency) -> Lightness {
    let shifted = shift(freq, true);
    let data = fft2(&shifted, true);

    data.mapv(|n| n.re)
}

fn fft2(data: &Frequency, inverse: bool) -> Frequency {
    let (h, w) = data.dim();

    let mut planner = FftPlanner::new();

    let f_col =
        if inverse {
            planner.plan_fft_inverse(h)
        } else {
            planner.plan_fft_forward(h)
        };

    let f_row =
        if inverse {
            planner.plan_fft_inverse(w)
        } else {
            planner.plan_fft_forward(w)
        };

    let mut data = data.clone();

    data.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let mut v = col.to_vec();
            f_col.process(&mut v);

            col.assign(&Array1::from(v));
        });

    data.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mut v = row.to_vec();
            f_row.process(&mut v);

            row.assign(&Array1::from(v));
        });

    if inverse {
        let scale = (h * w) as f32;
        data.mapv(|l| l / scale)
    } else {
        data
    }
}

fn shift<T: Copy>(data: &Array2<T>, inverse: bool) -> Array2<T> {
    let (h, w) = data.dim();

    let h_mid = (h + if inverse { 1 } else { 0 }) / 2;
    let w_mid = (w + if inverse { 1 } else { 0 }) / 2;

    Array2::from_shape_fn(data.dim(), |(y, x)| {
        let nx = (x + w_mid) % w;
        let ny = (y + h_mid) % h;

        data[(ny, nx)]
    })
}
