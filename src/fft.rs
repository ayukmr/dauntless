use crate::types::{Lightness, Frequency};

use rayon::prelude::*;

use std::sync::Arc;

use ndarray::{Array1, Array2, Axis};
use once_cell::sync::OnceCell;

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

struct Plans {
    row_fwd: Arc<dyn Fft<f32> + Sync + Send>,
    row_inv: Arc<dyn Fft<f32> + Sync + Send>,
    col_fwd: Arc<dyn Fft<f32> + Sync + Send>,
    col_inv: Arc<dyn Fft<f32> + Sync + Send>,
}

static PLANS: OnceCell<Arc<Plans>> = OnceCell::new();

pub fn to_freq(data: &Lightness) -> Frequency {
    let complex = data.mapv(Complex::from);
    let freq = fft2(&complex, false);
    shift(&freq, false)
}

pub fn from_freq(freq: &Frequency) -> Lightness {
    let shifted = shift(freq, true);
    let data = fft2(&shifted, true);
    data.mapv(|n| n.re)
}

fn fft2(data: &Frequency, inverse: bool) -> Frequency {
    let (h, w) = data.dim();

    let plans =
        PLANS
            .get_or_init(|| {
                let mut planner = FftPlanner::<f32>::new();
                Arc::new(Plans {
                    row_fwd: planner.plan_fft_forward(w),
                    row_inv: planner.plan_fft_inverse(w),
                    col_fwd: planner.plan_fft_forward(h),
                    col_inv: planner.plan_fft_inverse(h),
                })
            })
            .clone();

    let f_row = if inverse { &plans.row_inv } else { &plans.row_fwd };
    let f_col = if inverse { &plans.col_inv } else { &plans.col_fwd };

    let mut data = data.clone();

    data
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(w)
        .for_each(|row| {
            f_row.process(row);
        });

    data
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let mut v = col.to_vec();
            f_col.process(&mut v);

            col.assign(&Array1::from(v));
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
