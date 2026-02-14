use ndarray::Array2;

use crate::types::{Lightness, Mask};

#[derive(Default)]
pub struct CannyWorkspace {
    pub bh: Lightness,
    pub blur: Lightness,
    pub gx: Lightness,
    pub gy: Lightness,
    pub mag: Lightness,
    pub orient: Array2<(i8, i8)>,
    pub supp: Lightness,
    pub strong: Mask,
    pub weak: Mask,
    pub edges: Mask,
}

#[derive(Default)]
pub struct HarrisWorkspace {
    pub gx: Lightness,
    pub gy: Lightness,
    pub xx: Lightness,
    pub yy: Lightness,
    pub xy: Lightness,
    pub bh1: Lightness,
    pub bh2: Lightness,
    pub bh3: Lightness,
    pub sxx: Lightness,
    pub syy: Lightness,
    pub sxy: Lightness,
    pub resp: Lightness,
    pub corners: Mask,
}

impl CannyWorkspace {
    pub fn ensure(&mut self, h: usize, w: usize) {
        let d = (h, w);

        ensure_f32(&mut self.bh, d);
        ensure_f32(&mut self.blur, d);
        ensure_f32(&mut self.gx, d);
        ensure_f32(&mut self.gy, d);
        ensure_f32(&mut self.mag, d);
        ensure_orient(&mut self.orient, d);
        ensure_f32(&mut self.supp, d);

        ensure_bool(&mut self.strong, d);
        ensure_bool(&mut self.weak, d);
        ensure_bool(&mut self.edges, d);
    }
}

impl HarrisWorkspace {
    pub fn ensure(&mut self, h: usize, w: usize) {
        let d = (h, w);

        ensure_f32(&mut self.gx, d);
        ensure_f32(&mut self.gy, d);
        ensure_f32(&mut self.xx, d);
        ensure_f32(&mut self.yy, d);
        ensure_f32(&mut self.xy, d);
        ensure_f32(&mut self.bh1, d);
        ensure_f32(&mut self.bh2, d);
        ensure_f32(&mut self.bh3, d);
        ensure_f32(&mut self.sxx, d);
        ensure_f32(&mut self.syy, d);
        ensure_f32(&mut self.sxy, d);
        ensure_f32(&mut self.resp, d);

        ensure_bool(&mut self.corners, d);
    }
}

fn ensure_f32(a: &mut Array2<f32>, dim: (usize, usize)) {
    if a.dim() != dim {
        *a = Array2::zeros(dim);
    }
}

fn ensure_bool(a: &mut Array2<bool>, dim: (usize, usize)) {
    if a.dim() != dim {
        *a = Array2::from_elem(dim, false);
    }
}

fn ensure_orient(a: &mut Array2<(i8, i8)>, dim: (usize, usize)) {
    if a.dim() != dim {
        *a = Array2::from_elem(dim, (0, 0));
    }
}
