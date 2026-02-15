use std::collections::VecDeque;

use crate::types::{Dim, Lightness, Mask};

#[derive(Default)]
pub struct CannyWorkspace {
    pub bh: Lightness,
    pub blur: Lightness,
    pub gx: Lightness,
    pub gy: Lightness,
    pub mag: Lightness,
    pub orient: Vec<(i8, i8)>,
    pub supp: Lightness,
    pub strong: Mask,
    pub weak: Mask,
    pub edges: Mask,
    pub dq: VecDeque<(usize, usize)>,
}

#[derive(Default)]
pub struct HarrisWorkspace {
    pub gx: Lightness,
    pub gy: Lightness,
    pub xx: Lightness,
    pub yy: Lightness,
    pub xy: Lightness,
    pub bh: Lightness,
    pub sxx: Lightness,
    pub syy: Lightness,
    pub sxy: Lightness,
    pub resp: Lightness,
    pub corners: Mask,
}

impl CannyWorkspace {
    pub fn ensure(&mut self, dim: Dim) {
        let l = dim.len();

        ensure_f32(&mut self.bh, l);
        ensure_f32(&mut self.blur, l);
        ensure_f32(&mut self.gx, l);
        ensure_f32(&mut self.gy, l);
        ensure_f32(&mut self.mag, l);
        ensure_orient(&mut self.orient, l);
        ensure_f32(&mut self.supp, l);

        ensure_u8(&mut self.strong, l);
        ensure_u8(&mut self.weak, l);
        ensure_u8(&mut self.edges, l);
    }
}

impl HarrisWorkspace {
    pub fn ensure(&mut self, dim: Dim) {
        let l = dim.len();

        ensure_f32(&mut self.gx, l);
        ensure_f32(&mut self.gy, l);
        ensure_f32(&mut self.xx, l);
        ensure_f32(&mut self.yy, l);
        ensure_f32(&mut self.xy, l);
        ensure_f32(&mut self.bh, l);
        ensure_f32(&mut self.sxx, l);
        ensure_f32(&mut self.syy, l);
        ensure_f32(&mut self.sxy, l);
        ensure_f32(&mut self.resp, l);

        ensure_u8(&mut self.corners, l);
    }
}

fn ensure_f32(a: &mut Vec<f32>, len: usize) {
    if a.len() != len {
        *a = vec![0.0; len];
    }
}

fn ensure_u8(a: &mut Vec<u8>, len: usize) {
    if a.len() != len {
        *a = vec![0; len];
    }
}

fn ensure_orient(a: &mut Vec<(i8, i8)>, len: usize) {
    if a.len() != len {
        *a = vec![(0, 0); len];
    }
}
