use crate::types::{Dim, Lightness, Mask};

use std::collections::VecDeque;

#[derive(Default)]
pub struct Detector {
    pub ws: Workspace,
}

impl Detector {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Default)]
pub struct Workspace {
    pub bh: Lightness,
    pub blur: Lightness,
    pub gx: Lightness,
    pub gy: Lightness,
    pub mag: Lightness,
    pub supp: Lightness,
    pub edges: Mask,
    pub dq: VecDeque<(usize, usize)>,
    pub labels: Vec<u32>,
}

impl Workspace {
    pub fn ensure(&mut self, dim: Dim) {
        let l = dim.len();

        ensure_f32(&mut self.bh, l);
        ensure_f32(&mut self.blur, l);
        ensure_f32(&mut self.gx, l);
        ensure_f32(&mut self.gy, l);
        ensure_f32(&mut self.mag, l);
        ensure_f32(&mut self.supp, l);

        ensure_u8(&mut self.edges, l);
        ensure_u32(&mut self.labels, l);
    }
}

fn ensure_f32(a: &mut Vec<f32>, len: usize) {
    if a.len() != len {
        *a = vec![0.0; len];
    }
}

fn ensure_u32(a: &mut Vec<u32>, len: usize) {
    if a.len() != len {
        *a = vec![0; len];
    }
}

fn ensure_u8(a: &mut Vec<u8>, len: usize) {
    if a.len() != len {
        *a = vec![0; len];
    }
}
