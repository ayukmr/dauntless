use crate::{shapes, filters};
use crate::types::{Mask, Quads};
use crate::config::cfg;

pub fn candidates(edges: &Mask, corners: &Mask) -> Quads {
    let shapes = shapes::find_shapes(edges, corners);
    let mut res = shapes::filter_quads(shapes);

    let cfg = cfg();

    if cfg.filter_ratios {
        res = filters::filter_ratios(res);
    }
    if cfg.filter_angles {
        res = filters::filter_angles(res);
    }
    if cfg.filter_enclosed {
        res = filters::filter_enclosed(res);
    }

    res
}
