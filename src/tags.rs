use crate::{shapes, filters};
use crate::types::{Mask, Quads};
use crate::config::cfg;

pub fn tags(edges: &Mask, corners: &Mask) -> Quads {
    let shapes = shapes::find_shapes(edges, corners);
    let mut res = shapes::filter_quads(shapes);

    let cfg = cfg();

    if cfg.filter_paras {
        res = filters::filter_paras(res);
    }
    if cfg.filter_enclosed {
        res = filters::filter_enclosed(res);
    }

    res
}
