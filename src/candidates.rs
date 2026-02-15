use crate::{filters, shapes};

use crate::config::Config;
use crate::types::{Dim, Mask, Quads};

pub fn candidates(config: &Config, dim: Dim, edges: &Mask, corners: &Mask) -> Quads {
    let shapes = shapes::find_shapes(dim, edges, corners);
    let mut res = shapes::filter_quads(shapes);

    if config.filter_ratios {
        res = filters::filter_ratios(res);
    }
    if config.filter_angles {
        res = filters::filter_angles(res);
    }
    if config.filter_enclosed {
        res = filters::filter_enclosed(res);
    }

    res
}
