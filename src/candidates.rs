use crate::{filters, shapes};
use crate::config::Config;
use crate::types::{Dim, Mask, Quads};

pub fn candidates(config: &Config, dim: Dim, edges: &Mask, labels: &mut Vec<u32>) -> Quads {
    let mut res = shapes::find_shapes(dim, edges, labels);

    if config.filter_ratios {
        res = filters::filter_ratios(res);
    }
    if config.filter_angles {
        res = filters::filter_angles(res);
    }

    res
}
