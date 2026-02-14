use crate::detector::Detector;

use crate::types::{Mask, Quads};

impl Detector {
    pub fn candidates(&self, edges: &Mask, corners: &Mask) -> Quads {
        let shapes = self.find_shapes(edges, corners);
        let mut res = self.filter_quads(shapes);

        if self.config.filter_ratios {
            res = self.filter_ratios(res);
        }
        if self.config.filter_angles {
            res = self.filter_angles(res);
        }
        if self.config.filter_enclosed {
            res = self.filter_enclosed(res);
        }

        res
    }
}
