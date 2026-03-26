use crate::{candidates, decode, mask};
use crate::config::Config;
use crate::detector::Detector;
use crate::types::{Corners, Dim, Lightness, Mask, Point2D, Point3D, Tag};

use std::mem;

const TAG_M: f64 = 0.2;

impl Detector {
    pub fn tags(&mut self, w: usize, h: usize, config: &Config, data: &Lightness) -> Vec<Tag> {
        let dim = Dim { w, h };
        self.ws.ensure(dim);

        mask::canny(config, dim, data, &mut self.ws);

        let candidates = candidates::candidates(config, dim, &self.ws.edges);
        let half_fov_tan = (config.fov.to_radians() / 2.0).tan();

        candidates
            .into_iter()
            .map(|corners| {
                let id = decode::decode(dim, data, corners);

                let rot = rotation(corners);
                let pos = position(corners, dim, half_fov_tan);

                Tag { id, rot, pos, corners }
            })
            .collect()
    }

    pub fn process(&mut self, w: usize, h: usize, config: &Config, data: &Lightness) -> (Vec<Tag>, Mask) {
        let tags = self.tags(w, h, config, data);
        (tags, mem::take(&mut self.ws.edges))
    }
}

fn rotation((tl, tr, bl, br): Corners) -> f32 {
    let x0 = (tr.0 - tl.0) as f32;
    let x1 = (br.0 - bl.0) as f32;

    let vis = (x0 + x1) / 2.0;

    let y0 = (bl.1 - tl.1) as f32;
    let y1 = (br.1 - tr.1) as f32;

    let real = (y0 + y1) / 2.0;

    let sign = if y0 > y1 { 1.0 } else { -1.0 };

    if vis != 0.0 && real != 0.0 {
        (vis / real).clamp(-1.0, 1.0).acos() * sign
    } else {
        0.0
    }
}

fn position((tl, tr, bl, br): Corners, dim: Dim, half_fov_tan: f32) -> Point3D {
    let y0 = bl.1 - tl.1;
    let y1 = br.1 - tr.1;

    let cnrs_3d = [
        to_3d(Point2D(tl.0, tl.1), y0, dim, half_fov_tan),
        to_3d(Point2D(tr.0, tr.1), y1, dim, half_fov_tan),
        to_3d(Point2D(bl.0, bl.1), y0, dim, half_fov_tan),
        to_3d(Point2D(br.0, br.1), y1, dim, half_fov_tan),
    ];

    let x = cnrs_3d.iter().map(|pt| pt.0).sum::<f64>() / 4.0;
    let y = cnrs_3d.iter().map(|pt| pt.1).sum::<f64>() / 4.0;
    let z = cnrs_3d.iter().map(|pt| pt.2).sum::<f64>() / 4.0;

    Point3D(x, y, z)
}

fn to_3d(point: Point2D, vis: f64, dim: Dim, half_fov_tan: f32) -> Point3D {
    let img_w = dim.w as f64;
    let img_h = dim.h as f64;

    let scale = img_h / (2.0 * vis) * TAG_M;
    let aspect = img_w / img_h;

    let x = point.0 / img_w * 2.0 - 1.0;
    let y = point.1 / img_h * 2.0 - 1.0;

    Point3D(
        x * scale * aspect,
        -y * scale,
        scale / half_fov_tan as f64,
    )
}
