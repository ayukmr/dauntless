use crate::{decode, fft, mask, tags};
use crate::types::{Tag, Corners, Point2D, Point3D, Filter, Lightness};

use rayon::prelude::*;

const FOV: f32 = 75.0;
const HALF_FOV: f32 = FOV.to_radians() / 2.0;

pub fn process(data: Lightness, filter: Filter) -> Vec<Tag> {
    let freq = fft::to_freq(&data);

    let (edges, corners) = rayon::join(
        || mask::canny(&freq),
        || mask::harris(&freq),
    );

    let tags = tags::tags(&edges, &corners, filter);

    let (img_h, img_w) = data.dim();

    tags.into_par_iter().map(|pts| {
        let tl = *pts.iter().min_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();
        let tr = *pts.iter().min_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
        let bl = *pts.iter().max_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
        let br = *pts.iter().max_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();

        let corners = (tl, tr, bl, br);

        let id = decode::decode(&data, corners);

        let rot = rotation(corners);
        let pos = pos(corners, img_w as f32, img_h as f32);

        Tag { id, rot, pos, corners }
    }).collect()
}

fn rotation(corners: Corners) -> f32 {
    let (tl, tr, bl, br) = corners;

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

fn pos(corners: Corners, img_w: f32, img_h: f32) -> Point3D {
    let (tl, tr, bl, br) = corners;

    let y0 = (bl.1 - tl.1) as f32;
    let y1 = (br.1 - tr.1) as f32;

    let cnrs_3d = [
        to_3d((tl.0 as f32, tl.1 as f32), y0, img_w, img_h),
        to_3d((tr.0 as f32, tr.1 as f32), y1, img_w, img_h),
        to_3d((bl.0 as f32, bl.1 as f32), y0, img_w, img_h),
        to_3d((br.0 as f32, br.1 as f32), y1, img_w, img_h),
    ];

    let x = cnrs_3d.iter().map(|pt| pt.0).sum::<f32>() / 4.0;
    let y = cnrs_3d.iter().map(|pt| pt.1).sum::<f32>() / 4.0;
    let z = cnrs_3d.iter().map(|pt| pt.2).sum::<f32>() / 4.0;

    (x, y, z)
}

fn to_3d(point: Point2D, vis: f32, img_w: f32, img_h: f32) -> Point3D {
    let scale = img_h / (2.0 * vis) * 0.2;
    let aspect = img_w / img_h;

    let x = point.0 / img_w * 2.0 - 1.0;
    let y = point.1 / img_h * 2.0 - 1.0;

    (
        x * scale * aspect,
        -y * scale,
        scale / HALF_FOV.tan(),
    )
}
