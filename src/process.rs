use crate::{candidates, decode, mask};
use crate::types::{Corners, Lightness, Mask, Point2D, Point3D, Tag};
use crate::config::cfg;

const TAG_M: f32 = 0.2;

pub fn process(data: &Lightness) -> (Mask, Vec<Tag>) {
    let (edges, corners) = rayon::join(
        || mask::canny(data),
        || mask::harris(data),
    );

    let candidates = candidates::candidates(&edges, &corners);

    let (img_h, img_w) = data.dim();
    let half_fov_tan = (cfg().fov_rad / 2.0).tan();

    let tags =
        candidates
            .into_iter()
            .map(|corners| {
                let id = decode::decode(data, corners);

                let rot = rotation(corners);
                let pos = position(corners, img_w as f32, img_h as f32, half_fov_tan);

                Tag { id, rot, pos, corners }
            })
            .collect();

    (edges, tags)
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

fn position((tl, tr, bl, br): Corners, img_w: f32, img_h: f32, half_fov_tan: f32) -> Point3D {
    let y0 = (bl.1 - tl.1) as f32;
    let y1 = (br.1 - tr.1) as f32;

    let cnrs_3d = [
        to_3d((tl.0 as f32, tl.1 as f32), y0, img_w, img_h, half_fov_tan),
        to_3d((tr.0 as f32, tr.1 as f32), y1, img_w, img_h, half_fov_tan),
        to_3d((bl.0 as f32, bl.1 as f32), y0, img_w, img_h, half_fov_tan),
        to_3d((br.0 as f32, br.1 as f32), y1, img_w, img_h, half_fov_tan),
    ];

    let x = cnrs_3d.iter().map(|pt| pt.0).sum::<f32>() / 4.0;
    let y = cnrs_3d.iter().map(|pt| pt.1).sum::<f32>() / 4.0;
    let z = cnrs_3d.iter().map(|pt| pt.2).sum::<f32>() / 4.0;

    (x, y, z)
}

fn to_3d(point: Point2D, vis: f32, img_w: f32, img_h: f32, half_fov_tan: f32) -> Point3D {
    let scale = img_h / (2.0 * vis) * TAG_M;
    let aspect = img_w / img_h;

    let x = point.0 / img_w * 2.0 - 1.0;
    let y = point.1 / img_h * 2.0 - 1.0;

    (
        x * scale * aspect,
        -y * scale,
        scale / half_fov_tan,
    )
}
