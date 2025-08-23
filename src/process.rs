use crate::{mask, tags, decode};
use crate::types::{Corners, Lightness};

use rayon::prelude::*;

pub fn process(data: Lightness) -> Vec<(Option<u8>, Option<i8>, Corners)> {
    let edges = mask::canny(&data);
    let corners = mask::harris(&data);

    let tags = tags::tags(&edges, &corners);

    tags.into_par_iter().map(|pts| {
        let tl = *pts.iter().min_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();
        let tr = *pts.iter().min_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
        let bl = *pts.iter().max_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
        let br = *pts.iter().max_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();

        let corners = (tl, tr, bl, br);

        let id = decode::decode(&data, corners);

        let vis = tr.0 - tl.0;

        let y0 = br.1 - tr.1;
        let y1 = bl.1 - tl.1;

        let real = (y0 + y1) / 2;

        let sign = if y0 > y1 { -1.0 } else { 1.0 };

        let deg =
            if vis != 0 && real != 0 {
                let deg = (vis as f32 / real as f32).acos() * (180.0 / std::f32::consts::PI);

                Some((deg * sign) as i8)
            } else {
                None
            };

        (id, deg, corners)
    }).collect()
}
