use crate::{mask, tags, decode};
use crate::types::{Corners, Lightness};

pub fn process(data: Lightness) -> Vec<(Option<u8>, Option<i8>, Corners)> {
    let edges = mask::canny(&data);
    let corners = mask::harris(&data);

    let tags = tags::tags(&edges, &corners);

    tags.into_iter().map(|pts| {
        let tl = *pts.iter().min_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();
        let tr = *pts.iter().min_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
        let bl = *pts.iter().max_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
        let br = *pts.iter().max_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();

        let corners = (tl, tr, bl, br);

        let id = decode::decode(&data, corners);

        let y0 = bl.1 - tl.1;
        let y1 = br.1 - tr.1;

        let deg =
            if y0 != 0 && y1 != 0 {
                let ratio = (y0 as f32 / y1 as f32) - 1.0;
                let deg = ratio / (f32::sqrt(2.0) - 1.0) * 90.0;

                Some(deg as i8)
            } else {
                None
            };

        (id, deg, corners)
    }).collect()
}
