use crate::uf::UnionFind;
use crate::types::{Mask, Quads, Shapes};

use ndarray::Array2;

pub fn find_shapes(edges: &Mask, corners: &Mask) -> Shapes {
    let mut corners = corners.clone();

    let mut labels = Array2::from_elem(edges.dim(), 0);
    let mut label_pts = vec![Vec::new()];

    let mut next_label = 0;

    let (h, w) = edges.dim();

    let mut uf = UnionFind::new();

    for y in 2..h - 2 {
        for x in 2..w - 2 {
            if !edges[[y, x]] {
                continue;
            }

            let mut id = None;

            for yy in (y - 2)..=(y + 2) {
                for xx in (x - 2)..=(x + 2) {
                    let nid = labels[[yy, xx]];

                    if nid == 0 {
                        continue;
                    }

                    if let Some(id) = id {
                        uf.unite(id, nid);
                    } else {
                        id = Some(nid);
                    }
                }
            }

            let id = match id {
                Some(v) => v,
                None => {
                    next_label += 1;
                    uf.push(next_label);
                    label_pts.push(Vec::new());
                    next_label
                }
            };
            labels[[y, x]] = id;

            let mut corner = None;

            for yy in (y - 1)..=(y + 1) {
                for xx in (x - 1)..=(x + 1) {
                    if corners[[yy, xx]] {
                        corner = Some((yy, xx));
                        break;
                    }
                }
                if corner.is_some() {
                    break;
                }
            }

            if let Some((cy, cx)) = corner {
                label_pts[id as usize].push((cx as u32, cy as u32));
                corners[[cy, cx]] = false;
            }
        }
    }

    let mut merged = vec![Vec::new(); (next_label as usize) + 1];

    for id in 1..=next_label {
        let root = uf.find(id);
        let pts = &label_pts[id as usize];
        merged[root as usize].extend(pts);
    }

    merged
        .into_iter()
        .filter(|pts| !pts.is_empty())
        .collect()
}

pub fn filter_quads(shapes: Shapes) -> Quads {
    shapes
        .into_iter()
        .filter_map(|pts| {
            if pts.len() < 4 {
                return None;
            }

            let tl = *pts.iter().min_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();
            let tr = *pts.iter().min_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
            let bl = *pts.iter().max_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
            let br = *pts.iter().max_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();

            if tl == tr || tl == bl || tl == br || tr == bl || tr == br || bl == br {
                return None;
            }

            Some((tl, tr, bl, br))
        })
        .collect()
}
