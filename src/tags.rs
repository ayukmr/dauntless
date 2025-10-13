use crate::types::{Filter, Mask, Point, Points, Shapes};

use rayon::prelude::*;

use std::collections::{HashMap, HashSet};
use ndarray::{s, Array1, Array2};

const SEG_THRESH: f32 = 1.5;
const PARA_THRESH: f32 = 0.75;

pub fn tags(edges: &Mask, corners: &Mask, filter: Filter) -> Shapes {
    let mut shapes = find_shapes(edges, corners);

    if filter.quads {
        shapes = filter_quads(shapes);
    }
    if filter.paras {
        shapes = filter_paras(shapes);
    }
    if filter.enclosed {
        shapes = filter_enclosed(shapes);
    }

    shapes
}

fn find_shapes(edges: &Mask, corners: &Mask) -> Shapes {
    let mut corners = corners.clone();
    let mut annot = Array2::zeros(edges.dim());

    let mut res: HashMap<u32, Points> = HashMap::new();
    let mut merge = HashSet::new();

    let mut next_id = 1;

    let (h, w) = edges.dim();

    for y in 2..h - 2 {
        for x in 2..w - 2 {
            if !edges[[y, x]] {
                continue;
            }

            let patch = annot.slice(s![y - 2..=y + 2, x - 2..=x + 2]);

            let ids: Vec<u32> =
                patch
                    .iter()
                    .filter(|&&id| id != 0)
                    .cloned()
                    .collect();

            let id =
                if !ids.is_empty() {
                    for i0 in &ids {
                        for i1 in &ids {
                            if i0 != i1 && i0 < i1 {
                                merge.insert((*i0, *i1));
                            }
                        }
                    }

                    ids[0]
                } else {
                    next_id += 1;
                    next_id
                };

            let patch = corners.slice(s![y - 1..=y + 1, x - 1..=x + 1]);

            let corner =
                patch
                    .indexed_iter()
                    .find_map(|(i, &c)| c.then_some(i));

            if let Some((py, px)) = corner {
                let cx = x - 1 + px;
                let cy = y - 1 + py;

                res.entry(id).or_default().push((cx as u32, cy as u32));
                corners[[cy, cx]] = false;
            }

            annot[[y, x]] = id;
        }
    }

    let mut merge: Points = merge.into_iter().collect();
    merge.sort();

    for (i0, i1) in merge {
        let c0 = res.remove(&i0).unwrap_or_default();
        let c1 = res.remove(&i1).unwrap_or_default();

        let mut merged = c0;
        merged.extend(c1);

        res.insert(i1, merged);
    }

    res.into_values().filter(|pts| !pts.is_empty()).collect()
}

fn filter_quads(shapes: Shapes) -> Shapes {
    shapes
        .into_par_iter()
        .filter_map(|pts| {
            let count = pts.len();

            if count >= 4 {
                let tl = *pts.iter().min_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();
                let tr = *pts.iter().min_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
                let bl = *pts.iter().max_by_key(|p| -(p.0 as i32) + p.1 as i32).unwrap();
                let br = *pts.iter().max_by_key(|p|  (p.0 as i32) + p.1 as i32).unwrap();

                let outer = vec![tl, tr, bl, br];

                let mut counted: Points =
                    pts
                        .into_par_iter()
                        .filter(|pt| {
                            !outer.contains(pt) &&
                            !on_segment(pt, &tl, &tr) &&
                            !on_segment(pt, &tr, &br) &&
                            !on_segment(pt, &br, &bl) &&
                            !on_segment(pt, &bl, &tl)
                        })
                        .collect();

                counted.extend(outer);

                Some(counted)
            } else if count == 4 {
                Some(pts)
            } else {
                None
            }
        })
        .collect()
}

fn on_segment(p: &Point, a: &Point, b: &Point) -> bool {
    let p = Array1::from_vec(vec![p.0 as f32, p.1 as f32]);
    let a = Array1::from_vec(vec![a.0 as f32, a.1 as f32]);
    let b = Array1::from_vec(vec![b.0 as f32, b.1 as f32]);

    let ab = &b - &a;
    let ap = &p - &a;

    let cross = ab[0] * ap[1] - ab[1] * ap[0];
    let norm = f32::sqrt(ab[0].powi(2) + ab[1].powi(2));

    if cross > SEG_THRESH * norm {
        return false;
    }

    ap.dot(&ab) <= ab.dot(&ab)
}

fn filter_paras(quads: Shapes) -> Shapes {
    quads.into_par_iter().filter_map(|pts| {
        let mut h = pts.clone();
        h.sort_by_key(|p| (p.0, p.1));

        let mut v = pts.clone();
        v.sort_by_key(|p| (p.1, p.0));

        let hd0 = v[0].0 as f32 - v[1].0 as f32;
        let hd1 = v[2].0 as f32 - v[3].0 as f32;

        let vd0 = h[0].1 as f32 - h[1].1 as f32;
        let vd1 = h[2].1 as f32 - h[3].1 as f32;

        if hd0 == 0.0 || hd1 == 0.0 || vd0 == 0.0 || vd1 == 0.0 {
            return None;
        }

        let h_diff = (hd0 / hd1).abs();
        let v_diff = (vd0 / vd1).abs();

        if (h_diff - 1.0).abs() + (v_diff - 1.0).abs() < PARA_THRESH {
            Some(pts)
        } else {
            None
        }
    }).collect()
}

fn filter_enclosed(quads: Shapes) -> Shapes {
    quads.clone().into_par_iter().filter(|pts| {
        let (x0s, y0s): (Vec<u32>, Vec<u32>) =
            pts.iter().cloned().unzip();

        let x00 = x0s.iter().min();
        let x01 = x0s.iter().max();

        let y00 = y0s.iter().min();
        let y01 = y0s.iter().max();

        !quads.par_iter().any(|others| {
            if others == pts {
                return false;
            }

            let (x1s, y1s): (Vec<u32>, Vec<u32>) =
                others.iter().cloned().unzip();

            let x10 = x1s.iter().min();
            let x11 = x1s.iter().max();

            let y10 = y1s.iter().min();
            let y11 = y1s.iter().max();

            x10 < x00 && x11 > x01 && y10 < y00 && y11 > y01
        })
    }).collect()
}
