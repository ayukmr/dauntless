use crate::types::{Corners, Mask, Point, Points, Shapes};
use crate::config::cfg;

use rayon::prelude::*;

use ndarray::Array2;

const PARA_THRESH: f32 = 0.75;

pub fn tags(edges: &Mask, corners: &Mask) -> Shapes {
    let shapes = find_shapes(edges, corners);
    let mut res = filter_quads(shapes);

    if cfg().filter_paras {
        res = filter_paras(res);
    }
    if cfg().filter_enclosed {
        res = filter_enclosed(res);
    }

    res
}

fn find_shapes(edges: &Mask, corners: &Mask) -> Vec<Points> {
    let mut corners = corners.clone();

    let mut labels = Array2::from_elem(edges.dim(), None);
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

                    if let Some(nid) = nid {
                        if let Some(id) = id {
                            uf.unite(id, nid);
                        } else {
                            id = Some(nid);
                        }
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
            labels[[y, x]] = Some(id);

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

struct UnionFind {
    parent: Vec<u32>,
    height: Vec<u32>,
}

impl UnionFind {
    pub fn new() -> Self {
        Self {
            parent: vec![0],
            height: vec![1]
        }
    }

    pub fn push(&mut self, x: u32) {
        self.parent.push(x);
        self.height.push(1);
    }

    pub fn find(&mut self, x: u32) -> u32 {
        let xi = x as usize;
        let p = self.parent[xi];

        if p != x {
            self.parent[xi] = self.find(p);
        }
        self.parent[xi]
    }

    pub fn unite(&mut self, x: u32, y: u32) {
        let xr = self.find(x);
        let yr = self.find(y);

        if xr == yr {
            return;
        }
        let xi = xr as usize;
        let yi = yr as usize;

        if self.height[yi] > self.height[xi] {
            self.parent[xi] = yr;
            self.height[yi] += self.height[xi];
        } else {
            self.parent[yi] = xr;
            self.height[xi] += self.height[yi];
        }
    }
}

fn filter_quads(shapes: Vec<Points>) -> Shapes {
    shapes
        .into_par_iter()
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

fn filter_paras(quads: Shapes) -> Shapes {
    quads
        .into_par_iter()
        .filter_map(|corners| {
            let (tl, tr, bl, br) = corners;
            let t = dist(&tl, &tr);
            let b = dist(&bl, &br);
            let l = dist(&tl, &bl);
            let r = dist(&tr, &br);

            if t == 0.0 || b == 0.0 || l == 0.0 || r == 0.0 {
                return None;
            }

            let hd = (t / b).abs();
            let vd = (l / r).abs();

            if (hd - 1.0).abs() + (vd - 1.0).abs() < PARA_THRESH {
                Some(corners)
            } else {
                None
            }
        })
        .collect()
}

fn dist(a: &Point, b: &Point) -> f32 {
    let dx = a.0 as f32 - b.0 as f32;
    let dy = a.1 as f32 - b.1 as f32;

    (dx * dx + dy * dy).sqrt()
}

fn filter_enclosed(quads: Shapes) -> Shapes {
    let bds: Vec<Bounds> = quads.iter().map(bounds).collect();

    quads
        .into_par_iter()
        .enumerate()
        .filter(|(idx, _)| {
            let bd = &bds[*idx];

            !bds
                .iter()
                .enumerate()
                .any(|(j, o)| {
                    if j == *idx {
                        return false;
                    }
                    o.l < bd.l && o.r > bd.r && o.t < bd.t && o.b > bd.b
                })
        })
        .map(|(_i, pts)| pts)
        .collect()
}

struct Bounds {
    l: u32,
    r: u32,
    t: u32,
    b: u32,
}

fn bounds(corners: &Corners) -> Bounds {
    let (tl, tr, bl, br) = corners;

    let l = u32::min(tl.0, bl.0);
    let r = u32::max(tr.0, br.0);
    let t = u32::min(tl.1, tr.1);
    let b = u32::max(bl.1, br.1);

    Bounds { l, r, t, b }
}
