use crate::types::{Corners, Dim, Mask, Point2D, Quads};
use crate::uf::UnionFind;

pub fn find_shapes(dim: Dim, edges: &Mask) -> Quads {
    let mut labels = vec![0; edges.len()];
    let mut corners = vec![extrema()];

    let mut next_label = 0;

    let mut uf = UnionFind::new();

    let w = dim.w;
    let h = dim.h;

    for y in 2..h - 2 {
        let r = y * w;

        for x in 2..w - 2 {
            let i = x + r;

            if edges[i] == 0 {
                continue;
            }

            let mut id = None;

            for yy in (y - 2)..=y {
                let rr = yy * w;
                let xx_end = if yy == y { x } else { x + 2 };

                for xx in (x - 2)..=xx_end {
                    let ii = xx + rr;

                    let nid = labels[ii];
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
                    corners.push(extrema());
                    next_label
                }
            };

            labels[i] = id;

            let cnrs = &mut corners[id as usize];
            let pt = Point2D(x as f64, y as f64);

            cnrs.0 = cnrs.0.min(pt, &|p|  p.0 + p.1);
            cnrs.1 = cnrs.1.min(pt, &|p| -p.0 + p.1);
            cnrs.2 = cnrs.2.max(pt, &|p| -p.0 + p.1);
            cnrs.3 = cnrs.3.max(pt, &|p|  p.0 + p.1);
        }
    }

    let mut merged = vec![extrema(); (next_label as usize) + 1];

    for id in 1..=next_label {
        let root = uf.find(id);

        let cnrs = &mut merged[root as usize];
        let new = corners[id as usize];

        cnrs.0 = cnrs.0.min(new.0, &|p|  p.0 + p.1);
        cnrs.1 = cnrs.1.min(new.1, &|p| -p.0 + p.1);
        cnrs.2 = cnrs.2.max(new.2, &|p| -p.0 + p.1);
        cnrs.3 = cnrs.3.max(new.3, &|p|  p.0 + p.1);
    }

    merged
        .into_iter()
        .skip(1)
        .filter(|(tl, tr, bl, br)| {
            tl.0.is_finite()
                && tr.0.is_finite()
                && bl.0.is_finite()
                && br.0.is_finite()
                && *tl != *tr
                && *tl != *bl
                && *tl != *br
                && *tr != *bl
                && *tr != *br
                && *bl != *br
        })
        .collect()
}

fn extrema() -> Corners {
    (
        Point2D(f64::INFINITY,     f64::INFINITY),
        Point2D(f64::NEG_INFINITY, f64::INFINITY),
        Point2D(f64::INFINITY,     f64::NEG_INFINITY),
        Point2D(f64::NEG_INFINITY, f64::NEG_INFINITY),
    )
}
