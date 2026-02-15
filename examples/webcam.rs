use dauntless::Tag;

use std::time::Instant;

use opencv::{core, videoio, imgproc, highgui};
use opencv::prelude::*;

fn main() -> opencv::Result<()> {
    let mut cam = videoio::VideoCapture::new(2, videoio::CAP_ANY)?;
    highgui::named_window("webcam", highgui::WINDOW_AUTOSIZE)?;

    let mut last = Instant::now();
    let mut fps = 0.0;

    let mut detector = dauntless::Detector::default();

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;

        if frame.empty() {
            continue;
        }

        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64();

        last = now;

        if dt > 0.0 {
            fps = 0.9 * fps + 0.1 * (1.0 / dt);
        }

        show_text(&mut frame, &(fps as i32).to_string(), 20, 20)?;

        let mut light = Mat::default();

        imgproc::cvt_color(
            &frame,
            &mut light,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let w = light.cols();
        let h = light.rows();

        let scale = 400.0 / i32::max(w, h) as f32;

        let sw = (w as f32 * scale) as i32;
        let sh = (h as f32 * scale) as i32;

        let mut resized = Mat::default();

        imgproc::resize(
            &light,
            &mut resized,
            core::Size::new(sw, sh),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let data =
            &resized
                .data_bytes()?
                .to_vec()
                .into_iter()
                .map(|l| l as f32 / 255.0)
                .collect();

        let tags = detector.tags(sw as usize, sh as usize, data);

        for tag in tags {
            let Tag { id, rot, pos, corners: (tl, tr, bl, br) } = tag;

            let corners = [tl, tr, br, bl];

            let xs: Vec<u32> = corners.iter().map(|pt| pt.0).collect();
            let ys: Vec<u32> = corners.iter().map(|pt| pt.1).collect();

            let x = (xs.iter().min().unwrap() + xs.iter().max().unwrap()) / 2;
            let y = (ys.iter().min().unwrap() + ys.iter().max().unwrap()) / 2;

            let label = if let Some(id) = id {
                format!("{}, {}, {:?}", id, rot.to_degrees() as i32, pos)
            } else {
                format!("{}", rot.to_degrees() as i32)
            };

            show_text(
                &mut frame,
                &label,
                (x as f32 / scale) as i32,
                (y as f32 / scale) as i32,
            )?;

            for i in 0..4 {
                let c0 = corners[i];
                let c1 = corners[(i + 1) % 4];

                let p1 = core::Point::new(
                    (c0.0 as f32 / scale) as i32,
                    (c0.1 as f32 / scale) as i32,
                );

                let p2 = core::Point::new(
                    (c1.0 as f32 / scale) as i32,
                    (c1.1 as f32 / scale) as i32,
                );

                imgproc::line(
                    &mut frame,
                    p1,
                    p2,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0
                )?;
            }

            let tlr = (tl.0 as f32, tl.1 as f32);
            let trr = (tr.0 as f32, tr.1 as f32);
            let blr = (bl.0 as f32, bl.1 as f32);
            let brr = (br.0 as f32, br.1 as f32);

            let sx = w as f32 / sw as f32;
            let sy = h as f32 / sh as f32;

            let tlf = (tlr.0 * sx, tlr.1 * sy);
            let trf = (trr.0 * sx, trr.1 * sy);
            let blf = (blr.0 * sx, blr.1 * sy);
            let brf = (brr.0 * sx, brr.1 * sy);

            let hm = Homography::from_corners((tlf, trf, blf, brf));

            for gy in 0..6 {
                for gx in 0..6 {
                    let u = (gx as f32 + 1.5) / 8.0;
                    let v = (gy as f32 + 1.5) / 8.0;

                    let (x, y) = hm.map(u, v);

                    imgproc::circle(
                        &mut frame,
                        core::Point::new(x.round() as i32, y.round() as i32),
                        3,
                        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        -1,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
            }
        }

        highgui::imshow("webcam", &frame)?;

        if highgui::wait_key(1)? == 27 {
            break;
        }
    }

    Ok(())
}

fn show_text(frame: &mut Mat, label: &str, x: i32, y: i32) -> opencv::Result<()> {
    let size = imgproc::get_text_size(
        label,
        imgproc::FONT_HERSHEY_DUPLEX,
        0.65,
        2,
        &mut 0,
    )?;

    imgproc::put_text(
        frame,
        &label,
        core::Point::new(
            x - size.width / 2,
            y + size.height / 2,
        ),
        imgproc::FONT_HERSHEY_DUPLEX,
        0.65,
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )
}

struct Homography {
    mat: [f32; 9],
}

impl Homography {
    fn from_corners(corners: ((f32, f32), (f32, f32), (f32, f32), (f32, f32))) -> Homography {
        let ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) = corners;

        let dx1 = x1 - x3;
        let dx2 = x2 - x3;
        let dx3 = x0 - x1 + x3 - x2;

        let dy1 = y1 - y3;
        let dy2 = y2 - y3;
        let dy3 = y0 - y1 + y3 - y2;

        let denom = dx1 * dy2 - dx2 * dy1;

        let g = (dx3 * dy2 - dx2 * dy3) / denom;
        let h = (dx1 * dy3 - dx3 * dy1) / denom;

        Homography {
            mat: [
                x1 - x0 + g * x1, x2 - x0 + h * x2, x0,
                y1 - y0 + g * y1, y2 - y0 + h * y2, y0,
                g, h, 1.0,
            ],
        }
    }

    fn map(&self, u: f32, v: f32) -> (f32, f32) {
        let m = &self.mat;

        let x = m[0] * u + m[1] * v + m[2];
        let y = m[3] * u + m[4] * v + m[5];
        let w = m[6] * u + m[7] * v + m[8];

        (x / w, y / w)
    }
}
