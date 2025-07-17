use ndarray::Array2;

use opencv::{core, videoio, imgproc, highgui};
use opencv::prelude::*;

fn main() -> opencv::Result<()> {
    let mut cam = videoio::VideoCapture::new(1, videoio::CAP_ANY)?;
    highgui::named_window("webcam", highgui::WINDOW_AUTOSIZE)?;

    loop {
        let mut frame = core::Mat::default();
        cam.read(&mut frame)?;

        if frame.empty() {
            continue;
        }

        let mut light = Mat::default();

        imgproc::cvt_color(
            &frame,
            &mut light,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let h = light.rows();
        let w = light.cols();

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

        let data = Array2::from_shape_vec(
            (sh as usize, sw as usize),
            resized.data_bytes()?.to_vec(),
        ).unwrap().mapv(|l| l as f32) / 255.0;

        let tags = dauntless::tags(data);

        for (id, deg, (tl, tr, bl, br)) in tags {
            let corners = [tl, tr, br, bl];

            let xs: Vec<u32> = corners.iter().map(|pt| pt.0).collect();
            let ys: Vec<u32> = corners.iter().map(|pt| pt.1).collect();

            let x = (xs.iter().min().unwrap() + xs.iter().max().unwrap()) / 2;
            let y = (ys.iter().min().unwrap() + ys.iter().max().unwrap()) / 2;

            let label =
                if let Some(id) = id {
                    format!("{}@{}*", id, deg.unwrap_or_default())
                } else {
                    format!("{}*", deg.unwrap_or_default())
                };

            let size = imgproc::get_text_size(
                &label,
                imgproc::FONT_HERSHEY_DUPLEX,
                0.65,
                2,
                &mut 0,
            )?;

            imgproc::put_text(
                &mut frame,
                &label,
                core::Point::new(
                    (x as f32 / scale) as i32 - size.width / 2,
                    (y as f32 / scale) as i32 + size.height / 2,
                ),
                imgproc::FONT_HERSHEY_DUPLEX,
                0.65,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
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
        }

        highgui::imshow("webcam", &frame)?;

        if highgui::wait_key(10)? == 27 {
            break;
        }
    }

    Ok(())
}
