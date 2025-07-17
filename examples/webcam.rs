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
            for (x, y) in [tl, tr, bl, br] {
                let pos = core::Point::new(
                    (x as f32 / scale) as i32,
                    (y as f32 / scale) as i32,
                );

                imgproc::circle(
                    &mut frame,
                    pos,
                    5,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
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
