use std::time::Instant;
use image::imageops::{self, FilterType};

fn main() {
    let img = image::open("data/4.jpg").unwrap();
    let light = img.to_luma8();

    let (w, h) = light.dimensions();
    let scale = 400.0 / u32::max(w, h) as f32;

    let sw = (scale * w as f32) as usize;
    let sh = (scale * h as f32) as usize;

    let resized = imageops::resize(
        &light,
        sw as u32,
        sh as u32,
        FilterType::Triangle,
    );

    let data =
        &resized
            .into_vec()
            .into_iter()
            .map(|l| l as f32 / 255.0)
            .collect();

    let mut detector = dauntless::Detector::default();

    let runs = 1000;

    let mut times = [0.0; 1000];

    for i in 0..runs {
        let start = Instant::now();
        let _ = detector.tags(sw, sh, &data);
        let ms = start.elapsed().as_secs_f32() * 1000.0;

        print!("\r{:.2} ms", ms);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        times[i] = ms;
    }

    let avg = times.iter().sum::<f32>() / runs as f32;

    println!("\navg ms {:.2}", avg);
    println!("\navg fps {:.2}", 1000.0 / avg);
}
