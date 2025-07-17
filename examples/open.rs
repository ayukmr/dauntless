use ndarray::Array2;
use image::imageops::{self, FilterType};

fn main() {
    let img = image::open("data/3.jpg").unwrap();

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

    let data = Array2::from_shape_vec(
        (sh, sw),
        resized.into_vec(),
    ).unwrap().mapv(|l| l as f32) / 255.0;

    println!("{:?}", dauntless::tags(data));
}
