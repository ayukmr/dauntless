mod types;
mod config;
mod fft;
mod oper;
mod post;
mod mask;
mod uf;
mod shapes;
mod filters;
mod tags;
mod decode;
mod process;

pub use types::Tag;
pub use config::Config;

use types::Lightness;

use crate::types::Mask;

pub fn set_config(config: Config) {
    config::set(config);
}

pub fn get_config() -> Config {
    config::cfg()
}

pub fn tags(data: &Lightness) -> Vec<Tag> {
    process::process(data).1
}

pub fn tags2(data: &Lightness) -> (Mask, Vec<Tag>) {
    process::process(data)
}
