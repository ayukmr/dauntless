use std::sync::{Arc, RwLock};

use once_cell::sync::OnceCell;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Config {
    pub half_fov: f32,

    pub harris_k: f32,
    pub harris_thresh: f32,

    pub hyst_low: f32,
    pub hyst_high: f32,

    pub filter_paras: bool,
    pub filter_enclosed: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            half_fov: 75.0_f32.to_radians() / 2.0,

            harris_k: 0.01,
            harris_thresh: 0.05,

            hyst_low: 0.0125,
            hyst_high: 0.05,

            filter_paras: true,
            filter_enclosed: true,
        }
    }
}

static CONFIG: OnceCell<Arc<RwLock<Config>>> = OnceCell::new();

pub fn set(config: Config) {
    if let Some(c) = CONFIG.get() {
        *c.write().unwrap() = config;
    } else {
        CONFIG.set(Arc::new(RwLock::new(config))).ok();
    }
}

pub fn cfg() -> Config {
    *CONFIG.get().unwrap().read().unwrap()
}
