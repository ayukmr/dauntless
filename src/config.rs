use crate::detector::Detector;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Config {
    pub fov_rad: f32,

    pub harris_k: f32,
    pub harris_thresh: f32,

    pub hyst_low: f32,
    pub hyst_high: f32,

    pub filter_ratios: bool,
    pub filter_angles: bool,
    pub filter_enclosed: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fov_rad: 75.0_f32.to_radians(),

            harris_k: 0.01,
            harris_thresh: 0.05,

            hyst_low: 0.0125,
            hyst_high: 0.05,

            filter_ratios: true,
            filter_angles: true,
            filter_enclosed: false,
        }
    }
}

impl Detector {
    pub fn set_config(&mut self, config: Config) {
        self.config = config;
    }

    pub fn get_config(&self) -> Config {
        self.config
    }
}
