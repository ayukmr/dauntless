#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Config {
    pub fov: f32,

    pub hyst_high: f32,
    pub hyst_low: f32,

    pub filter_ratios: bool,
    pub filter_angles: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fov: 75.0_f32,

            hyst_high: 0.05,
            hyst_low: 0.025,

            filter_ratios: true,
            filter_angles: true,
        }
    }
}
