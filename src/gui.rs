use std::time::Duration;
use eframe::CreationContext;
use egui::{CentralPanel, Context};


#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub(crate) struct App {
    counter: u64,
}

impl Default for App {
    fn default() -> Self {
        Self {
            counter: 0,
        }
    }
}

impl App {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Self::default()
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hello, world!");
            self.counter += 1;
            ui.label(self.counter.to_string());
            if ui.button("Reset!").clicked() {
                self.counter = 0;
            }
            ctx.request_repaint_after(Duration::from_secs_f64(1.0 / 60.0))
        });
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
}