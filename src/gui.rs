use std::time::Duration;
use eframe::CreationContext;
use egui::{CentralPanel, Context, TopBottomPanel};

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub(crate) struct App {
    fov: u64,
    paused: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            fov: 70,
            paused: false,
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
        ctx.input(|i| {
            let zoom_delta = i.zoom_delta();
            if zoom_delta != 1.0 {
                let new_zoom = (ctx.zoom_factor() * zoom_delta)
                    .clamp(0.5, 2.0);
                ctx.set_zoom_factor(new_zoom);
            }
        });

        CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.label("FOV: ");
                    ui.add(egui::DragValue::new(&mut self.fov).speed(0.01));
            });
        });
        ctx.request_repaint_after(Duration::from_secs_f64(1.0 / 60.0))
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
}