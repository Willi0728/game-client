use std::time::Duration;
use eframe::CreationContext;
use egui::{CentralPanel, Context};
use egui::WidgetType::DragValue;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub(crate) struct App {
    counter: u64,
    button_is_present: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            counter: 0,
            button_is_present: true,
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
            ctx.input(|i| i.zoom_delta());
            ui.heading("Hello, world!");
            self.counter += 1;
            ui.label(self.counter.to_string());
            if ui.button("Reset!").clicked() {
                self.counter = 0;
            }
            if self.button_is_present {
                if ui.button("Delete myself").clicked() {
                    self.button_is_present = false;
                }
            }
            ui.with_layout(egui::Layout::bottom_up(egui::Align::RIGHT), |ui| {
                if !self.button_is_present {
                    if ui.button("").clicked() {
                        self.button_is_present = true;
                    }
                }
            });
            ctx.request_repaint_after(Duration::from_secs_f64(1.0 / 60.0))
        });
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
}