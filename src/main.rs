#[cfg(not(target_arch = "wasm32"))]
use eframe::NativeOptions;
#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use eframe::web_sys;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::wasm_bindgen;

mod gui;
mod projector;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let options = NativeOptions::default();
    eframe::run_native(
        "Project",
        options,
        Box::new(|cc| Ok(Box::new(gui::App::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn start() {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let window = web_sys::window().expect("no global `window` exists");
    let document = window.document().expect("should have a document on window");
    let canvas = document
        .get_element_by_id("canvas") 
        .expect("failed to find canvas element")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("element is not a canvas");

    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(gui::App::new(cc)))),
            )
            .await
            .expect("failed to start eframe");
    });
}
