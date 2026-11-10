//! Prints what auto exposure measures of a saved frame and what it would do (a diagnostic).
fn main() {
    let path = std::env::args().nth(1).expect("a JPEG");
    let bytes = std::fs::read(path).unwrap();
    match continuum_capture::exposure::luma_plane("MJPEG", &bytes, 1280, 720, 0) {
        Some((plane, w, h)) => {
            let l = continuum_capture::exposure::luma(&plane, 1).unwrap();
            println!("{w}x{h} luma {:.1} clipped {:.3}", l.mean, l.clipped);
            let range = continuum_capture::exposure::Range { min: 1, max: 2500, step: 1 };
            let mut a = continuum_capture::exposure::AutoExposure::new(range, 100_000, 33_333_333, Default::default());
            println!("step from 156: {:?} ({})", a.step(0, 156, l), a.last());
        }
        None => println!("luma_plane gave None"),
    }
}
