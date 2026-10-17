#[cfg(not(feature = "mock"))]
use dlss_wgpu::{
    neural_rendering::{
        self, DlssNeuralRendering, NeuralRenderingFrame, NeuralRenderingRect,
        NeuralRenderingSettings,
    },
    DlssSdk,
};
#[cfg(not(feature = "mock"))]
use std::path::PathBuf;
#[cfg(not(feature = "mock"))]
use wgpu::{hal::api::Vulkan, *};

#[cfg(not(feature = "mock"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: neural_rendering_probe <DLL directory>")?,
    );
    let project = uuid::Uuid::from_u128(0x44b98449_0c15_4a22_bfa2_7e860675ab61);
    let mut support = dlss_wgpu::FeatureSupport::default();
    let instance = dlss_wgpu::create_instance(
        project,
        &InstanceDescriptor::new_without_display_handle(),
        &mut support,
    )?;
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default()))?;
    println!("adapter: {:?}", adapter.get_info());
    let descriptor = DeviceDescriptor::default();
    let (device, queue) = unsafe {
        let raw = adapter
            .as_hal::<Vulkan>()
            .ok_or("Vulkan adapter required")?;
        let open = raw.open_with_callback(
            descriptor.required_features,
            &adapter.limits(),
            &descriptor.memory_hints,
            Some(Box::new(|mut args| {
                assert!(
                    neural_rendering::register_device_extensions(&mut args, &raw),
                    "NR extensions unavailable"
                );
            })),
        )?;
        adapter.create_device_from_hal::<Vulkan>(open, &descriptor)?
    };
    let sdk = DlssSdk::new(project, device.clone())?;
    println!("NGX initialized");
    sdk.lock()
        .unwrap()
        .enable_neural_rendering(&path.join("nvngx_dlssnr.dll"))?;
    println!("NR library initialized");
    let make = |label, format, usage| {
        device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = make(
        "scene",
        TextureFormat::Rgba16Float,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    );
    let output = make(
        "output",
        TextureFormat::Rgba16Float,
        TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
    );
    let depth = make(
        "depth",
        TextureFormat::R32Float,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    );
    let motion = make(
        "motion",
        TextureFormat::Rg16Float,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    );
    let mut pixels = Vec::with_capacity(512 * 512 * 8);
    for y in 0..512 {
        for x in 0..512 {
            for h in [
                if x / 32 % 2 == 0 { 0x3400u16 } else { 0x4000 },
                if y / 32 % 2 == 0 { 0x3800 } else { 0x3c00 },
                0x3800,
                0x3c00,
            ] {
                pixels.extend_from_slice(&h.to_le_bytes());
            }
        }
    }
    for (texture, bytes, stride) in [
        (&color, pixels.clone(), 4096),
        (&depth, vec![0u8; 512 * 512 * 4], 2048),
        (&motion, vec![0u8; 512 * 512 * 4], 2048),
    ] {
        queue.write_texture(
            texture.as_image_copy(),
            &bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(512),
            },
            texture.size(),
        );
    }
    let cv = color.create_view(&Default::default());
    let ov = output.create_view(&Default::default());
    let dv = depth.create_view(&Default::default());
    let mv = motion.create_view(&Default::default());
    let rect = NeuralRenderingRect {
        origin: [0, 0],
        size: [512, 512],
    };
    for (size, intensity) in [([512, 512], 1.0), ([256, 384], 1.0), ([512, 512], 0.0)] {
        let mut context = DlssNeuralRendering::with_settings(
            size,
            NeuralRenderingSettings {
                intensity,
                ..Default::default()
            },
            sdk.clone(),
            &device,
            &queue,
        )?;
        let color_rect = NeuralRenderingRect {
            origin: [0, 0],
            size,
        };
        println!("NR context: {size:?}, intensity={intensity}");
        for index in 0..8 {
            let mut encoder = device.create_command_encoder(&Default::default());
            let [nr, resolve] = unsafe {
                context.render(
                    NeuralRenderingFrame {
                        color: &cv,
                        output: &ov,
                        depth: &dv,
                        motion_vectors: &mv,
                        color_rect,
                        guide_rect: rect,
                        motion_vector_scale: [-512.0, -512.0],
                        inverted_depth: true,
                        reset: index == 0,
                    },
                    &mut encoder,
                    &adapter,
                )?
            };
            queue.submit([encoder.finish(), nr, resolve]);
        }
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("result"),
            size: pixels.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4096),
                    rows_per_image: Some(512),
                },
            },
            output.size(),
        );
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.slice(..).map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        receiver.recv()??;
        let result = readback.slice(..).get_mapped_range()?;
        let changed = result
            .chunks_exact(2)
            .zip(pixels.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        let nonfinite = result
            .chunks_exact(2)
            .filter(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7c00 == 0x7c00)
            .count();
        println!("NR evaluated: changed half-float channels={changed}, nonfinite={nonfinite}");
        let rgb: Vec<f32> = result
            .chunks_exact(8)
            .flat_map(|p| [half(p[0], p[1]), half(p[2], p[3]), half(p[4], p[5])])
            .collect();
        let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
        let minimum = rgb.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = rgb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "NR output RGB: mean={mean}, min={minimum}, max={maximum}, first={:?}",
            &rgb[..9]
        );
        assert!(
            nonfinite == 0
                && mean > 0.01
                && maximum - minimum > 0.05
                && if intensity == 0.0 {
                    changed == 0
                } else {
                    changed > 0
                },
            "NR must produce a finite, nonuniform, non-passthrough result"
        );
        for y in 0..512usize {
            for x in 0..512usize {
                if x >= size[0] as usize || y >= size[1] as usize {
                    let offset = (y * 512 + x) * 8;
                    assert_eq!(
                        &result[offset..offset + 8],
                        &pixels[offset..offset + 8],
                        "pixels outside the viewport must be preserved"
                    );
                }
            }
        }
        drop(result);
        readback.unmap();
        drop(context);
        // Recreate immediately: the next submit may execute the previous retirement callback.
        println!("NR feature retirement queued");
    }
    drop(sdk);
    Ok(())
}

#[cfg(not(feature = "mock"))]
fn half(a: u8, b: u8) -> f32 {
    let h = u16::from_le_bytes([a, b]);
    let e = ((h >> 10) & 31) as i32;
    let m = (h & 1023) as f32;
    let v = if e == 0 {
        m * 2.0f32.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2.0f32.powi(e - 15)
    };
    if h & 0x8000 != 0 {
        -v
    } else {
        v
    }
}

#[cfg(not(feature = "mock"))]
fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    struct Wake(std::thread::Thread);
    impl std::task::Wake for Wake {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Wake(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::park();
    }
}

#[cfg(feature = "mock")]
fn main() {
    eprintln!("The neural rendering probe requires the native backend, not mock.");
}
