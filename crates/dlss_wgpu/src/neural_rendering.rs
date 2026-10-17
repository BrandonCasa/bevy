//! Native Vulkan NGX feature 18. The application supplies `nvngx_dlssnr.dll`.

use crate::{nvsdk_ngx::*, DlssError, DlssSdk};
use std::{
    path::Path,
    ptr,
    sync::{Arc, Mutex},
};
use wgpu::{hal::api::Vulkan, Device, Queue};

/// Errors at the neural rendering DLL boundary.
#[derive(Debug, thiserror::Error)]
pub enum NeuralRenderingError {
    #[error(transparent)]
    Ngx(#[from] DlssError),
    #[error(transparent)]
    Library(#[from] libloading::Error),
    #[error("unsupported neural rendering library interface (expected Windows x64 NR 310.8)")]
    UnsupportedLibrary,
    #[error("neural rendering has not been enabled on this SDK")]
    NotEnabled,
    #[error("neural rendering requires a Vulkan device")]
    UnsupportedBackend,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

type Init = unsafe extern "C" fn(
    u64,
    *const u16,
    ash::vk::Instance,
    ash::vk::PhysicalDevice,
    ash::vk::Device,
    ash::vk::PFN_vkGetInstanceProcAddr,
    ash::vk::PFN_vkGetDeviceProcAddr,
    i32,
    *const NVSDK_NGX_Parameter,
) -> i32;
type Create = unsafe extern "C" fn(
    ash::vk::CommandBuffer,
    i32,
    *mut NVSDK_NGX_Parameter,
    *mut *mut NVSDK_NGX_Handle,
) -> i32;
type Evaluate = unsafe extern "C" fn(
    ash::vk::CommandBuffer,
    *mut NVSDK_NGX_Handle,
    *mut NVSDK_NGX_Parameter,
    *mut std::ffi::c_void,
) -> i32;
type Release = unsafe extern "C" fn(*mut NVSDK_NGX_Handle) -> i32;
type Populate = unsafe extern "C" fn(*mut NVSDK_NGX_Parameter) -> i32;
type Shutdown = unsafe extern "C" fn(ash::vk::Device) -> i32;

pub(crate) struct Runtime {
    _library: libloading::Library,
    create: Create,
    evaluate: Evaluate,
    release: Release,
    populate: Populate,
    shutdown: Shutdown,
    encode: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    // Pending handles remain owned here until their completion callback or final shutdown.
    retired: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl DlssSdk {
    /// Loads an application-supplied NR 310.8 DLL, including the community Ada build.
    ///
    /// This explicitly opts into the library's private ABI. The known caller-name checks are
    /// adapted in this process only, after verifying their instruction sequences. No file or
    /// driver is modified. Call once before creating neural rendering contexts.
    pub fn enable_neural_rendering(&mut self, path: &Path) -> Result<(), NeuralRenderingError> {
        if self.neural_rendering.is_some() {
            return Ok(());
        }
        self.neural_rendering = Some(unsafe { Runtime::load(path, &self.device)? });
        Ok(())
    }
}

impl Runtime {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    unsafe fn load(path: &Path, device: &Device) -> Result<Self, NeuralRenderingError> {
        // Serialize loader compatibility changes before exposing function pointers to rendering.
        static LOAD: Mutex<()> = Mutex::new(());
        let _load = LOAD.lock().unwrap();
        let library = unsafe { libloading::Library::new(path.canonicalize()?)? };
        let version = unsafe {
            library.get::<unsafe extern "C" fn() -> u32>(b"NVSDK_NGX_GetSnippetVersion\0")?
        };
        if unsafe { version() } != 0x0136_0800 {
            return Err(NeuralRenderingError::UnsupportedLibrary);
        }
        // These offsets are relative to each named export, not ASLR-sensitive image addresses.
        // Each is the conditional branch following the caller module's nvngx.dll substring test.
        let checks: &[(&[u8], usize)] = &[
            (b"NVSDK_NGX_VULKAN_Init_Ext2\0", 0xb1),
            (b"NVSDK_NGX_VULKAN_CreateFeature\0", 0x9d),
            (b"NVSDK_NGX_VULKAN_PopulateParameters_Impl\0", 0x96),
            (b"NVSDK_NGX_VULKAN_ReleaseFeature\0", 0x93),
            (b"NVSDK_NGX_VULKAN_Shutdown1\0", 0x91),
        ];
        let mut addresses = Vec::with_capacity(checks.len());
        for &(name, offset) in checks {
            let entry = unsafe { library.get::<unsafe extern "C" fn()>(name)? };
            let branch = unsafe { (*entry as *mut u8).add(offset) };
            let bytes = unsafe { std::slice::from_raw_parts(branch.sub(3), 5) };
            if bytes != [0x48, 0x85, 0xc0, 0x75, 0x30] && bytes != [0x48, 0x85, 0xc0, 0xeb, 0x30] {
                return Err(NeuralRenderingError::UnsupportedLibrary);
            }
            addresses.push(branch);
        }
        for address in addresses {
            unsafe { allow_native_caller(address)? };
        }
        let init: Init = unsafe { *library.get(b"NVSDK_NGX_VULKAN_Init_Ext2\0")? };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dlss_neural_rendering_hdr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("neural_rendering.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let runtime = Self {
            create: unsafe { *library.get(b"NVSDK_NGX_VULKAN_CreateFeature\0")? },
            evaluate: unsafe { *library.get(b"NVSDK_NGX_VULKAN_EvaluateFeature\0")? },
            release: unsafe { *library.get(b"NVSDK_NGX_VULKAN_ReleaseFeature\0")? },
            populate: unsafe { *library.get(b"NVSDK_NGX_VULKAN_PopulateParameters_Impl\0")? },
            shutdown: unsafe { *library.get(b"NVSDK_NGX_VULKAN_Shutdown1\0")? },
            encode: pipeline("encode"),
            resolve: pipeline("resolve"),
            retired: Default::default(),
            _library: library,
        };
        let hal =
            unsafe { device.as_hal::<Vulkan>() }.ok_or(NeuralRenderingError::UnsupportedBackend)?;
        let instance = hal.shared_instance();
        use std::os::windows::ffi::OsStrExt;
        let data: Vec<u16> = std::env::temp_dir()
            .as_os_str()
            .encode_wide()
            .chain([0])
            .collect();
        check_ngx_result(unsafe {
            init(
                0,
                data.as_ptr(),
                instance.raw_instance().handle(),
                hal.raw_physical_device(),
                hal.raw_device().handle(),
                instance.entry().static_fn().get_instance_proc_addr,
                instance.raw_instance().fp_v1_0().get_device_proc_addr,
                0x13,
                ptr::null(),
            )
        })?;
        Ok(runtime)
    }

    #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
    unsafe fn load(_path: &Path, _device: &Device) -> Result<Self, NeuralRenderingError> {
        Err(NeuralRenderingError::UnsupportedLibrary)
    }

    pub(crate) unsafe fn shutdown(&self, device: ash::vk::Device) {
        // Polling can race a callback already running on another submission thread. This lock
        // joins it before shutdown; a late callback finds its handles already removed.
        let mut retired = self.retired.lock().unwrap();
        for (feature, parameters) in retired.drain(..) {
            unsafe {
                let _ = (self.release)(feature as *mut _);
                let _ = NVSDK_NGX_VULKAN_DestroyParameters(parameters as *mut _);
            }
        }
        let _ = unsafe { (self.shutdown)(device) };
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
unsafe fn allow_native_caller(address: *mut u8) -> Result<(), std::io::Error> {
    unsafe extern "system" {
        fn VirtualProtect(
            address: *mut std::ffi::c_void,
            size: usize,
            protection: u32,
            old: *mut u32,
        ) -> i32;
        fn FlushInstructionCache(
            process: *mut std::ffi::c_void,
            address: *const std::ffi::c_void,
            size: usize,
        ) -> i32;
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
    }
    if unsafe { *address } == 0xeb {
        return Ok(());
    }
    let mut old = 0;
    if unsafe { VirtualProtect(address.cast(), 1, 0x40, &mut old) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    unsafe {
        address.write(0xeb);
    }
    let mut previous = 0;
    let restored = unsafe { VirtualProtect(address.cast(), 1, old, &mut previous) };
    let flushed = unsafe { FlushInstructionCache(GetCurrentProcess(), address.cast(), 1) };
    if restored == 0 || flushed == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Registers the vendor extensions required by the neural rendering kernels.
/// Returns false without modifying the extension list if any requirement is missing.
pub fn register_device_extensions(
    args: &mut wgpu::hal::vulkan::CreateDeviceCallbackArgs,
    adapter: &wgpu::hal::vulkan::Adapter,
) -> bool {
    let required = [
        c"VK_NVX_binary_import",
        c"VK_NVX_image_view_handle",
        c"VK_KHR_push_descriptor",
        c"VK_KHR_buffer_device_address",
    ];
    if !required
        .iter()
        .all(|e| adapter.physical_device_capabilities().supports_extension(e))
    {
        return false;
    }
    for extension in required {
        if !args.extensions.contains(&extension) {
            args.extensions.push(extension);
        }
    }
    true
}

/// Camera-specific neural rendering context. Creation is expensive; retain until size/settings change.
pub struct DlssNeuralRendering {
    device: Device,
    _sdk: Arc<Mutex<DlssSdk>>,
    parameters: *mut NVSDK_NGX_Parameter,
    feature: *mut NVSDK_NGX_Handle,
    size: [u32; 2],
    settings: NeuralRenderingSettings,
    queue: Queue,
    proxy: wgpu::TextureView,
    answer: wgpu::TextureView,
    encode: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    bindings: [Option<FrameBindings>; 2],
}

/// Settings latched by the model when a context is created.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuralRenderingSettings {
    /// Model intensity, in the range 0..=1. Zero preserves the source image.
    pub intensity: f32,
    /// Model preset identifier; zero selects the DLL's default.
    pub preset: u32,
    /// Model style identifier; zero selects its default.
    pub style: u32,
    /// Scale applied to linear HDR before the reversible proxy encoding. Must be positive.
    pub exposure: f32,
}

impl Default for NeuralRenderingSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            preset: 0,
            style: 0,
            exposure: 1.0,
        }
    }
}

impl NeuralRenderingSettings {
    /// Checks settings without invoking NGX or touching a GPU.
    pub fn validate(&self) -> Result<(), DlssError> {
        if !self.intensity.is_finite()
            || !(0.0..=1.0).contains(&self.intensity)
            || !self.exposure.is_finite()
            || self.exposure <= 0.0
        {
            return Err(DlssError::InvalidParameters);
        }
        Ok(())
    }
}

/// Input rectangle in texels, independent for the scene and its depth/motion guides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuralRenderingRect {
    pub origin: [u32; 2],
    pub size: [u32; 2],
}

/// One frame of linear RGBA16F HDR color and depth/motion guides.
pub struct NeuralRenderingFrame<'a> {
    pub color: &'a wgpu::TextureView,
    pub output: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub motion_vectors: &'a wgpu::TextureView,
    pub color_rect: NeuralRenderingRect,
    pub guide_rect: NeuralRenderingRect,
    /// Converts the supplied motion vector values into pixel displacements.
    pub motion_vector_scale: [f32; 2],
    pub inverted_depth: bool,
    pub reset: bool,
}

struct FrameBindings {
    color: wgpu::TextureView,
    output: wgpu::TextureView,
    rect: NeuralRenderingRect,
    encode: wgpu::BindGroup,
    resolve: wgpu::BindGroup,
}

impl DlssNeuralRendering {
    /// Creates a neural rendering context after enabling the application-supplied library.
    pub fn new(
        size: [u32; 2],
        sdk: Arc<Mutex<DlssSdk>>,
        device: &Device,
        queue: &Queue,
    ) -> Result<Self, NeuralRenderingError> {
        Self::with_settings(size, NeuralRenderingSettings::default(), sdk, device, queue)
    }

    /// Creates a context with explicit model settings.
    pub fn with_settings(
        size: [u32; 2],
        settings: NeuralRenderingSettings,
        sdk: Arc<Mutex<DlssSdk>>,
        device: &Device,
        queue: &Queue,
    ) -> Result<Self, NeuralRenderingError> {
        if size.contains(&0) {
            return Err(DlssError::InvalidParameters.into());
        }
        settings.validate()?;
        let _guard = sdk.lock().unwrap();
        if _guard.device != *device {
            return Err(DlssError::InvalidParameters.into());
        }
        let runtime = _guard
            .neural_rendering
            .as_ref()
            .ok_or(NeuralRenderingError::NotEnabled)?;
        let encode = runtime.encode.clone();
        let resolve = runtime.resolve.clone();
        let mut parameters = ptr::null_mut();
        unsafe {
            check_ngx_result(NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters))?;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dlss_neural_rendering_create"),
        });
        let mut feature = ptr::null_mut();
        let result = unsafe {
            check_ngx_result((runtime.populate)(parameters)).and_then(|()| {
                NVSDK_NGX_Parameter_SetUI(parameters, c"DLSSNR.Enabled".as_ptr(), 1);
                NVSDK_NGX_Parameter_SetUI(parameters, c"DLSSNR.Width".as_ptr(), size[0]);
                NVSDK_NGX_Parameter_SetUI(parameters, c"DLSSNR.Height".as_ptr(), size[1]);
                NVSDK_NGX_Parameter_SetUI(parameters, c"CreationNodeMask".as_ptr(), 1);
                NVSDK_NGX_Parameter_SetUI(parameters, c"VisibilityNodeMask".as_ptr(), 1);
                NVSDK_NGX_Parameter_SetUI(
                    parameters,
                    c"DLSSNR.Hint.Render.Preset".as_ptr(),
                    settings.preset,
                );
                NVSDK_NGX_Parameter_SetUI(parameters, c"DLSSNR.Style".as_ptr(), settings.style);
                NVSDK_NGX_Parameter_SetF(
                    parameters,
                    c"DLSSNR.Intensity".as_ptr(),
                    settings.intensity,
                );
                encoder.as_hal_mut::<Vulkan, _, _>(|encoder| {
                    check_ngx_result((runtime.create)(
                        encoder.unwrap().raw_handle(),
                        18,
                        parameters,
                        &mut feature,
                    ))
                })
            })
        };
        if let Err(error) = result {
            unsafe {
                let _ = NVSDK_NGX_VULKAN_DestroyParameters(parameters);
            }
            return Err(error.into());
        }
        drop(_guard);
        // Submission can run retirement callbacks, which acquire the SDK mutex.
        queue.submit([encoder.finish()]);
        let texture = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Ok(Self {
            device: device.clone(),
            _sdk: sdk,
            parameters,
            feature,
            size,
            settings,
            queue: queue.clone(),
            proxy: texture("nr_proxy"),
            answer: texture("nr_answer"),
            encode,
            resolve,
            bindings: [None, None],
        })
    }

    /// Context resolution and settings, used to decide whether recreation is necessary.
    pub fn configuration(&self) -> ([u32; 2], NeuralRenderingSettings) {
        (self.size, self.settings)
    }

    /// Encodes HDR preprocessing into `encoder`, returning NR and HDR resolve command buffers.
    /// Submit all three in order: `[encoder.finish(), nr, resolve]`.
    ///
    /// # Safety
    /// The context and all input textures must outlive GPU execution. Submit these buffers in
    /// the documented order on the context's queue before destroying the context.
    pub unsafe fn render(
        &mut self,
        frame: NeuralRenderingFrame<'_>,
        encoder: &mut wgpu::CommandEncoder,
        adapter: &wgpu::Adapter,
    ) -> Result<[wgpu::CommandBuffer; 2], NeuralRenderingError> {
        self.validate_frame(&frame)?;
        let matches = |b: &FrameBindings| {
            b.color == *frame.color && b.output == *frame.output && b.rect == frame.color_rect
        };
        let slot = self
            .bindings
            .iter()
            .position(|b| b.as_ref().is_some_and(matches))
            .unwrap_or_else(|| self.bindings.iter().position(Option::is_none).unwrap_or(0));
        if self.bindings[slot].as_ref().is_none_or(|b| !matches(b)) {
            use wgpu::util::DeviceExt;
            let words = [
                frame.color_rect.origin[0],
                frame.color_rect.origin[1],
                self.size[0],
                self.size[1],
                self.settings.exposure.to_bits(),
                self.settings.intensity.to_bits(),
                0,
                0,
            ];
            let mut bytes = [0u8; 32];
            for (word, bytes) in words.into_iter().zip(bytes.chunks_exact_mut(4)) {
                bytes.copy_from_slice(&word.to_ne_bytes());
            }
            let uniform = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("nr_view"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let make = |pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry<'_>]| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("nr_view"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries,
                })
            };
            let tex = |binding, view| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            };
            let encode = make(
                &self.encode,
                &[
                    tex(0, frame.color),
                    tex(1, &self.proxy),
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            );
            let resolve = make(
                &self.resolve,
                &[
                    tex(0, frame.color),
                    tex(2, &self.answer),
                    tex(3, frame.output),
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            );
            self.bindings[slot] = Some(FrameBindings {
                color: frame.color.clone(),
                output: frame.output.clone(),
                rect: frame.color_rect,
                encode,
                resolve,
            });
        }
        let bindings = self.bindings[slot].as_ref().unwrap();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nr_encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.encode);
            pass.set_bind_group(0, &bindings.encode, &[]);
            pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        }
        encoder.transition_resources(
            std::iter::empty(),
            [
                (&self.proxy, wgpu::TextureUses::RESOURCE),
                (frame.depth, wgpu::TextureUses::RESOURCE),
                (frame.motion_vectors, wgpu::TextureUses::RESOURCE),
                (&self.answer, wgpu::TextureUses::STORAGE_READ_WRITE),
            ]
            .into_iter()
            .map(|(v, state)| wgpu::TextureTransition {
                texture: v.texture(),
                selector: None,
                state,
            }),
        );
        let mut nr = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nr_evaluate"),
            });
        let sdk = self._sdk.lock().unwrap();
        let runtime = sdk
            .neural_rendering
            .as_ref()
            .ok_or(NeuralRenderingError::NotEnabled)?;
        let mut color = texture_to_ngx(&self.proxy, adapter);
        let mut output = texture_to_ngx(&self.answer, adapter);
        let mut depth = texture_to_ngx(frame.depth, adapter);
        let mut motion = texture_to_ngx(frame.motion_vectors, adapter);
        unsafe {
            for (name, value) in [
                (c"DLSSNR.Color", ptr::from_mut(&mut color)),
                (c"DLSSNR.Output", ptr::from_mut(&mut output)),
                (c"DLSSNR.Depth", ptr::from_mut(&mut depth)),
                (c"DLSSNR.MVec", ptr::from_mut(&mut motion)),
            ] {
                NVSDK_NGX_Parameter_SetVoidPointer(self.parameters, name.as_ptr(), value.cast());
            }
            let native = NeuralRenderingRect {
                origin: [0, 0],
                size: self.size,
            };
            for (keys, rect) in [
                (
                    [
                        c"DLSSNR.ColorSubrectBaseX",
                        c"DLSSNR.ColorSubrectBaseY",
                        c"DLSSNR.ColorSubrectWidth",
                        c"DLSSNR.ColorSubrectHeight",
                    ],
                    native,
                ),
                (
                    [
                        c"DLSSNR.OutputSubrectBaseX",
                        c"DLSSNR.OutputSubrectBaseY",
                        c"DLSSNR.OutputSubrectWidth",
                        c"DLSSNR.OutputSubrectHeight",
                    ],
                    native,
                ),
                (
                    [
                        c"DLSSNR.DepthSubrectBaseX",
                        c"DLSSNR.DepthSubrectBaseY",
                        c"DLSSNR.DepthSubrectWidth",
                        c"DLSSNR.DepthSubrectHeight",
                    ],
                    frame.guide_rect,
                ),
                (
                    [
                        c"DLSSNR.MVecSubrectBaseX",
                        c"DLSSNR.MVecSubrectBaseY",
                        c"DLSSNR.MVecSubrectWidth",
                        c"DLSSNR.MVecSubrectHeight",
                    ],
                    frame.guide_rect,
                ),
            ] {
                for (key, value) in keys.into_iter().zip([
                    rect.origin[0],
                    rect.origin[1],
                    rect.size[0],
                    rect.size[1],
                ]) {
                    NVSDK_NGX_Parameter_SetUI(self.parameters, key.as_ptr(), value);
                }
            }
            NVSDK_NGX_Parameter_SetUI(
                self.parameters,
                c"DLSSNR.Reset".as_ptr(),
                frame.reset as u32,
            );
            NVSDK_NGX_Parameter_SetUI(
                self.parameters,
                c"DLSSNR.DepthInverted".as_ptr(),
                frame.inverted_depth as u32,
            );
            NVSDK_NGX_Parameter_SetF(
                self.parameters,
                c"DLSSNR.MVecScaleX".as_ptr(),
                frame.motion_vector_scale[0],
            );
            NVSDK_NGX_Parameter_SetF(
                self.parameters,
                c"DLSSNR.MVecScaleY".as_ptr(),
                frame.motion_vector_scale[1],
            );
            nr.as_hal_mut::<Vulkan, _, _>(|encoder| {
                check_ngx_result((runtime.evaluate)(
                    encoder.unwrap().raw_handle(),
                    self.feature,
                    self.parameters,
                    ptr::null_mut(),
                ))
            })?;
        }
        let mut resolve = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nr_hdr_resolve"),
            });
        {
            let mut pass = resolve.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nr_hdr_resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve);
            pass.set_bind_group(0, &bindings.resolve, &[]);
            pass.dispatch_workgroups(
                frame.output.texture().width().div_ceil(8),
                frame.output.texture().height().div_ceil(8),
                1,
            );
        }
        Ok([nr.finish(), resolve.finish()])
    }

    fn validate_frame(&self, frame: &NeuralRenderingFrame<'_>) -> Result<(), DlssError> {
        let fits = |rect: NeuralRenderingRect, v: &wgpu::TextureView| {
            !rect.size.contains(&0)
                && rect.origin[0]
                    .checked_add(rect.size[0])
                    .is_some_and(|x| x <= v.texture().width())
                && rect.origin[1]
                    .checked_add(rect.size[1])
                    .is_some_and(|y| y <= v.texture().height())
        };
        if frame.color_rect.size != self.size
            || !fits(frame.color_rect, frame.color)
            || !fits(frame.color_rect, frame.output)
            || !fits(frame.guide_rect, frame.depth)
            || !fits(frame.guide_rect, frame.motion_vectors)
            || frame.color.texture() == frame.output.texture()
            || frame.color.texture().size() != frame.output.texture().size()
            || frame.motion_vector_scale.iter().any(|x| !x.is_finite())
        {
            return Err(DlssError::InvalidParameters);
        }
        for v in [frame.color, frame.output, frame.depth, frame.motion_vectors] {
            if v.texture().sample_count() != 1
                || v.texture().dimension() != wgpu::TextureDimension::D2
                || v.texture().depth_or_array_layers() != 1
            {
                return Err(DlssError::UnsupportedInputFormat);
            }
        }
        if frame.color.texture().format() != wgpu::TextureFormat::Rgba16Float
            || frame.output.texture().format() != wgpu::TextureFormat::Rgba16Float
            || !frame
                .output
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || !frame
                .color
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        {
            return Err(DlssError::UnsupportedInputFormat);
        }
        Ok(())
    }
}

impl Drop for DlssNeuralRendering {
    fn drop(&mut self) {
        // Retire after the last submitted work. No frame-count heuristic or device-wide idle.
        let feature = self.feature as usize;
        let parameters = self.parameters as usize;
        let (release, retired) = {
            let sdk = self._sdk.lock().unwrap();
            let runtime = sdk.neural_rendering.as_ref().unwrap();
            runtime.retired.lock().unwrap().push((feature, parameters));
            (runtime.release, runtime.retired.clone())
        };
        let sdk = Arc::downgrade(&self._sdk);
        // NGX retains native image/view handles. Keep our scratch images and cached bindings
        // alive through feature release, not just until the Rust context is dropped.
        let resources = (
            self.proxy.clone(),
            self.answer.clone(),
            std::mem::take(&mut self.bindings),
        );
        self.queue.on_submitted_work_done(move || {
            // A weak reference avoids a device -> callback -> SDK -> device cycle. SDK teardown
            // drains these callbacks while its DLL and the NGX runtime are still alive.
            let sdk = sdk.upgrade();
            let _guard = sdk.as_ref().map(|sdk| sdk.lock().unwrap());
            let mut retired = retired.lock().unwrap();
            if let Some(index) = retired
                .iter()
                .position(|entry| *entry == (feature, parameters))
            {
                retired.swap_remove(index);
                unsafe {
                    let _ = release(feature as *mut _);
                    let _ = NVSDK_NGX_VULKAN_DestroyParameters(parameters as *mut _);
                }
            }
            drop(resources);
        });
    }
}

// Access is serialized by the owner; NGX handles belong to the retained Vulkan device.
unsafe impl Send for DlssNeuralRendering {}
unsafe impl Sync for DlssNeuralRendering {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nonfinite_and_out_of_range_settings() {
        assert!(NeuralRenderingSettings::default().validate().is_ok());
        for intensity in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(NeuralRenderingSettings {
                intensity,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
        for exposure in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(NeuralRenderingSettings {
                exposure,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
    }
}
