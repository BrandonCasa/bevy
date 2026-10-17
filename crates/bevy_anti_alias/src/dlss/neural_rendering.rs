//! Native Vulkan DLSS Neural Rendering. Enable `dlss_neural_rendering`, insert
//! [`DlssNeuralRenderingConfig`] and [`super::DlssProjectId`] before `DlssInitPlugin`,
//! then add [`DlssNeuralRendering`] to an HDR camera with `Msaa::Off`.
//!
//! The DLL is supplied by the application and is never bundled or downloaded by Bevy.
//! This feature uses the private NR 310.8 interface, including the community Ada build.
#![doc = include_str!("NEURAL_RENDERING.md")]

use super::DlssSdk;
use bevy_app::App;
use bevy_camera::{Camera, Camera3d, CameraMainTextureUsages, Hdr, MainPassResolutionOverride};
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass, ViewPrepassTextures},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_math::{UVec2, Vec4Swizzles};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_resource::{TextureFormat, TextureUsages},
    renderer::{
        raw_vulkan_init::AdditionalVulkanFeatures, RenderAdapter, RenderContext, RenderDevice,
        RenderQueue, ViewQuery,
    },
    sync_component::SyncComponent,
    sync_world::RenderEntity,
    view::{prepare_view_targets, ExtractedView, Msaa, ViewTarget},
    ExtractSchedule, MainWorld, Render, RenderApp, RenderSystems,
};
use dlss_wgpu::neural_rendering::{
    DlssNeuralRendering as Context, NeuralRenderingFrame, NeuralRenderingRect,
    NeuralRenderingSettings,
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tracing::warn;

/// Explicit application opt-in to loading the private NR interface.
#[derive(Resource, Clone)]
pub struct DlssNeuralRenderingConfig {
    /// Absolute path to the application-supplied `nvngx_dlssnr.dll`.
    pub library_path: PathBuf,
}

/// Result of initializing NR on this device. No camera is enabled automatically.
#[derive(Resource, Debug)]
pub enum DlssNeuralRenderingStatus {
    Ready,
    Unavailable(String),
}

/// Applies neural rendering to the HDR scene before tonemapping and UI.
/// Requires a Vulkan renderer, an initialized NR library, and [`Msaa::Off`].
/// Works independently of the camera's optional DLSS SR/RR component.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct DlssNeuralRendering {
    /// Model intensity in 0..=1. Zero skips the pass entirely.
    pub intensity: f32,
    /// Model preset; zero chooses the DLL default.
    pub preset: u32,
    /// Model style; zero chooses the DLL default.
    pub style: u32,
    /// Positive scale applied before encoding linear HDR for the model.
    pub exposure: f32,
    /// Reset model history after a camera cut. Consumed during extraction.
    pub reset: bool,
}

impl Default for DlssNeuralRendering {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            preset: 0,
            style: 0,
            exposure: 1.0,
            reset: true,
        }
    }
}

impl DlssNeuralRendering {
    fn settings(&self) -> NeuralRenderingSettings {
        NeuralRenderingSettings {
            intensity: self.intensity,
            preset: self.preset,
            style: self.style,
            exposure: self.exposure,
        }
    }
}

impl SyncComponent<RenderApp> for DlssNeuralRendering {
    type Target = Self;
}

pub(super) struct NeuralRenderingExtensions;

#[derive(Component)]
struct PreparedNr {
    size: UVec2,
    settings: NeuralRenderingSettings,
    // None remembers a failed configuration, avoiding a create/retry loop every frame.
    context: Mutex<Option<Context>>,
}

pub(super) fn setup(app: &mut App, sdk: &Arc<Mutex<dlss_wgpu::DlssSdk>>) {
    let Some(config) = app.world().get_resource::<DlssNeuralRenderingConfig>() else {
        return;
    };
    let supported = app
        .sub_app(RenderApp)
        .world()
        .resource::<AdditionalVulkanFeatures>()
        .has::<NeuralRenderingExtensions>();
    let result = if supported {
        sdk.lock()
            .unwrap()
            .enable_neural_rendering(&config.library_path)
            .map_err(|error| error.to_string())
    } else {
        Err("Required Vulkan NR extensions are unavailable".to_string())
    };
    if let Err(error) = result {
        warn!("DLSS Neural Rendering unavailable: {error}");
        app.insert_resource(DlssNeuralRenderingStatus::Unavailable(error));
        return;
    }
    app.insert_resource(DlssNeuralRenderingStatus::Ready);
    app.sub_app_mut(RenderApp)
        .add_systems(ExtractSchedule, extract)
        .add_systems(
            Render,
            prepare
                .in_set(RenderSystems::PrepareViews)
                .before(prepare_view_targets),
        )
        .add_systems(
            Core3d,
            render
                .in_set(Core3dSystems::EarlyPostProcess)
                .after(super::node::dlss_super_resolution)
                .after(super::node::dlss_ray_reconstruction)
                .after(crate::taa::temporal_anti_alias),
        );
}

fn extract(mut commands: Commands, mut main: ResMut<MainWorld>) {
    let mut cameras = main.query_filtered::<(
        RenderEntity,
        &Camera,
        &Msaa,
        Option<&mut DlssNeuralRendering>,
    ), With<Camera3d>>();
    for (entity, camera, msaa, settings) in cameras.iter_mut(&mut main) {
        let Ok(mut render_entity) = commands.get_entity(entity) else {
            continue;
        };
        if let Some(mut settings) = settings
            && camera.is_active
            && *msaa == Msaa::Off
            && settings.intensity != 0.0
        {
            render_entity.insert(settings.clone());
            settings.reset = false;
        } else {
            render_entity.remove::<(DlssNeuralRendering, PreparedNr)>();
        }
    }
}

fn prepare(
    mut commands: Commands,
    mut views: Query<(
        Entity,
        &ExtractedView,
        &DlssNeuralRendering,
        &mut CameraMainTextureUsages,
        Option<&PreparedNr>,
    )>,
    sdk: Res<DlssSdk>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    for (entity, view, settings, mut usages, prepared) in &mut views {
        let size = view.viewport.zw();
        let configuration = settings.settings();
        usages.0 |= TextureUsages::STORAGE_BINDING;
        if size.x == 0
            || size.y == 0
            || prepared.is_some_and(|p| {
                p.size == size
                    && p.settings.preset == configuration.preset
                    && p.settings.style == configuration.style
                    && p.settings.intensity.to_bits() == configuration.intensity.to_bits()
                    && p.settings.exposure.to_bits() == configuration.exposure.to_bits()
            })
        {
            continue;
        }
        let context = match Context::with_settings(
            size.to_array(),
            configuration,
            sdk.0.clone(),
            device.wgpu_device(),
            &queue,
        ) {
            Ok(context) => Some(context),
            Err(error) => {
                warn!("DLSS Neural Rendering context: {error}");
                None
            }
        };
        commands.entity(entity).insert(PreparedNr {
            size,
            settings: configuration,
            context: Mutex::new(context),
        });
    }
}

fn render(
    view: ViewQuery<(
        &DlssNeuralRendering,
        &PreparedNr,
        &ExtractedView,
        &ViewTarget,
        &ViewPrepassTextures,
        Option<&MainPassResolutionOverride>,
    )>,
    adapter: Res<RenderAdapter>,
    mut ctx: RenderContext,
) {
    let (settings, prepared, view, target, prepass, resolution) = view.into_inner();
    let (Some(depth), Some(motion)) = (&prepass.depth, &prepass.motion_vectors) else {
        return;
    };
    if target.main_texture_format() != TextureFormat::Rgba16Float {
        return;
    }
    let mut context = prepared.context.lock().unwrap();
    let Some(context_ref) = context.as_mut() else {
        return;
    };
    let guide_size = resolution.map_or(prepared.size, |r| r.0);
    let frame = NeuralRenderingFrame {
        color: target.main_texture_view(),
        output: target.main_texture_other_view(),
        depth: &depth.texture.default_view,
        motion_vectors: &motion.texture.default_view,
        color_rect: NeuralRenderingRect {
            origin: view.viewport.xy().to_array(),
            size: prepared.size.to_array(),
        },
        guide_rect: NeuralRenderingRect {
            origin: [0, 0],
            size: guide_size.to_array(),
        },
        motion_vector_scale: (-guide_size.as_vec2()).to_array(),
        inverted_depth: true,
        reset: settings.reset,
    };
    // The render world retains this context and the view resources through submission; returned
    // command buffers are appended in order by RenderContext and use the same renderer queue.
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "dlss_neural_rendering");
    match unsafe { context_ref.render(frame, ctx.command_encoder(), &adapter) } {
        Ok(buffers) => {
            target.post_process_write();
            for buffer in buffers {
                ctx.add_command_buffer(buffer);
            }
        }
        Err(error) => {
            warn!("DLSS Neural Rendering stopped for this camera: {error}");
            context.take();
        }
    }
    span.end(ctx.command_encoder());
}
