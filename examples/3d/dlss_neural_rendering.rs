//! Application-owned DLSS NR library, native Vulkan, and a per-camera toggle.
//! Run with `--features dlss_neural_rendering --example dlss_neural_rendering -- <DLL path>`.

use bevy::{
    anti_alias::dlss::{
        neural_rendering::{
            DlssNeuralRendering, DlssNeuralRenderingConfig, DlssNeuralRenderingStatus,
        },
        DlssProjectId,
    },
    prelude::*,
};

fn main() {
    let library_path = std::env::args_os()
        .nth(1)
        .expect("Pass the path to nvngx_dlssnr.dll")
        .into();
    App::new()
        .insert_resource(DlssProjectId(bevy::asset::uuid::Uuid::from_u128(
            0x44b98449_0c15_4a22_bfa2_7e860675ab61,
        )))
        .insert_resource(DlssNeuralRenderingConfig { library_path })
        .add_plugins(DefaultPlugins.set(bevy::render::RenderPlugin {
            render_creation: bevy::render::settings::RenderCreation::Automatic(Box::new(
                bevy::render::settings::WgpuSettings {
                    backends: Some(bevy::render::settings::Backends::VULKAN),
                    ..default()
                },
            )),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, toggle)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    status: Option<Res<DlssNeuralRenderingStatus>>,
) {
    info!(
        ?status,
        "Neural rendering initialization; F11 toggles the camera pass"
    );
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 2.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.7, 0.4, 0.2))),
        Transform::from_xyz(0.0, 1.0, 0.0),
    ));
    commands.spawn((
        PointLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 6.0, 4.0),
    ));
    commands.spawn((
        Camera3d::default(),
        Msaa::Off,
        DlssNeuralRendering::default(),
        Transform::from_xyz(4.0, 3.0, 6.0).looking_at(Vec3::Y, Vec3::Y),
    ));
}

fn toggle(keys: Res<ButtonInput<KeyCode>>, mut cameras: Query<&mut DlssNeuralRendering>) {
    if keys.just_pressed(KeyCode::F11) {
        for mut camera in &mut cameras {
            camera.intensity = if camera.intensity == 0.0 { 1.0 } else { 0.0 };
            camera.reset = true;
        }
    }
}
