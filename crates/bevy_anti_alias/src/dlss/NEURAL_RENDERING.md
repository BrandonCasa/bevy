# DLSS Neural Rendering

Enable the Bevy fork's `dlss_neural_rendering` feature. This is native Windows x64
Vulkan integration of NGX feature 18, including the supplied community Ada/RTX 4000
build of NR 310.8. The application supplies the DLL; no binary is embedded,
downloaded, or copied by these crates.

Before adding `DefaultPlugins`, insert `DlssProjectId` and
`DlssNeuralRenderingConfig { library_path }`. The normal `DlssInitPlugin` registers
the required Vulkan extensions before device creation. Select the Vulkan backend.
Check `DlssNeuralRenderingStatus` after plugin initialization, then add
`DlssNeuralRendering` to an HDR camera with `Msaa::Off`. SR and RR are optional on
that camera. See `examples/3d/dlss_neural_rendering.rs`.

The pass consumes linear RGBA16F color and the depth/motion prepasses. It runs
after SR/RR/TAA and before tonemapping and UI. A bounded sRGB proxy feeds the
model; its linear-light color ratio is composed against the original HDR scene.
Alpha and pixels outside a camera viewport are preserved. `exposure` controls the
HDR proxy scale. Intensity zero bypasses the pass. Set `reset` after a camera cut;
extraction consumes that flag. Model settings are latched at creation, so changing
them or the viewport size recreates the context. Do not animate model settings
every frame.

The low-level implementation lives in the independent `crates/dlss_wgpu` package.
It uses wgpu's HAL access and tracked texture transitions; wgpu core is unchanged.
The DLL is loaded directly because the NGX core rejects the community binary's
signature. The private ABI adapter accepts NR 310.8 and verifies the known caller
check instruction sequences before adapting those checks in the loaded module.
No driver, system library, other process, or on-disk DLL is patched. Unknown
interfaces return an error. Capability initialization and camera creation failures
are reported once; failed configurations are not retried every frame.

Contexts own their scratch images, and each device owns the encode/resolve
pipelines. Both ping-pong bind groups are cached. Feature retirement follows GPU
submission completion, and final SDK teardown drains pending retirements before
unloading the DLL. There is no frame-delay retirement heuristic or per-frame GPU
idle wait.

For applications with external Bevy plugins, use one consistent Bevy source for
all dependencies, as with other Bevy forks. Cargo patches in a dependency are not
inherited by its consumers. Origins supplies these patches at its workspace root.

## Validation

The `dlss_wgpu` package can build and run independently of Bevy:

```text
cargo test --manifest-path crates/dlss_wgpu/Cargo.toml --features neural_rendering --lib
cargo run --manifest-path crates/dlss_wgpu/Cargo.toml --features neural_rendering --example neural_rendering_probe -- <DLL directory>
```

The probe runs real Vulkan evaluations, reads back the result, and rejects
nonfinite, uniform, or unchanged output. It is an API/lifecycle check, not a claim
of visual quality in every scene. SDK build prerequisites remain those documented
by `dlss_wgpu`. `DLSS_WGPU_LOG=1` enables NGX diagnostic logging.

ABI investigation references:

- https://github.com/Dagherbou/OptiScaler_DLSSNR/tree/dlss-neural-rendering/OptiScaler/dlssnr
- https://github.com/faisalkindi/DLSS5oneclick

This implementation uses the observed interface and independently written Rust
and WGSL. It does not incorporate OptiScaler's interception or shader code.
