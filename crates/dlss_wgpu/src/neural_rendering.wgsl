struct View {
    origin: vec2<u32>,
    size: vec2<u32>,
    exposure: f32,
    intensity: f32,
    padding: vec2<u32>,
}
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var proxy: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var answer: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> view: View;

fn linear_to_srgb(x: vec3<f32>) -> vec3<f32> {
    return select(12.92 * x, 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x > vec3<f32>(0.0031308));
}
fn srgb_to_linear(x: vec3<f32>) -> vec3<f32> {
    let p = max(x, vec3<f32>(0.0));
    return select(p / 12.92, pow((p + 0.055) / 1.055, vec3<f32>(2.4)), p > vec3<f32>(0.04045));
}
fn encode_hdr(color: vec3<f32>) -> vec3<f32> {
    let x = max(color * view.exposure, vec3<f32>(0.0));
    return x / (vec3<f32>(1.0) + x);
}
@compute @workgroup_size(8, 8)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= view.size) { return; }
    let color = textureLoad(scene, vec2<i32>(id.xy + view.origin), 0);
    textureStore(proxy, id.xy, vec4<f32>(linear_to_srgb(encode_hdr(color.rgb)), 1.0));
}
@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(output)) { return; }
    let color = textureLoad(scene, vec2<i32>(id.xy), 0);
    // Preserve the rest of the render target when a camera uses a sub-viewport.
    if any(id.xy < view.origin) || any(id.xy - view.origin >= view.size) || view.intensity == 0.0 {
        textureStore(output, id.xy, color); return;
    }
    let model = srgb_to_linear(textureLoad(answer, vec2<i32>(id.xy - view.origin), 0).rgb);
    let encoded = encode_hdr(color.rgb);
    // Anchor the model's linear-light ratio to the original HDR scene, preserving highlights.
    let ratio = clamp(model / max(encoded, vec3<f32>(0.0001)), vec3<f32>(0.0), vec3<f32>(16.0));
    textureStore(output, id.xy, vec4<f32>(min(color.rgb * ratio, vec3<f32>(65504.0)), color.a));
}
