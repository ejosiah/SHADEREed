#version 460

#define ATMOSPHERE_TOP_RADIUS 6420.0
#define ATMOSPHERE_BOTTOM_RADIUS 6360.0
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define IRRADIANCE_TEXTURE_WIDTH 64
#define IRRADIANCE_TEXTURE_HEIGHT 16
#define SUN_ANGULAR_RADIUS 0.004675
#define RAYLEIGH_SCATTERING vec3(0.005802,0.013558,0.033100)
#define MIE_EXTINCTION vec3(0.000650,0.001881,0.000085)
#define ABSORPTION_EXTINCTION vec3(0.000650,0.001881,0.000085)
#define SOLAR_IRRADIANCE vec3(1.474000, 1.850400, 1.911980)

struct DensityProfileLayer {
  float width;
  float exp_term;
  float exp_scale;
  float linear_term;
  float constant_term;
};

// An atmosphere density profile made of several layers on top of each other
// (from bottom to top). The width of the last layer is ignored, i.e. it always
// extend to the top atmosphere boundary. The profile values vary between 0
// (null density) to 1 (maximum density).
struct DensityProfile {
  DensityProfileLayer layers[2];
};

const DensityProfile rayleigh_density = DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000,0.000000,0.000000,0.000000,0.000000),DensityProfileLayer(0.000000,1.000000,-0.125000,0.000000,0.000000)));
const DensityProfile mie_density = DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000,0.000000,0.000000,0.000000,0.000000),DensityProfileLayer(0.000000,1.000000,-0.833333,0.000000,0.000000)));
const DensityProfile absorption_density = DensityProfile(DensityProfileLayer[2](DensityProfileLayer(25.000000,0.000000,0.000000,0.066667,-0.666667),DensityProfileLayer(0.000000,0.000000,0.000000,-0.066667,2.666667)));


float ClampCosine(float mu) {
  return clamp(mu, -1.0,1.0);
}

float ClampDistance(float d) {
  return max(d, 0.0);
}

float ClampRadius(float r) {
  return clamp(r, ATMOSPHERE_BOTTOM_RADIUS, ATMOSPHERE_TOP_RADIUS);
}

float SafeSqrt(float a) {
  return sqrt(max(a, 0.0));
}

float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

float GetLayerDensity(DensityProfileLayer layer, float altitude) {
  float density = layer.exp_term * exp(layer.exp_scale * altitude) +
      layer.linear_term * altitude + layer.constant_term;
  return clamp(density, 0.0, 1.0);
}

float GetProfileDensity(DensityProfile profile, float altitude) {
  return altitude < profile.layers[0].width ?
      GetLayerDensity(profile.layers[0], altitude) :
      GetLayerDensity(profile.layers[1], altitude);
}

void GetRMuSFromIrradianceTextureUv(vec2 uv, out float r, out float mu_s) {
    float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
    float x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
    r = ATMOSPHERE_BOTTOM_RADIUS +
        x_r * (ATMOSPHERE_TOP_RADIUS - ATMOSPHERE_BOTTOM_RADIUS);
    mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
}

float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) +
                        ATMOSPHERE_TOP_RADIUS * ATMOSPHERE_TOP_RADIUS;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}


vec2 GetTransmittanceTextureUvFromRMu(float r, float mu) {
    float H = sqrt(ATMOSPHERE_TOP_RADIUS * ATMOSPHERE_TOP_RADIUS -
                    ATMOSPHERE_BOTTOM_RADIUS * ATMOSPHERE_BOTTOM_RADIUS);
    float rho =
            SafeSqrt(r * r - ATMOSPHERE_TOP_RADIUS * ATMOSPHERE_BOTTOM_RADIUS);
    float d = DistanceToTopAtmosphereBoundary(r, mu);
    float d_min = ATMOSPHERE_TOP_RADIUS - r;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;
    return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
                GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}



vec3 GetTransmittanceToTopAtmosphereBoundary(sampler2D transmittance_texture, float r, float mu) {

    vec2 uv = GetTransmittanceTextureUvFromRMu(r, mu);
    return vec3(texture(transmittance_texture, uv));
}

vec3 ComputeDirectIrradiance(sampler2D transmittance_texture, float r, float mu_s) {
    float alpha_s = SUN_ANGULAR_RADIUS;
    float average_cosine_factor =
            mu_s < -alpha_s ? 0.0 : (mu_s > alpha_s ? mu_s :
                                     (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));
    return SOLAR_IRRADIANCE *
           GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu_s) * average_cosine_factor;
}

vec3 ComputeDirectIrradianceTexture(sampler2D transmittance_texture, vec2 uv) {
    float r;
    float mu_s;
    GetRMuSFromIrradianceTextureUv(uv, r, mu_s);
    return ComputeDirectIrradiance(transmittance_texture, r, mu_s);
}

uniform sampler2D transmittance_texture;

uniform vec2 ViewportSize;

layout(location = 0) out vec4 delta_irradiance;
layout(location = 1) out vec3 irradiance;

void main(){
	vec2 uv = gl_FragCoord.xy/ViewportSize;
	
	delta_irradiance.rgb = ComputeDirectIrradianceTexture(transmittance_texture, uv);
	irradiance = vec3(0); 
}