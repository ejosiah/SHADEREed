#version 460



layout(location = 0) out vec4 fragColor;

uniform vec2 ViewportSize;

#define ATMOSPHERE_TOP_RADIUS 6420.0
#define ATMOSPHERE_BOTTOM_RADIUS 6360.0
#define TRANSMITTANCE_TEXTURE_WIDTH int(ViewportSize.x)
#define TRANSMITTANCE_TEXTURE_HEIGHT int(ViewportSize.y)
#define RAYLEIGH_SCATTERING vec3(0.005802,0.013558,0.033100)
#define MIE_EXTINCTION vec3(0.000650,0.001881,0.000085)
#define ABSORPTION_EXTINCTION vec3(0.000650,0.001881,0.000085)


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

float DistanceToTopAtmosphereBoundary(float r, float mu) {
//    assert(r <= atmosphere.top_radius);
//    assert(mu >= -1.0 && mu <= 1.0);
    float discriminant = r * r * (mu * mu - 1.0) +
                        ATMOSPHERE_TOP_RADIUS * ATMOSPHERE_TOP_RADIUS;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}



float ComputeOpticalLengthToTopAtmosphereBoundary(DensityProfile profile,
        float r, float mu) {
    // assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
//  assert(mu >= -1.0 && mu <= 1.0);
    // Number of intervals for the numerical integration.
    const int SAMPLE_COUNT = 500;
    // The integration step, i.e. the float of each integration interval.
    float dx =
            DistanceToTopAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT);
    // Integration loop.
    float result = 0.0;
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        float d_i = float(i) * dx;
        // Distance between the current sample point and the planet center.
        float r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
        // Number density at the current sample point (divided by the number density
        // at the bottom of the atmosphere, yielding a dimensionless number).
        float y_i = GetProfileDensity(profile, r_i - ATMOSPHERE_BOTTOM_RADIUS);
        // Sample weight (from the trapezoidal rule).
        float weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5 : 1.0;
        result += y_i * weight_i * dx;
    }
    return result;
}



float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}


void GetRMuFromTransmittanceTextureUv(vec2 uv, out float r, out float mu) {
//    assert(uv.x >= 0.0 && uv.x <= 1.0);
//    assert(uv.y >= 0.0 && uv.y <= 1.0);
    float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
    float x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
    // Distance to top atmosphere boundary for a horizontal ray at ground level.
    float H = sqrt(ATMOSPHERE_TOP_RADIUS * ATMOSPHERE_TOP_RADIUS -
                    ATMOSPHERE_BOTTOM_RADIUS * ATMOSPHERE_BOTTOM_RADIUS);
    // Distance to the horizon, from which we can compute r:
    float rho = H * x_r;
    r = sqrt(rho * rho + ATMOSPHERE_BOTTOM_RADIUS * ATMOSPHERE_BOTTOM_RADIUS);
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
    // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
    // from which we can recover mu:
    float d_min = ATMOSPHERE_TOP_RADIUS - r;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    mu = d == 0.0 ? float(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = ClampCosine(mu);
}

vec3 ComputeTransmittanceToTopAtmosphereBoundary(float r, float mu) {
  return exp(-(
      RAYLEIGH_SCATTERING *
          ComputeOpticalLengthToTopAtmosphereBoundary(rayleigh_density, r, mu) +
      MIE_EXTINCTION *
          ComputeOpticalLengthToTopAtmosphereBoundary(mie_density, r, mu) + ABSORPTION_EXTINCTION *
          ComputeOpticalLengthToTopAtmosphereBoundary(absorption_density, r, mu)));
}


vec3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(vec2 uv) {
  const vec2 TRANSMITTANCE_TEXTURE_SIZE =
      vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
  float r;
  float mu;
  GetRMuFromTransmittanceTextureUv(uv, r, mu);
  return ComputeTransmittanceToTopAtmosphereBoundary(r, mu);
}


void main(){
	vec2 uv = (gl_FragCoord.xy + .5)/ViewportSize;
	vec3 transmittance = ComputeTransmittanceToTopAtmosphereBoundaryTexture(uv);

	fragColor = vec4(transmittance, 1);
}


