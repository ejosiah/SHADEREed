#version 460


#define TWO_PI 6.283185307179586476925286766559
#define PI 3.1415926535897932384626433832795
#define FOUR_PI 12.566370614359172953850573533118
#define RAD 0.01745329251994329576923690768489
#define SPEED 0.1
#define EARTH_RADIUS 6.371
#define ATMOSPHERE_THICKNESS (0.1 * EARTH_RADIUS)
#define ATMOSPHERE_TOP (ATMOSPHERE_THICKNESS + EARTH_RADIUS)
#define AVERAGE_DENSITY_HEIGHT 0.25
#define H (AVERAGE_DENSITY_HEIGHT * ATMOSPHERE_THICKNESS)
#define KR 0.0025
#define RAYLEIGH_SCALE_DEPTH 0.25
#define MIE_SCALE_DEPTH 0.1
#define KR_FOUR_PI (FOUR_PI * KR)
#define KM 0.0010
#define KM_FOUR_PI (FOUR_PI * KM)
#define SUN_INTENSITY 20.0
#define LAMBDA vec3(0.650, 0.570, 0.475)
#define LAMBDA_4 pow(LAMBDA, vec3(4))
#define INVERSE_LAMBDA_4 (1.0/LAMBDA_4)
#define SAMPLE_COUNT 10

const float g = -0.99;
const float g2 = g * g;


struct Sphere{
	vec3 center;
	float radius;
};

struct Ray{
	vec3 origin;
	vec3 direction;
};

bool test(in Ray ray, in Sphere sphere, out vec2 t);

vec4 axisAngle(vec3 axis, float angle);

vec3 rotatePoint(vec4 q, vec3 v);


vec3 rotationAxis();


uniform sampler2D colorMap;
uniform sampler2D normalMap;
uniform sampler2D starMap;

uniform vec3 CameraPosition3;
uniform vec3 CameraDirection3;
uniform float time;
uniform mat4 View;
uniform vec3 SunDirection;
uniform float exposure;
uniform bool atmospherOn;


vec3 sunPosition = vec3(1000);
vec3 sunDirection = normalize(sunPosition);

vec4 paintEarth(Ray ray, out float depth);

vec4 paintSun(Ray ray, out float depth);

vec3 hdr(vec3 color);


layout(location = 0) in struct {
	vec3 viewDir;
	vec2 uv;
} fs_in;

layout(location = 0) out vec4 fragColor;

vec3 radiance = vec3(10);

void main(){
	Ray ray;
	ray.origin = (inverse(View) * vec4(0, 0, 0, 1)).xyz;
	ray.direction = normalize(fs_in.viewDir);
	
	vec4 axis = axisAngle(rotationAxis(), time * SPEED);
//	sunPosition = rotatePoint(-axis, sunPosition);
	//sunPosition = normalize(SunDirection) * 80;
	
	vec3 color = mix(vec3(1), vec3(0, 0.3, 0.8), fs_in.uv.y);
	color = vec3(0);
	vec3 stars = texture(starMap, fs_in.uv).rgb;
	
	vec3 sunD = normalize(sunPosition - ray.origin);
	float sun = max(0, dot(ray.direction, sunD));
	sun = smoothstep(0.9995, 0.9995, sun);

	float ed;
	vec4 earth = paintEarth(ray, ed);

	color = mix(color,  earth.rgb, earth.a);
	
	if(ed < 0) color = stars;
	
	float sd;
	vec3 sunColor = paintSun(ray, sd).rgb;

	fragColor.rgb =  color + sunColor;
	
	
	fragColor.rgb = hdr(fragColor.rgb);

	fragColor.rgb = pow(fragColor.rgb, vec3(0.45));
}

vec3 hdr(vec3 color){
	return 1.0 - exp(-max(exposure, 0.1) * color);
	//return color/(color + 1);
}

bool test(in Ray ray, in Sphere sphere, out vec2 t){
	vec3 d = ray.direction;
	vec3 m = ray.origin - sphere.center;
	float r = sphere.radius;
	
	float a = dot(d, d);
	float b = dot(m, d);
	float c = dot(m, m) - r * r;
	
	if(c > 0 && b > 0) return false;
	
	float discr = b * b - a * c;
	if(discr < 0) return false;
	
	float sqrtDiscr = sqrt(discr);
	float tMin = (-b - sqrtDiscr)/a;
	float tMax = (-b + sqrtDiscr)/a;
	t.x = max(0, min(tMin, tMax));
	t.y = max(tMin, tMax);
	
	return true;
}

vec4 axisAngle(vec3 axis, float angle){
    float halfAngle = angle * 0.5;
    float w = cos(halfAngle);
    vec3 xyz = axis * sin(halfAngle);

    return vec4(xyz, w);
}

// Optimized point rotation using quaternion
// Source: https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
vec3 rotatePoint(vec4 q, vec3 v) {
    const vec3 qAxis = vec3(q.x, q.y, q.z);
    return 2.0f * dot(qAxis, v) * qAxis + (q.w * q.w - dot(qAxis, qAxis)) * v + 2.0f * q.w * cross(qAxis, v);
}


vec3 rotationAxis(){
	vec4 axis = axisAngle(vec3(0, 0, 1), 23.5 * RAD);
	return rotatePoint(axis, vec3(0, -1, 0));
}

float scale(float cos0){
	float x = 1.0 - cos0;
	return AVERAGE_DENSITY_HEIGHT * exp(-0.00287 + x*(0.459 + x*(3.83 + x*(-6.80 + x*5.25))));
}

float phaseFunc(float cos0, float g){
	float g2 = g*g;
	return 1.5 * ((1.0 - g2) / (2.0 + g2)) * (1.0 + cos0*cos0) / pow(1.0 + g2 - 2.0*g*cos0, 1.5);
}

vec4 rayMarchAtmosphere(Ray ray, vec4 earthColor, float earth_t){
	//return earthColor;
	
	Sphere atmosphere = Sphere(vec3(0), ATMOSPHERE_TOP);
	vec2 t = vec2(0);
	vec3 scatterColor = vec3(0);
	
	if(test(ray, atmosphere, t)){

		float tNear = t.x;
		float tFar =  min(earth_t, t.y);
		float stepSize = SAMPLE_COUNT/(tFar - tNear);
 		
 		vec3 entryPoint = ray.origin + ray.direction * tNear;

 		
 		float startAngle = dot(ray.direction, entryPoint)/ATMOSPHERE_TOP;
 		float startDepth = exp(-1.0/AVERAGE_DENSITY_HEIGHT);
 		float startOffset = startDepth * scale(startAngle);
 		startOffset = 0;
		
		
		vec3 stepDir = ray.direction * stepSize;
		vec3 samplePoint = entryPoint + stepDir;
		for(int i = 0; i < SAMPLE_COUNT; i++){
			float sampleHeight = length(samplePoint);
			float h = (sampleHeight - EARTH_RADIUS);
			float depth = exp(-h/H);
			float sunAngle = dot(sunDirection, samplePoint)/sampleHeight;
			float cameraAngle = dot(ray.direction, samplePoint)/sampleHeight;
			
			float scatter = (startOffset + depth*(scale(sunAngle) - scale(cameraAngle)));
			vec3 attenuate = exp(-scatter * (INVERSE_LAMBDA_4 * KR_FOUR_PI + KM_FOUR_PI));
			scatterColor += attenuate * depth * stepSize;
			samplePoint += stepDir;

		}
	}
	

	vec3 color = scatterColor * SUN_INTENSITY * KM_FOUR_PI;
	vec3 secondaryColor = color * SUN_INTENSITY * KR_FOUR_PI * INVERSE_LAMBDA_4;
	
	float cos0 = dot(sunDirection, -ray.direction);
	float miePhase = phaseFunc(cos0, g);
	
	vec4 aColor = vec4(0);
	aColor.rgb = color + miePhase * secondaryColor;
	aColor.a = aColor.b;
	
	if(earthColor.a != 0){
		aColor.rgb = mix(earthColor.rgb, aColor.rgb, aColor.a);
	}
	
	return aColor;
}

vec4 paintEarth(Ray ray, out float depth){
	vec4 color = vec4(0);
	
	Sphere sphere = Sphere(vec3(0), EARTH_RADIUS);
	vec2 t = vec2(0);
	if(test(ray, sphere, t)){
		color.a = 1;
		vec3 p = ray.origin + ray.direction * t.x;
		vec3 x = p - sphere.center;
		
		vec4 axis = axisAngle(rotationAxis(), time * SPEED);
		vec3 N = normalize(x);
		x = rotatePoint(axis, x);
	
		
		float theta = atan(x.z, x.x);
        float phi = acos(x.y/sphere.radius);
		vec2 uv = vec2(theta/TWO_PI + .5, phi/PI);	
		uv *= -1;	

		vec3 sN = -1 + 2 * texture(normalMap, uv).xyz;
		
		
        vec3 T = vec3(-sin(theta) * sin(phi), 0, cos(theta) * sin(phi));
        vec3 B = vec3(cos(theta) * cos(phi), -sin(phi), sin(theta) * cos(phi));		
		//vec3 L = -normalize(rotatePoint(axis, ray.direction));
		vec3 L = sunDirection;
		
		mat3 objToWorldMaxtrix = mat3(T, B, N);
		N = objToWorldMaxtrix * sN;	

        vec3 albedo = texture(colorMap, uv).rgb;
		color.rgb = SUN_INTENSITY * albedo * dot(N, L) / PI;
	}
	

	
	Sphere atmosphere = Sphere(vec3(0), ATMOSPHERE_TOP);
	vec2 t1 = vec2(0);
	if(atmospherOn && test(ray, atmosphere, t1)){
		color = rayMarchAtmosphere(ray, color, t.x);
	}
	depth = max(t.x, t1.x);
	return color;
	
}

vec4 paintSun(Ray ray, out float depth){
	vec4 color = vec4(0);
	Sphere sphere = Sphere(sunPosition, EARTH_RADIUS * 10);
	
	vec2 t;
	if(test(ray, sphere, t)){
		color = vec4(1, 0.55, 0.15, 1) * SUN_INTENSITY;
	}
	depth = t.x;
	return color;
}