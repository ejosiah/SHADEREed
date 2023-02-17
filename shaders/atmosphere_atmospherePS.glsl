#version 460


#define TWO_PI 6.283185307179586476925286766559
#define PI 3.1415926535897932384626433832795
#define FOUR_PI 12.566370614359172953850573533118
#define RAD 0.01745329251994329576923690768489
#define SPEED 0.1
#define EARTH_RADIUS 6.371
#define ATMOSPHERE_THICKNESS (0.1 * EARTH_RADIUS)
#define ATMOSPHERE_TOP (ATMOSPHERE_THICKNESS + EARTH_RADIUS)
#define H (0.25 * ATMOSPHERE_THICKNESS)
#define KR 0.0025
#define FOUR_PI_KR (FOUR_PI * KR)
#define KM 0.0010
#define FOUR_PI_KM (FOUR_PI * KM)
#define SUN_INTENSITY 20.0
#define LAMBDA vec3(0.650, 0.570, 0.475)
#define  LAMBDA_4 pow(LAMBDA, vec3(4))
#define SAMPLE_COUNT 5

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

uniform vec3 CameraPosition3;
uniform vec3 CameraDirection3;
uniform float time;
uniform mat4 View;


vec4 paintAtmosphere(Ray ray);

vec4 paintEarth(Ray ray);


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
	
	vec3 color = mix(vec3(1), vec3(0, 0.3, 0.8), fs_in.uv.y);
	vec4 earth = paintEarth(ray);

	color = mix(color,  earth.rgb, earth.a);
	
	
	fragColor.rgb = color;
	fragColor.rgb = pow(fragColor.rgb, vec3(0.45));
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
	t.y = max(tMin, tMin);
	
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


vec4 paintEarth(Ray ray){
	vec4 color = vec4(0);
	
	Sphere sphere = Sphere(vec3(0), EARTH_RADIUS);
	vec2 t = vec2(1000000, -1000000);
	if(test(ray, sphere, t)){
		color.a = 1;
		vec3 p = ray.origin + ray.direction * t.x;
		vec3 x = p - sphere.center;
		
		vec4 axis = axisAngle(rotationAxis(), time * SPEED);
		
		x = rotatePoint(axis, x);
		
		float theta = atan(x.z, x.x);
        float phi = acos(x.y/sphere.radius);
		vec2 uv = vec2(theta/TWO_PI + .5, phi/PI);	
		uv *= -1;	
		vec3 N = normalize(x);
		vec3 sN = -1 + 2 * texture(normalMap, uv).xyz;
		
		
        vec3 T = vec3(-sin(theta) * sin(phi), 0, cos(theta) * sin(phi));
        vec3 B = vec3(cos(theta) * cos(phi), -sin(phi), sin(theta) * cos(phi));		
		vec3 L = -normalize(rotatePoint(axis, ray.direction));
		
		mat3 objToWorldMaxtrix = mat3(T, B, N);
		N = objToWorldMaxtrix * sN;	

        vec3 albedo = texture(colorMap, uv).rgb;
		color.rgb = albedo * dot(N, L) / PI;
	}
	

	Sphere atmosphere = Sphere(vec3(0), ATMOSPHERE_TOP);
	vec2 t1 = vec2(0);
	if(test(ray, atmosphere, t1)){
		vec3 albedo = vec3(.2, 0, 0);
		if(color.a != 0 && t1.x < t.x){
			color.rgb = mix(color.rgb, albedo, 0.5);
		}else{
			color = vec4(albedo, 0.5);
		}
	}
	
	return color;
	
}

vec4 paintAtmosphere(Ray ray){
	vec4 color = vec4(0);
	Sphere sphere = Sphere(vec3(0), ATMOSPHERE_TOP);
	
	vec2 t;
	if(test(ray, sphere, t)){
		color = vec4(1, 0, 0, 0.5);
	}
	
	return color;
}