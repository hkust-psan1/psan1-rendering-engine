#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "common_structs.h"
#include "random.h"

using namespace optix;

// use offline rendering
// #define OFFLINE

// rtDeclareVariable(unsigned int, thread_index, attribute thread_index, );
rtDeclareVariable(uint2, thread_index, rtLaunchIndex, );
rtDeclareVariable(uint2, thread_dim, rtLaunchDim, );

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, tangent, attribute tangent, ); 
rtDeclareVariable(float3, bitangent, attribute bitangent, ); 
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, isect_dist, rtIntersectionDistance, );

rtDeclareVariable(float3, tile_size, , ); 
rtDeclareVariable(float3, tile_color_dark, , );
rtDeclareVariable(float3, tile_color_light, , );
rtDeclareVariable(float3, ambient_light_color, , );

rtDeclareVariable(int, is_emissive, , );

rtDeclareVariable(float3, k_emission, , );
rtDeclareVariable(float3, k_ambient, , );
rtDeclareVariable(float3, k_diffuse, , );
rtDeclareVariable(float3, k_specular, , );
rtDeclareVariable(float3, k_reflective, , );
rtDeclareVariable(int, ns, , );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, cutoff_color, , );
rtDeclareVariable(int, reflection_maxdepth, , );
rtDeclareVariable(float, importance_cutoff, , );

rtDeclareVariable(int, has_diffuse_map, , );
rtDeclareVariable(int, has_normal_map, , );
rtDeclareVariable(int, has_specular_map, , );

rtTextureSampler<float4, 2> kd_map;
rtTextureSampler<float4, 2> ks_map;
rtTextureSampler<float4, 2> normal_map;

// rtBuffer<BasicLight> lights;
rtBuffer<RectangleLight> area_lights;

struct PerRayData_radiance
{
	float3 result;
	float importance;
	int depth;
};

struct PerRayData_shadow
{
	float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

#define PI 3.1415926

/*  randomize a vector based on normal distribution, used for 
    normal vectors in glossy reflection and refraction */
static __device__ float3 randomizeVector(const float3& v, float amount, unsigned int& seed) {
	// two random number for normal distribution
	float rand1 = rnd(seed), rand2 = rnd(seed);

	// X and Y are normally distributed random numbers
	float X = sqrt(- 2 * log(rand1)) * cos(2 * PI * rand2) * amount;
	float Y = sqrt(- 2 * log(rand1)) * sin(2 * PI * rand2) * amount;

	// make a vector not parallel to v to find v's tangent and bitangent
	float3 u = v;
	u.x += 1;

	// tangent
	float3 e1 = cross(u, v);

	//bitangent
	float3 e2 = cross(v, e1);

	normalize(e1);
	normalize(e2);

	// randomize v in its tangent and bitangent direction
	float3 rand_vec = v + e1 * X + e2 * Y;
	normalize(rand_vec);

	return rand_vec;
}

static __device__ __inline__ float3 TraceRay(float3 origin, float3 direction, int depth, float importance )
{
	optix::Ray ray = optix::make_Ray( origin, direction, radiance_ray_type, 0.0f, RT_DEFAULT_MAX );
	PerRayData_radiance prd;
	prd.depth = depth;
	prd.importance = importance;

	rtTrace( top_object, ray, prd );
	return prd.result;
}

RT_PROGRAM void any_hit_shadow()
{
	if (!is_emissive) {
		prd_shadow.attenuation = make_float3(0.0f);
		rtTerminateRay();
	}
}

RT_PROGRAM void closest_hit_radiance()
{
	if (is_emissive) { // emissive object, just return the light color
		prd_radiance.result = make_float3(1, 1, 1);
		return;
	}

	// seed used for random number generation
	unsigned int seed = tea<16>(thread_index.y * thread_dim.x + thread_index.x, thread_index.y);;

	const float3 ray_dir = ray.direction; 
	const float3 uvw = texcoord;

	float3 kd;
	if (has_diffuse_map) { // has diffuse map, sample the texture
		kd = make_float3( tex2D( kd_map, uvw.x, uvw.y ) );
	} else {
		kd = k_diffuse;
	}

	float3 ks;
	if (has_specular_map) { // has specular map, sample the texture
		ks = make_float3( tex2D( ks_map, uvw.x, uvw.y ) );
	} else {
		ks = k_specular;
	}

	float3 kr = k_reflective;

	// Here tangent, bitangent and normal are attribute variables set in the ray-generation program,
	// transform them to the work space using the normal transformation matrix
	const float3 T = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, tangent)); // tangent	
	const float3 B = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, bitangent)); // bitangent	
	const float3 N = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal	

	// normal vector after considering normal map (if any)
	float3 normal;

	if (has_normal_map) { // has normal map, sample the texture
		const float3 k_normal = make_float3( tex2D( normal_map, uvw.x, uvw.y) );
		const float3 coeff = k_normal * 2 - make_float3(1, 1, 1); // transform from RGB to normal
		normal = T * coeff.x + B * coeff.y + N * coeff.z;
	} else {
		normal = N;
	}

	// first hit point
	const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);

	const int depth = prd_radiance.depth;

	// starting from the ambient color
	float3 result = kd * ambient_light_color;
		
	for (int i = 0; i < area_lights.size(); i++) {
		RectangleLight light = area_lights[i];

		// according to whether offline rendering is activated, use different number of samples to
		// achieve soft or hard shadow
#ifdef OFFLINE
		const int numShadowSamples = 10;
#else
		const int numShadowSamples = 1;
#endif

		for (int j = 0; j < numShadowSamples; j++) {
#ifdef OFFLINE
			float2 random_pair = make_float2(rnd(seed), rnd(seed));
			// randomly sample points in the area light source
			float3 sampledPos = light.pos + random_pair.x * light.r1 + random_pair.y * light.r2;
#else
			float3 sampledPos = light.pos;
#endif

			float Ldist = length(sampledPos - fhp);

			float3 L = normalize(sampledPos - fhp);
			float3 H = normalize(L - ray.direction);

			float nDl = dot(normal, L);

			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1);

			if(nDl > 0) {
				optix::Ray shadow_ray = optix::make_Ray( fhp, L, shadow_ray_type, scene_epsilon, Ldist );
				rtTrace(top_shadower, shadow_ray, shadow_prd);
				result += light.color * shadow_prd.attenuation 
					* (kd * nDl + ks * max(pow(dot(H, normal), ns), .0f)) / numShadowSamples;
			}
		}
	}

	// beer attenuation for reflection
	const float beer_attenuation = 1.0f;

	float3 refl_color = make_float3(0, 0, 0);
	float reflection = 1.0f;

	if (depth < min(reflection_maxdepth, max_depth)) {
		// reflection direction
		const float3 r = reflect(ray_dir, normal);
		float importance = prd_radiance.importance * reflection * optix::luminance(kr * beer_attenuation);

		// number of samples to take for each reflection
#ifdef OFFLINE
		const int numGlossySample = 10;
		
		if (importance > importance_cutoff) {
			for (int i = 0; i < numGlossySample; i++) {
				float3 randomizedRefl = randomizeVector(r, 0.1, seed);
				refl_color += TraceRay(fhp, randomizedRefl, depth + 1, importance / float(numGlossySample));
			}
		}
		refl_color /= numGlossySample;
#else
		refl_color = TraceRay(fhp, r, depth + 1, importance);
#endif
	}

	result += kd * reflection * kr * refl_color;
	
	prd_radiance.result = result;
}

