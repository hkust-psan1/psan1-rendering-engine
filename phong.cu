#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "common_structs.h"
#include "node_shading_system.h"
#include "random.h"

using namespace optix;

rtDeclareVariable(uint2, thread_index, rtLaunchIndex, );
rtDeclareVariable(uint2, thread_dim, rtLaunchDim, );

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

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

rtDeclareVariable(float3, ambient_light_color, , );

rtDeclareVariable(int, is_emissive, , );
rtDeclareVariable(float3, k_emission, , );

rtDeclareVariable(float3, k_diffuse, , );

rtDeclareVariable(float3, k_specular, , );

rtDeclareVariable(float3, k_reflective, , );
rtDeclareVariable(float, glossiness, , );

rtDeclareVariable(float3, alpha, , );
rtDeclareVariable(float, IOR, , ) = 1.4;

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, cutoff_color, , );

rtDeclareVariable(float, importance_cutoff, , );

rtDeclareVariable(float3, emissive_color, ,);
rtDeclareVariable(float, emissive_weight, ,);

rtDeclareVariable(float3, diffuse_color, ,);
rtDeclareVariable(float, diffuse_weight, ,);

rtDeclareVariable(float3, reflective_color, ,);
rtDeclareVariable(float, reflective_weight, ,);

rtDeclareVariable(float3, refractive_color, ,);
rtDeclareVariable(float, refractive_weight, ,);

rtDeclareVariable(float3, subsurf_scatter_color, , );
rtDeclareVariable(float, subsurf_scatter_weight, , );
rtDeclareVariable(float, subsurf_att, , ) = 0.1;

rtDeclareVariable(int, has_diffuse_map, , );
rtDeclareVariable(int, has_normal_map, , );
rtDeclareVariable(int, has_specular_map, , );

rtDeclareVariable(int, anisotropic, , );

rtTextureSampler<float4, 2> kd_map = NULL;
rtTextureSampler<float4, 2> ks_map = NULL;
rtTextureSampler<float4, 2> normal_map = NULL;

rtDeclareVariable(int, soft_shadow_on, ,) = false;
rtDeclareVariable(int, glossy_on, ,) = false;
rtDeclareVariable(int, gi_on, ,) = false;

rtDeclareVariable(unsigned int, frame_number, , );

rtBuffer<RectangleLight> area_lights;
rtBuffer<SpotLight> spot_lights;
rtBuffer<DirectionalLight> directional_lights;

struct PerRayData_radiance {
	float3 result;
	float importance;
	int depth;
	int ss;
};

struct PerRayData_shadow {
	float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

static __device__ __inline__ float3 exp( const float3& x ) {
	return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

/* vector can be ignored */
static __device__ __inline__ bool float_vec_zero(float3 v) {
	return length(v) < scene_epsilon;
}

/*	calculate fresnel reflection */
static __device__ __inline__ float3 schlick(float nDi, const float3& rgb) {
	float r = fresnel_schlick(nDi, 5, rgb.x, 1);
	float g = fresnel_schlick(nDi, 5, rgb.y, 1);
	float b = fresnel_schlick(nDi, 5, rgb.z, 1);
	return make_float3(r, g, b);
}

static __device__ __inline__ float float3_sum(float3 v) {
	return v.x + v.y + v.z;
}

/*  randomize a vector based on normal distribution, used for 
    normal vectors in glossy reflection and refraction */
static __device__ float3 randomize_vector(const float3& v, float amount, unsigned int& seed) {
	// two random number for normal distribution
	float rand1 = rnd(seed), rand2 = rnd(seed);

	// X and Y are normally distributed random numbers
	float X = sqrt(- 2 * log(rand1)) * cos(2 * 3.14159 * rand2) * amount / 5;
	float Y = sqrt(- 2 * log(rand1)) * sin(2 * 3.14159 * rand2) * amount / 5;

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

static __device__ __inline__ void createONB( const optix::float3& n, optix::float3& U, optix::float3& V) {
	U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
	if ( dot(U, U) < 1.e-3f ) {
		U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
	}
	U = normalize( U );
	V = cross( n, U );
}

static __device__ __inline__ float3 TraceRay(float3 origin, float3 direction, int depth, 
											 float importance, int ss = false) {
	optix::Ray ray = optix::make_Ray( origin, direction, radiance_ray_type, 0.0f, RT_DEFAULT_MAX );
	PerRayData_radiance prd;
	prd.result = make_float3(0);
	prd.depth = depth;
	prd.importance = importance;
	prd.ss = ss;

	rtTrace( top_object, ray, prd );
	return prd.result;
}

RT_PROGRAM void any_hit_shadow()
{
	if (!is_emissive) { // the hit object is not emissive
		prd_shadow.attenuation *= alpha;

		if (float_vec_zero(prd_shadow.attenuation)) {
			rtTerminateRay();
		} else {
			rtIgnoreIntersection();
		}
	}
	/*
	float3 world_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float nDi = fabs(dot(world_normal, ray.direction));

  prd_shadow.attenuation *= 1-fresnel_schlick(nDi, 5, 1-make_float3(1), make_float3(1));
  if(optix::luminance(prd_shadow.attenuation) < importance_cutoff)
    rtTerminateRay();
  else
    rtIgnoreIntersection();
	*/
}

RT_PROGRAM void closest_hit_radiance()
{
	// seed used for random number generation
	unsigned int seed = tea<16>(thread_index.y * thread_dim.x + thread_index.x, thread_index.y + frame_number);

	float3 kd;
	if (has_diffuse_map) { // has diffuse map, sample the texture
		kd = make_float3( tex2D( kd_map, texcoord.x, texcoord.y ) );
	} else {
		kd = k_diffuse;
	}

	// front hit point
	float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);

	// back hit point
	float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);

	// a trick to replace the diffuse factor with ss color
	if (prd_radiance.ss) {
		kd = subsurf_scatter_color;
		fhp = bhp;
	}

	// Here tangent, bitangent and normal are attribute variables set in the ray-generation program,
	// transform them to the work space using the normal transformation matrix
	const float3 T = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, tangent)); // tangent	
	const float3 B = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, bitangent)); // bitangent	
	const float3 N = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal	

	// normal vector after considering normal map (if any)
	float3 normal;

	if (has_normal_map) { // has normal map, sample the texture
		const float3 k_normal = make_float3( tex2D( normal_map, texcoord.x, texcoord.y) );
		const float3 coeff = k_normal * 2 - make_float3(1); // transform from RGB to normal
		normal = T * coeff.x + B * coeff.y + N * coeff.z;
	} else {
		normal = N;
	}

	prd_radiance.result = k_emission;

	float3 diffuse_result = make_float3(0);
		
	for (int i = 0; i < area_lights.size(); i++) {
		RectangleLight light = area_lights[i];
		float3 sampledPos;

		if (soft_shadow_on) {
			sampledPos = light.pos + rnd(seed) * light.r1 + rnd(seed) * light.r2;
		} else {
			sampledPos = light.pos + 0.5 * light.r1 + 0.5 * light.r2;
		}

		float Ldist = length(sampledPos - fhp);

		float distance_attenuation = 1 / (1 + light.attenuation_coeff * Ldist 
			+ light.attenuation_coeff * Ldist * Ldist);

		float3 L = normalize(sampledPos - fhp);
		float3 H = normalize(L - ray.direction);

		float nDl = dot(normal, L);

		// cast shadow ray
		PerRayData_shadow shadow_prd;
		shadow_prd.attenuation = make_float3(1);

		if(nDl > 0) {
			optix::Ray shadow_ray = optix::make_Ray( fhp, L, shadow_ray_type, scene_epsilon, Ldist );
			rtTrace(top_shadower, shadow_ray, shadow_prd);

			diffuse_result += light.intensity * light.color * distance_attenuation * shadow_prd.attenuation 
				* (kd * nDl);
		}
	}

	for (int i = 0; i < spot_lights.size(); i++) {
		SpotLight light = spot_lights[i];

		float Ldist = length(light.pos - fhp);

		float distance_attenuation = 1 / (1 + light.attenuation_coeff * Ldist 
			+ light.attenuation_coeff * Ldist * Ldist);

		float3 L = normalize(light.pos - fhp);
		float3 H = normalize(L - ray.direction);

		float nDl = dot(normal, L);

		// different in direction
		float dir_diff = dot(normalize(light.direction), -L);

		// intensity of spotlight drops as the angle increases
		float angle_attenuation = pow(dir_diff, light.dropoff_rate);

		PerRayData_shadow shadow_prd;
		shadow_prd.attenuation = make_float3(1);

		if(nDl > 0 && dir_diff > cos(light.angle)) {
			optix::Ray shadow_ray = optix::make_Ray( fhp, L, shadow_ray_type, scene_epsilon, Ldist );
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			diffuse_result += angle_attenuation * light.intensity * light.color 
				* distance_attenuation * shadow_prd.attenuation 
				* (kd * nDl);
		}
	}

	for (int i = 0; i < directional_lights.size(); i++) {
		DirectionalLight light = directional_lights[i];

		float3 L = normalize(- light.direction);
		float3 H = normalize(L - ray.direction);

		float nDl = dot(normal, L);

		PerRayData_shadow shadow_prd;
		shadow_prd.attenuation = make_float3(1);

		if (nDl > 0) {
			optix::Ray shadow_ray = optix::make_Ray( fhp, L, shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX );
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			diffuse_result += light.intensity * light.color * shadow_prd.attenuation
				* (kd * nDl);
		}
	}


	const int new_depth = prd_radiance.depth + 1;

	if (new_depth > max_depth) { // max depth exceeded, stop further tracing
		return;
	}

	float diffuse_amount = length(kd);
	float reflective_amount = length(k_reflective);
	float refractive_amount = length(alpha);
	float total_amount = diffuse_amount + reflective_amount + refractive_amount;

	/* global illunimation */
	float diffuse_importance = diffuse_amount / total_amount * prd_radiance.importance;

	if (gi_on && diffuse_importance > importance_cutoff) {
		// randomly sample a vector in the hemisphere 
		float3 p;
		cosine_sample_hemisphere(rnd(seed), rnd(seed), p);

		// create two vectors perpendicular to the normal
		float3 v1, v2;
		createONB(normal, v1, v2);

		float3 random_ray_direction = v1 * p.x + v2 * p.y + normal * p.z;

		// reduce importance more for indirect lighting to speed up rendering (divide by 2)
		diffuse_result += 0.8 * diffuse_importance * kd 
			* TraceRay(fhp, random_ray_direction, new_depth, diffuse_importance / 2);
	}
	
	// stop here if the ray is from subsurface scattering
	if (prd_radiance.ss) {
		float distance = length(fhp - ray.origin);
		float attenuation = 1 / (1 + subsurf_att * distance + subsurf_att * distance * distance);
		prd_radiance.result = diffuse_result * attenuation;
		return;
	}

	/* refraction */
	float fresnel_reflection = 1;
	float refractive_importance = refractive_amount / total_amount * prd_radiance.importance;

	if (!float_vec_zero(alpha) && refractive_importance > importance_cutoff) {

		float3 transmission_direction;
		if (refract(transmission_direction, ray.direction, normal, IOR)) {
			// check whether it is internal or external refraction
			float cos_theta = dot(ray.direction, normal);
			if (cos_theta < 0) { // external
				cos_theta = - cos_theta;
			} else { // internal
				cos_theta = dot(transmission_direction, normal);
			}

			fresnel_reflection = fresnel_schlick(cos_theta, 3, 0.1, 1);

			float importance = prd_radiance.importance * (1.0f - fresnel_reflection) * luminance(alpha);

			if (importance > importance_cutoff) {
				if (glossy_on) {
					float3 randomizedRefr = randomize_vector(transmission_direction, glossiness, seed);
					prd_radiance.result += (1 - fresnel_reflection) * alpha
						* TraceRay(bhp, randomizedRefr, new_depth, importance);
				} else {
					prd_radiance.result += (1 - fresnel_reflection) * alpha 
						* TraceRay(bhp, transmission_direction, new_depth, importance);
				}
			}
		}
	}

	/* reflection */
	float3 glossy_result = make_float3(0);
	// float reflective_importance = reflective_amount / total_amount * prd_radiance.importance;

	if (!float_vec_zero(k_reflective)) {
		// reflection direction
		const float3 refl = reflect(ray.direction, normal);

		float importance = prd_radiance.importance * fresnel_reflection * luminance(k_reflective);

		if (importance > importance_cutoff) {
			if (glossy_on) {
				float3 randomizedRefl;
				if (anisotropic) {
					if (rnd(seed) > 0.0) {
						randomizedRefl = normalize(refl + B * (rnd(seed) - 0.5) * 2);
					} else {
						randomizedRefl = randomize_vector(refl, glossiness, seed);
					}
				} else {
					randomizedRefl = randomize_vector(refl, glossiness, seed);
				}
				glossy_result += fresnel_reflection * k_reflective
					* TraceRay(fhp, randomizedRefl, new_depth, importance);
			} else {
				glossy_result += fresnel_reflection * k_reflective 
					* TraceRay(fhp, refl, new_depth, importance);
			}
		}
	}
	
	if (has_specular_map) {
		float3 ks = make_float3(tex2D(ks_map, texcoord.x, texcoord.y));
		float fac = (ks.x + ks.y + ks.z ) / 3.0;
		prd_radiance.result += (diffuse_result * (1 - fac) + glossy_result * fac) * 2;
	} else {
		prd_radiance.result += diffuse_result + glossy_result;
	}

	/* subsurface scattering */
	if (!float_vec_zero(subsurf_scatter_color)) {
		// randomly sample a vector in the hemisphere 
		float3 p;
		cosine_sample_hemisphere(rnd(seed), rnd(seed), p);

		// create two vectors perpendicular to the normal
		float3 v1, v2;
		createONB(normal, v1, v2);

		float3 random_ray_direction = - (v1 * p.x + v2 * p.y + normal * p.z);

		const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);

		prd_radiance.result += subsurf_scatter_color
			* TraceRay(bhp, random_ray_direction, new_depth, prd_radiance.importance * 0.8, true);
	}
}