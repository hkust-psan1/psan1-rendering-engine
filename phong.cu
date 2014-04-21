#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "common_structs.h"
#include "random.h"

using namespace optix;

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

rtDeclareVariable(float3, ambient_light_color, , );

rtDeclareVariable(int, is_emissive, , );
rtDeclareVariable(float3, k_emission, , );

rtDeclareVariable(float3, k_diffuse, , );

rtDeclareVariable(float3, k_specular, , );
rtDeclareVariable(int, ns, , );

rtDeclareVariable(float3, k_reflective, , );
rtDeclareVariable(float, glossiness, , );

rtDeclareVariable(float3, alpha, , );
rtDeclareVariable(float, IOR, , ) = 1.4;

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, cutoff_color, , );

rtDeclareVariable(float, importance_cutoff, , );

rtDeclareVariable(float, linear_attenuation_factor, ,) = 0.2;
rtDeclareVariable(float, quadratic_attenuation_factor, ,) = 0.2;

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

rtDeclareVariable(int, has_diffuse_map, , );
rtDeclareVariable(int, has_normal_map, , );
rtDeclareVariable(int, has_specular_map, , );

rtTextureSampler<float4, 2> kd_map = NULL;
rtTextureSampler<float4, 2> ks_map = NULL;
rtTextureSampler<float4, 2> normal_map = NULL;

rtDeclareVariable(int, soft_shadow_on, ,) = false;
rtDeclareVariable(int, glossy_on, ,) = false;
rtDeclareVariable(int, gi_on, ,) = false;

rtDeclareVariable(Shader, my_shader, ,);

rtDeclareVariable(unsigned int, frame_number, , );

rtBuffer<RectangleLight> area_lights;
rtBuffer<SpotLight> spot_lights;
rtBuffer<DirectionalLight> directional_lights;

rtBuffer<Shader> shaders;

struct PerRayData_radiance {
	float3 result;
	float importance;
	int depth;
};

struct PerRayData_shadow {
	float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

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

static __device__ __inline__ float3 TraceRay(float3 origin, float3 direction, int depth, float importance) {
	optix::Ray ray = optix::make_Ray( origin, direction, radiance_ray_type, 0.0f, RT_DEFAULT_MAX );
	PerRayData_radiance prd;
	prd.result = make_float3(0);
	prd.depth = depth;
	prd.importance = importance;

	rtTrace( top_object, ray, prd );
	return prd.result;
}

RT_PROGRAM void any_hit_shadow()
{
	if (!is_emissive) { // the hit object is not emissive
		prd_shadow.attenuation *= alpha;
		if (float_vec_zero(prd_shadow.attenuation)) {
			rtTerminateRay();
		}
	}
}

RT_PROGRAM void closest_hit_radiance_pt()
{
	if (is_emissive) { // emissive object, just return the light color
		prd_radiance.result = k_emission;
		return;
	}

	// seed used for random number generation
	unsigned int seed = tea<16>(thread_index.y * thread_dim.x + thread_index.x, thread_index.y + frame_number);

	float3 kd;
	if (has_diffuse_map) { // has diffuse map, sample the texture
		kd = make_float3( tex2D( kd_map, texcoord.x, texcoord.y ) );
	} else {
		kd = diffuse_color;
	}

	float r_weight;
	if (has_specular_map) { // has specular map, sample the texture
		float3 r_vec = make_float3(tex2D(ks_map, texcoord.x, texcoord.y)); // sample the texture
		r_weight = (r_vec.x + r_vec.y + r_vec.z) / 3;
	} else {
		r_weight = refractive_weight;
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

	// front hit point
	const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);

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

		float distance_attenuation = 1 / (1 + linear_attenuation_factor * Ldist 
			+ quadratic_attenuation_factor * Ldist * Ldist);

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

		float distance_attenuation = 1 / (1 + linear_attenuation_factor * Ldist 
			+ quadratic_attenuation_factor * Ldist * Ldist);

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

	/* reflection */
	float reflective_importance = reflective_amount / total_amount * prd_radiance.importance;

	if (!float_vec_zero(k_reflective) && reflective_importance > importance_cutoff) {

		// reflection direction
		const float3 refl = reflect(ray.direction, normal);

		// number of samples to take for each reflection
		if (glossy_on) {
			// const int num_glossy_sample = 10;
			const int num_glossy_sample = 1;
			
			for (int i = 0; i < num_glossy_sample; i++) {
				float3 randomizedRefl = randomize_vector(refl, glossiness, seed);
				prd_radiance.result += reflective_importance * k_reflective
					* TraceRay(fhp, randomizedRefl, new_depth, reflective_importance) / num_glossy_sample;
			}
		} else {
			prd_radiance.result += k_reflective 
				* TraceRay(fhp, refl, new_depth, reflective_importance);
		}
	}

	/* refraction */
	float refractive_importance = refractive_amount / total_amount * prd_radiance.importance;

	if (!float_vec_zero(alpha) && refractive_importance > importance_cutoff) {
		// back hit point
		const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);

		float3 transmission_direction;
		if (refract(transmission_direction, ray.direction, normal, IOR)) {
			// check whether it is internal or external refraction
			/*
			float cos_theta = dot(ray.direction, normal);
			if (cos_theta < 0) { // external
				cos_theta = - cos_theta;
			} else { // internal
				cos_theta = dot(transmission_direction, normal);
			}

			float reflection = fresnel_schlick(cos_theta, 3, 0.1, 1);

			float importance = prd_radiance.importance * (1.0f-reflection) * luminance( alpha );
			*/

			prd_radiance.result += alpha 
				* TraceRay(bhp, ray.direction, new_depth, refractive_importance / 2);
		}
	}
}

RT_PROGRAM void closest_hit_radiance()
{
	if (is_emissive) { // emissive object, just return the light color
		prd_radiance.result = k_emission;
		return;
	}

	// seed used for random number generation
	unsigned int seed = tea<16>(thread_index.y * thread_dim.x + thread_index.x, thread_index.y + frame_number);

	float3 kd;
	if (has_diffuse_map) { // has diffuse map, sample the texture
		kd = make_float3( tex2D( kd_map, texcoord.x, texcoord.y ) );
	} else {
		kd = k_diffuse;
	}

	float3 ks;
	if (has_specular_map) { // has specular map, sample the texture
		ks = make_float3( tex2D( ks_map, texcoord.x, texcoord.y ) );
	} else {
		ks = k_specular;
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

	// front hit point
	const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);

	// starting from the ambient color
	prd_radiance.result = kd * ambient_light_color;
		
	for (int i = 0; i < area_lights.size(); i++) {
		RectangleLight light = area_lights[i];
		float3 sampledPos;

		if (soft_shadow_on) {
			sampledPos = light.pos + rnd(seed) * light.r1 + rnd(seed) * light.r2;
		} else {
			sampledPos = light.pos + 0.5 * light.r1 + 0.5 * light.r2;
		}

		float Ldist = length(sampledPos - fhp);

		float distance_attenuation = 1 / (1 + linear_attenuation_factor * Ldist 
			+ quadratic_attenuation_factor * Ldist * Ldist);

		float3 L = normalize(sampledPos - fhp);
		float3 H = normalize(L - ray.direction);

		float nDl = dot(normal, L);

		// cast shadow ray
		PerRayData_shadow shadow_prd;
		shadow_prd.attenuation = make_float3(1);

		if(nDl > 0) {
			optix::Ray shadow_ray = optix::make_Ray( fhp, L, shadow_ray_type, scene_epsilon, Ldist );
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			/*
			prd_radiance.result += light.intensity * light.color * distance_attenuation * shadow_prd.attenuation 
				* (kd * nDl + ks * max(pow(dot(H, normal), ns), .0f));
				*/
			prd_radiance.result += light.intensity * light.color * distance_attenuation * shadow_prd.attenuation 
				* (kd * nDl);
		}
	}

	for (int i = 0; i < spot_lights.size(); i++) {
		SpotLight light = spot_lights[i];

		float Ldist = length(light.pos - fhp);

		float distance_attenuation = 1 / (1 + linear_attenuation_factor * Ldist 
			+ quadratic_attenuation_factor * Ldist * Ldist);

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
			prd_radiance.result += angle_attenuation * light.intensity * light.color 
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
			prd_radiance.result += light.intensity * light.color * shadow_prd.attenuation
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
		prd_radiance.result += 0.8 * diffuse_importance * kd 
			* TraceRay(fhp, random_ray_direction, new_depth, diffuse_importance / 2);
	}

	/* subsurface scattering */
	if (!float_vec_zero(subsurf_scatter_color)) {
		const float3 ss_dir = - reflect(ray.direction, normal);

		float3 randomized_ss_dir = randomize_vector(ss_dir, 1, seed);
		prd_radiance.result += 0.8 * subsurf_scatter_color
			* TraceRay(fhp, randomized_ss_dir, new_depth, prd_radiance.importance * 0.8);
	}

	/* reflection */
	float reflective_importance = reflective_amount / total_amount * prd_radiance.importance;

	if (!float_vec_zero(k_reflective) && reflective_importance > importance_cutoff) {

		// reflection direction
		const float3 refl = reflect(ray.direction, normal);

		// number of samples to take for each reflection
		if (glossy_on) {
			// const int num_glossy_sample = 10;
			const int num_glossy_sample = 1;
			
			for (int i = 0; i < num_glossy_sample; i++) {
				float3 randomizedRefl = randomize_vector(refl, glossiness, seed);
				prd_radiance.result += reflective_importance * k_reflective
					* TraceRay(fhp, randomizedRefl, new_depth, reflective_importance) / num_glossy_sample;
			}
		} else {
			prd_radiance.result += k_reflective 
				* TraceRay(fhp, refl, new_depth, reflective_importance);
		}
	}

	/* refraction */
	float refractive_importance = refractive_amount / total_amount * prd_radiance.importance;

	if (!float_vec_zero(alpha) && refractive_importance > importance_cutoff) {
		// back hit point
		const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);

		float3 transmission_direction;
		if (refract(transmission_direction, ray.direction, normal, IOR)) {
			// check whether it is internal or external refraction
			/*
			float cos_theta = dot(ray.direction, normal);
			if (cos_theta < 0) { // external
				cos_theta = - cos_theta;
			} else { // internal
				cos_theta = dot(transmission_direction, normal);
			}

			float reflection = fresnel_schlick(cos_theta, 3, 0.1, 1);

			float importance = prd_radiance.importance * (1.0f-reflection) * luminance( alpha );
			*/

			prd_radiance.result += alpha 
				* TraceRay(bhp, ray.direction, new_depth, refractive_importance / 2);
		}
	}
	
}
