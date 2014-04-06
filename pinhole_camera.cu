#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "helpers.h"
#include "random.h"

#define DOF

using namespace optix;

struct PerRayData_radiance
{
	float3 result;
	float importance;
	int depth;
};

// whether to used jittered rendering
rtDeclareVariable(int, aa_on, ,) = false;
rtDeclareVariable(int, dof_on, ,) = false;

// dimension of the jittered sampling grid
rtDeclareVariable(float, jitter_grid_size, ,) = 1.f;

// base of the jitter index
rtDeclareVariable(float2, jitter_base, ,);

rtDeclareVariable(float3, eye, , ) = { 1.0f, 0.0f, 0.0f };
rtDeclareVariable(float3, U, , ) = { 0.0f, 1.0f, 0.0f };
rtDeclareVariable(float3, V, , ) = { 0.0f, 0.0f, 1.0f };
rtDeclareVariable(float3, W, , ) = { -1.0f, 0.0f, 0.0f };
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(float, scene_epsilon, , ) = 0.1f;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int,	radiance_ray_type, , );

rtBuffer<float4, 2> output_buffer;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(float, aperture_radius, , );
rtDeclareVariable(float, focal_scale, , ) = 0.8;

rtDeclareVariable(unsigned int, frame_number, , );

__device__ __forceinline__ void write_output( float3 c )
{
	output_buffer[launch_index] = make_float4(c, 1.f);
}

__device__ __forceinline__ float3 read_output()
{
	return make_float3(output_buffer[launch_index]);
}

RT_PROGRAM void pinhole_camera()
{
	unsigned int seed = tea<16>(launch_index.y * launch_dim.x + launch_index.x, launch_index.y + frame_number);;

	float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;

	float3 result;

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	float3 ray_origin = eye;
	float3 ray_direction = d.x * U + d.y * V + W;

	if (aa_on) {
		float x_index = launch_index.x + jitter_base.x + jitter_grid_size * rnd(seed);
		float y_index = launch_index.y + jitter_base.y + jitter_grid_size * rnd(seed);
		float2 jittered_d = make_float2(x_index, y_index) / make_float2(launch_dim) * 2.f - 1.f;

		ray_direction = jittered_d.x * U + jittered_d.y * V + W;
	}

	if (dof_on) {
		float3 ray_target = ray_origin + focal_scale * ray_direction;
		float2 sample = square_to_disk(make_float2(rnd(seed), rnd(seed)));
		ray_origin += 0.1 * (sample.x * normalize(U) + sample.y * normalize(V));
		ray_direction = ray_target - ray_origin;
	}

	optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	rtTrace(top_object, ray, prd);

	result = prd.result;
	
	/*
	int supersampling = 4;
	float sampling_step = 0.25;

	for (int i = 0; i < supersampling; i++) {
		for (int j = 0; j < supersampling; j++) {
			float base_x_index = launch_index.x - 0.5 + sampling_step * i;
			float base_y_index = launch_index.y - 0.5 + sampling_step * i;

			float x_index = base_x_index + sampling_step * rnd(seed);
			float y_index = base_y_index + sampling_step * rnd(seed);

			float2 jittered_d = make_float2(x_index, y_index) / make_float2(launch_dim) * 2.f - 1.f;
			float3 ray_direction = normalize(jittered_d.x * U + jittered_d.y * V + W);

			optix::Ray ray = optix::make_Ray(eye, ray_direction, 
				radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

			rtTrace(top_object, ray, prd);

			result += prd.result / (supersampling * supersampling);
		}
	}
	*/

	write_output(result);
}

RT_PROGRAM void exception()
{
	write_output(bad_color);
}
