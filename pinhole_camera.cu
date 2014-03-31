#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "helpers.h"
#include "random.h"

using namespace optix;

struct PerRayData_radiance
{
	float3 result;
	float importance;
	int depth;
};

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
rtDeclareVariable(float, focal_scale, , );


// #define TIME_VIEW


__device__ __forceinline__ void write_output( float3 c )
{
	output_buffer[launch_index] = make_float4(c, 1.f);
}

__device__ __forceinline__ float3 read_output()
{
	return make_float3(output_buffer[launch_index]);
}

#define PI 3.1415926

RT_PROGRAM void pinhole_camera()
{
#ifdef TIME_VIEW
	clock_t t0 = clock(); 
#endif
	unsigned int seed = tea<16>(launch_index.y * launch_dim.x + launch_index.x, launch_index.y);;

	float3 result = make_float3(0, 0, 0);
	float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;

	const int numDofSamples = 50;
	for (int i = 0; i < numDofSamples; i++) {
		PerRayData_radiance prd;
		prd.importance = 1.f / numDofSamples;
		prd.depth = 0;

		// randomly sample eye positions on a disk
		float rand_dist = rnd(seed) * 0.01;
		float rand_angle = rnd(seed) * 2 * PI;

		float rand_x = rand_dist * cos(rand_angle);
		float rand_y = rand_dist * sin(rand_angle);

		// float3 ray_origin = eye + make_float3(0, 0, 0.1 * i);
		float3 ray_origin = eye + V * rand_x + U * rand_y;
		float3 ray_direction = normalize(d.x * U + d.y * V + W * 2);

		optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

		rtTrace(top_object, ray, prd);

		result += prd.result / numDofSamples;
	}

#ifdef TIME_VIEW
	clock_t t1 = clock(); 
 
	float expected_fps	 = 1.0f;
	float pixel_time		 = ( t1 - t0 ) * time_view_scale * expected_fps;
	write_output( make_float3( pixel_time ) );
#else
	// write_output(prd.result);
	write_output(result);
#endif
}

RT_PROGRAM void exception()
{
	write_output(bad_color);
}
