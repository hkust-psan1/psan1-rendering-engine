#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "helpers.h"

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

RT_PROGRAM void pinhole_camera()
{
#ifdef TIME_VIEW
	clock_t t0 = clock(); 
#endif
	float3 result = make_float3(0, 0, 0);

	for (float i = 0; i <= 0.1; i += 0.25) {
		for (float j = 0; j <= 0.1; j += 0.25) {
			PerRayData_radiance prd;
			prd.importance = 1.f;
			prd.depth = 0;

			float2 d = (make_float2(launch_index) + make_float2(i, j)) / make_float2(launch_dim) * 2.f - 1.f;
			float3 ray_origin = eye;
			float3 ray_direction = normalize(d.x * U + d.y * V + W);

			optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

			rtTrace(top_object, ray, prd);

			result += prd.result;
		}
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
