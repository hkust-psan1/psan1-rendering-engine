
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

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

rtDeclareVariable(float3,        eye, , ) = { 1.0f, 0.0f, 0.0f };
rtDeclareVariable(float3,        U, , )   = { 0.0f, 1.0f, 0.0f };
rtDeclareVariable(float3,        V, , )   = { 0.0f, 0.0f, 1.0f };
rtDeclareVariable(float3,        W, , )   = { -1.0f, 0.0f, 0.0f };
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , ) = 0.1f;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtBuffer<float4, 2>              output_buffer;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
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
  return make_float3( output_buffer[launch_index] );
}

RT_PROGRAM void pinhole_camera()
{
#ifdef TIME_VIEW
  clock_t t0 = clock(); 
#endif
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

#ifdef TIME_VIEW
  clock_t t1 = clock(); 
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  write_output( make_float3( pixel_time ) );
#else
    write_output(prd.result);
#endif
}

RT_PROGRAM void exception()
{
  write_output(bad_color);
}
