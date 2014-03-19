
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
#include "commonStructs.h"

using namespace optix;

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(rtObject,     top_shadower, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(int,          max_depth, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, tangent, attribute tangent, ); 
rtDeclareVariable(float3, bitangent, attribute bitangent, ); 
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, isect_dist, rtIntersectionDistance, );

rtDeclareVariable(float3,       tile_size, , ); 
rtDeclareVariable(float3,       tile_color_dark, , );
rtDeclareVariable(float3,       tile_color_light, , );
rtDeclareVariable(float3,       ambient_light_color, , );
rtDeclareVariable(float3,       ka, , );
rtDeclareVariable(float3,       kr, , );
// rtDeclareVariable(float3,       ks, , );
rtDeclareVariable(int,       ns, , );

rtTextureSampler<float4, 2>     kd_map;
rtTextureSampler<float4, 2>     ks_map;
rtTextureSampler<float4, 2>		normal_map;
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3,       cutoff_color, , );
rtDeclareVariable(int,          reflection_maxdepth, , );
rtDeclareVariable(float,        importance_cutoff, , );

rtDeclareVariable(int, has_normal_map, , );


rtBuffer<BasicLight> lights;

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
  prd_shadow.attenuation = make_float3(0.0f);
  rtTerminateRay();
}

// -----------------------------------------------------------------------------

RT_PROGRAM void closest_hit_radiance()
{
  const float3 i = ray.direction; 
  const float3 uvw = texcoord;
  const float3 kd = make_float3( tex2D( kd_map, uvw.x, uvw.y ) );
  const float3 ks = make_float3( tex2D( ks_map, uvw.x, uvw.y ) );

  const float3 T = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, tangent)); // tangent  
  const float3 B = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, bitangent)); // bitangent  
  const float3 N = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal  

  // normal vector after considering normal map (if any)
  float3 normal;

  if (has_normal_map) {
	  const float3 k_normal = make_float3( tex2D( normal_map, uvw.x, uvw.y) );
	  const float3 coeff = k_normal * 2 - make_float3(1, 1, 1);
	  normal = T * coeff.x + B * coeff.y + N * coeff.z;
  } else {
	  normal = N;
  }

  const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);

  //attenuation
  const float beer_attenuation = 1.0f;

  const int depth = prd_radiance.depth;
  float3 result = kd * ambient_light_color;
  float3 r;
  float reflection = 1.0f;
    
  float3 color = cutoff_color;
  
  unsigned int num_lights = lights.size();
  for (int i = 0; i < num_lights; i++) {
	BasicLight light = lights[i];
	float Ldist = length(light.pos - fhp);

    float3 L = normalize(light.pos - fhp);
	float3 H = normalize(L - ray.direction);
	// float nDl = dot(N, L);
	float nDl = dot(normal, L);

	// cast shadow ray
    PerRayData_shadow shadow_prd;
    shadow_prd.attenuation = make_float3(1);

	if(nDl > 0) {
      optix::Ray shadow_ray = optix::make_Ray( fhp, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
	  result += light.color * shadow_prd.attenuation * (kd * nDl + ks * max(pow(dot(H, normal), ns), .0f));
    }
  }

  if (depth < min(reflection_maxdepth, max_depth))
  {
    // r = reflect(i, N);
    r = reflect(i, normal);
      
    float importance = prd_radiance.importance * reflection * optix::luminance( kr * beer_attenuation );
    if ( importance > importance_cutoff ) 
	{
      color = TraceRay( fhp, r, depth+1, importance );
    }
  }

  result += kd * reflection * kr * color;
  
  prd_radiance.result = result;
}

