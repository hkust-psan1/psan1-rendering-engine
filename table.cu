#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, Ks, , );

struct PerRayData_radiance {
	float3 result;
	float importance;
	int depth;
};

struct PerRayData_shadow {
	float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayLoad, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

RT_PROGRAM void any_hit_shadow() {
}

RT_PROGRAM void closest_hit_radiance() {
}