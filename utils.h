#include "random.h"
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

inline float random1() {
	return (float)rand()/(float)RAND_MAX;
}

inline float2 random2() {
	return make_float2( random1(), random1() );
}

inline float3 random3() {
	return make_float3( random1(), random1(), random1() );
}

inline float4 random4() {
	return make_float4( random1(), random1(), random1(), random1() );
}