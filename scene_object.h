#ifndef SCENE_OBJECT_
#define SCENE_OBJECT_

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "common_structs.h"

using namespace optix;

class Collider;

class SceneObject {
public:
	SceneObject();

	void attachCollider(Collider* c);

	void parseMtlFile(Material mat, std::string mtl_path);

	virtual void initGraphics(std::string obj_path, std::string mtl_path);

	RectangleLight createAreaLight();

	inline Transform getTransform() const { return m_transform; };
	inline void setTransformMatrix(float m[]) { m_transform->setMatrix(false, m, NULL); };

	inline Collider* getCollider() const { return m_collider; };

	std::string m_renderObjFilename;
	std::string m_diffuseMapFilename;
	std::string m_normalMapFilename;
	std::string m_specularMapFilename;

	bool m_emissive;
	float3 m_ke;
	float3 m_ka;
	float3 m_kd;
	float3 m_ks;
	float3 m_kr;
	float m_ns;

	static Context context;
	static Program mesh_intersect;
	static Program mesh_bounds;
	static Program closest_hit;
	static Program any_hit;

protected:
	Transform m_transform;
	float* m_initialTransformMtx;

	std::string m_objPath;

	Collider* m_collider;
};


#endif