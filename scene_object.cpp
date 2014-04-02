#include "scene_object.h"
#include "collider.h"

#include <ObjLoader.h>
#include <ImageLoader.h>

#include <sstream>

Context SceneObject::context = NULL;
Program SceneObject::closest_hit = NULL;
Program SceneObject::any_hit= NULL;
Program SceneObject::mesh_intersect = NULL;
Program SceneObject::mesh_bounds = NULL;

float3 float3FromString(std::string s) {
	std::stringstream ss(s);
	std::string seg;

	float f[3];

	for (int i = 0; i < 3; i++) {
		getline(ss, seg, ' ');

		f[i] = atof(seg.c_str());
	}

	return make_float3(f[0], f[1], f[2]);
}

SceneObject::SceneObject() {
	m_initialTransformMtx = NULL;

	m_ke = make_float3(0, 0, 0);
	m_ka = make_float3(0.2, 0.2, 0.2);
	m_kd = make_float3(0.5, 0.5, 0.5);
	m_ks = make_float3(0.3, 0.3, 0.3);
	m_kr = make_float3(0, 0, 0);
	m_ns = 10;
	
	m_emissive = false;
	m_collider = NULL;
}

void SceneObject::attachCollider(Collider* c) {
	m_collider = c;
	c->setParent(this);
}

void SceneObject::parseMtlFile(Material mat, std::string mtl_path) {
	std::ifstream mtlInput(mtl_path);

	std::vector<std::string> lines;

	for (std::string line; getline(mtlInput, line); ) {
		std::stringstream ss(line);

		std::string type, value;
		getline(ss, type, ' ');
		getline(ss, value, '\n');
		
		if (type == "Ka") {
			mat["k_ambient"]->setFloat(float3FromString(value));
		} else if (type == "Kd") {
			mat["k_diffuse"]->setFloat(float3FromString(value));
		} else if (type == "Ks") {
			mat["k_specular"]->setFloat(float3FromString(value));
		} else if (type == "Ke") {
			float ke = atoi(value.c_str());
			if (ke > 0) {
				mat["k_emission"]->setFloat(make_float3(ke, ke, ke));
				mat["is_emissive"]->setInt(true);
				m_emissive = true;
			}
		} else if (type == "map_Kd") {

		}
	}
}

void SceneObject::initGraphics(std::string obj_path, std::string mtl_path) {
	m_objPath = obj_path;

	GeometryGroup group = context->createGeometryGroup();

	Material mat = context->createMaterial();
	mat->setClosestHitProgram(0, closest_hit);
	mat->setAnyHitProgram(1, any_hit);

	ObjLoader loader(m_objPath.c_str(), context, group, mat);
	loader.setIntersectProgram(mesh_intersect);
	loader.setBboxProgram(mesh_bounds);
	loader.load();

	m_transform = context->createTransform();
	m_transform->setChild(group);

	mat["is_emissive"]->setInt(m_emissive);
	mat["k_emission"]->setFloat(m_ke);
	mat["k_ambient"]->setFloat(m_ka);
	mat["k_diffuse"]->setFloat(m_kd);
	mat["k_specular"]->setFloat(m_ks);
	mat["k_reflective"]->setFloat(m_kr);
	mat["ns"]->setInt(m_ns);

	mat["importance_cutoff"]->setFloat( 0.01f );
	mat["cutoff_color"]->setFloat( 0.2f, 0.2f, 0.2f );
	mat["reflection_maxdepth"]->setInt( 5 );

	mat["kd_map"]->setTextureSampler(loadTexture(context, 
		obj_path + m_diffuseMapFilename, 
		make_float3(1, 1, 1)));
	mat["ks_map"]->setTextureSampler(loadTexture(context, 
		obj_path + m_specularMapFilename, 
		make_float3(1, 1, 1)));
	mat["normal_map"]->setTextureSampler(loadTexture(context, 
		obj_path + m_normalMapFilename, 
		make_float3(1, 1, 1)));

	mat["has_diffuse_map"]->setInt(!m_diffuseMapFilename.empty());
	mat["has_normal_map"]->setInt(!m_normalMapFilename.empty());
	mat["has_specular_map"]->setInt(!m_specularMapFilename.empty());

	parseMtlFile(mat, mtl_path);
}

RectangleLight SceneObject::createAreaLight() {
	ConvexDecomposition::WavefrontObj wo;
	wo.loadObj(m_objPath.c_str());

	// now we only support rectangular area light
	assert(wo.mVertexCount == 4);

	float3 v1 = make_float3(wo.mVertices[0], wo.mVertices[1], wo.mVertices[2]);
	float3 v2 = make_float3(wo.mVertices[3], wo.mVertices[4], wo.mVertices[5]);
	float3 v3 = make_float3(wo.mVertices[6], wo.mVertices[7], wo.mVertices[8]);
	float3 v4 = make_float3(wo.mVertices[9], wo.mVertices[10], wo.mVertices[11]);

	RectangleLight light;
	light.pos = v1;
	light.r1 = v2 - v1;
	light.r2 = v4 - v1;

	// temp value
	light.color = make_float3(1, 1, 1);

	return light;
}