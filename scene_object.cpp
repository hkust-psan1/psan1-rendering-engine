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

	m_ke = make_float3(0);
	m_ka = make_float3(0.2, 0.2, 0.2);
	m_kd = make_float3(0.5, 0.5, 0.5);
	m_ks = make_float3(0);
	m_krefl = make_float3(0);
	m_krefr = make_float3(0);
	m_ns = 10;
	m_glossiness = 0;
	
	m_emissive = false;
	m_collider = NULL;
}

void SceneObject::attachCollider(Collider* c) {
	m_collider = c;
	c->setParent(this);
}

void SceneObject::parseMtlFile(Material mat, std::string mtl_path, std::string tex_dir) {
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
			mat["kd_map"]->setTextureSampler(loadTexture(
				context, tex_dir + value, make_float3(1, 1, 1)));
			mat["has_diffuse_map"]->setInt(true);
		}
	}
}

void SceneObject::initGraphics(std::string obj_path, std::string mtl_path, std::string tex_dir) {
	int tmp, pos = 0;
    int lastIndex;
    while ((tmp = obj_path.find('/', pos)) != std::string::npos) {
        pos = tmp + 1;
        lastIndex = tmp;
    }
	std::string objName = obj_path.substr(lastIndex + 1, obj_path.length() - 1);

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

	// used for the kitchen scene
	if (obj_path.find("Wall") != std::string::npos) {
		m_diffuseMapFilename = "brick_COLOR.ppm";
		m_ks = make_float3(0);
	} else if (obj_path.find("Light") != std::string::npos) {
		m_ke = make_float3(1);
		m_emissive = true;
	} else if (obj_path.find("Floor") != std::string::npos) {
		m_diffuseMapFilename = "floor_COLOR.ppm";
		// m_krefl = make_float3(0.5);
		/*
		m_glossiness = 0.3;
		*/
	} else if (obj_path.find("Table") != std::string::npos) {
		m_diffuseMapFilename = "table_COLOR.ppm";
		m_ks = make_float3(0.1);
	} else if (obj_path.find("Cabinet") != std::string::npos) {
		m_diffuseMapFilename = "cabinet.ppm";
	} else if (obj_path.find("Marble-Top") != std::string::npos) {
		m_diffuseMapFilename = "marble-table_COLOR.ppm";
	} else if (obj_path.find("Board") != std::string::npos) {
		m_diffuseMapFilename = "wooden_board.ppm";
	} else if (obj_path.find("Suzanne") != std::string::npos) {
		m_ks = make_float3(0.6);
		m_krefl = make_float3(0.5);
	} else if (obj_path.find("Hidden") != std::string::npos) {
		m_kd = make_float3(0.8);
	} else if (obj_path.find("Left") != std::string::npos) {
		m_kd = make_float3(0.8);
	} else if (obj_path.find("Roof") != std::string::npos) {
		// m_krefl = make_float3(0.5);
	} else if (obj_path.find("Glass") != std::string::npos) {
		m_kd = make_float3(0.1);
		m_krefr = make_float3(0.9);
	}

	// used for the bowling scene

	mat["is_emissive"]->setInt(m_emissive);
	mat["k_emission"]->setFloat(m_ke);
	mat["k_ambient"]->setFloat(m_ka);
	mat["k_diffuse"]->setFloat(m_kd);
	mat["k_specular"]->setFloat(m_ks);
	mat["k_reflective"]->setFloat(m_krefl);
	mat["k_refractive"]->setFloat(m_krefr);
	mat["ns"]->setInt(m_ns);
	mat["glossiness"]->setFloat(m_glossiness);

	mat["importance_cutoff"]->setFloat( 0.01f );
	mat["cutoff_color"]->setFloat( 0.2f, 0.2f, 0.2f );
	mat["reflection_maxdepth"]->setInt( 5 );

	mat["kd_map"]->setTextureSampler(loadTexture(context, 
		tex_dir + m_diffuseMapFilename, make_float3(1, 1, 1)));
	mat["ks_map"]->setTextureSampler(loadTexture(context, 
		tex_dir + m_specularMapFilename, make_float3(1, 1, 1)));
	mat["normal_map"]->setTextureSampler(loadTexture(context, 
		tex_dir + m_normalMapFilename, make_float3(1, 1, 1)));

	mat["has_diffuse_map"]->setInt(!m_diffuseMapFilename.empty());
	mat["has_normal_map"]->setInt(!m_normalMapFilename.empty());
	mat["has_specular_map"]->setInt(!m_specularMapFilename.empty());

	// parseMtlFile(mat, mtl_path, tex_dir);
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