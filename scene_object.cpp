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
	m_ka = make_float3(0, 0, 0);
	m_kd = make_float3(0, 0, 0);
	m_ks = make_float3(0);
	m_krefl = make_float3(0);
	m_alpha = make_float3(0);
	m_ss = make_float3(0);
	m_ns = 10;
	m_glossiness = 0;

	m_attenuation_coeff = 0.1;
	
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

	if (0) {
		if (obj_path.find("Wall") != std::string::npos) {
			m_diffuseMapFilename = "brick_COLOR.ppm";
			m_ks = make_float3(0);
		} else if (obj_path.find("Light") != std::string::npos) {
			m_ke = make_float3(1, 0.8, 0.6);
			m_emissive = true;
		} else if (obj_path.find("Floor") != std::string::npos) {
			m_diffuseMapFilename = "floor_COLOR.ppm";
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
		} else if (obj_path.find("GlassPane") != std::string::npos) {
			m_kd = make_float3(0);
			m_krefl = make_float3(0);
			m_alpha = make_float3(1.0, 0.9, 0.9);
		} else if (obj_path.find("GlassSphere") != std::string::npos) {
			m_kd = make_float3(0);
			m_krefl = make_float3(0);
			m_alpha = make_float3(1.0, 1.0, 1.0);
		}
	}

	// used for the kitchen scene
	if (0) {
		if (obj_path.find("MainLightSource") != std::string::npos) {
			m_ke = make_float3(1);
			m_intensity = 5;
			m_attenuation_coeff = 0.1;
			m_emissive = true;
		} else if (obj_path.find("BalconyLightSource") != std::string::npos) {
			m_ke = make_float3(0.25, 0.63, 0.80);
			m_intensity = 1;
			m_attenuation_coeff = 0.1;
			m_emissive = true;
		} else if (obj_path.find("MainStructure") != std::string::npos) {
			m_kd = make_float3(0.9);
		} else if (obj_path.find("BackWall") != std::string::npos) {
			m_kd = make_float3(1);
			m_krefl = make_float3(0.1);
		} else if (obj_path.find("WindowFrame") != std::string::npos) {
			m_kd = make_float3(0.8);
			m_krefl = make_float3(0.5);
			m_glossiness = 0.4;
		} else if (obj_path.find("WindowGlass") != std::string::npos) {
			m_krefl = make_float3(0.2);
			m_alpha = make_float3(0.9);
		} else if (obj_path.find("Floor") != std::string::npos) {
			m_diffuseMapFilename = "wood_floor_interior_2.ppm";
		} else if (obj_path.find("TableFrame") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("TableTop") != std::string::npos) {
			m_kd = make_float3(0.1);
			m_krefl = make_float3(0.6);
			m_glossiness = 0.1;
		} else if (obj_path.find("ChairFrame") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("ChairTop") != std::string::npos) {
			m_kd = make_float3(0.1);
			m_krefl = make_float3(0.6);
			m_glossiness = 0.2;
		} else if (obj_path.find("Cabinet") != std::string::npos) {
			m_diffuseMapFilename = "drawer_front_color.ppm";
			m_specularMapFilename = "drawer_front_spec.ppm";
		} else if (obj_path.find("PlantLeaves") != std::string::npos) {
			m_kd = make_float3(0.6, 0.9, 0.2);
			m_ss = make_float3(0.6, 0.9, 0.2);
		} else if (obj_path.find("FlowerPot") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("Plates") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("Bowls") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("Bottles") != std::string::npos) {
			m_krefl = make_float3(0.3);
			m_alpha = make_float3(0.7);
		} else if (obj_path.find("MetalContainer") != std::string::npos) {
			m_krefl = make_float3(0.8);
		} else if (obj_path.find("LampCover") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("WaterFaucet") != std::string::npos) {
			m_krefl = make_float3(0.8);
			m_glossiness = 0.02;
		} else if (obj_path.find("BackBoards") != std::string::npos) {
			m_kd = make_float3(0.8, 0.56, 0.2);
		} else if (obj_path.find("Hood_Middle") != std::string::npos) {
			m_diffuseMapFilename = "hood.ppm";
			m_krefl = make_float3(0.4);
			m_glossiness = 0.02;
		}
	}

	// used for the dining room scene
	if (0) {
		if (obj_path.find("HungLightSource") != std::string::npos) {
			m_ke = make_float3(1);
			m_intensity = 0.2;
			m_emissive = true;
		} else if (obj_path.find("SideRoofLight") != std::string::npos) {
		} else if (obj_path.find("MiddleRoofLight") != std::string::npos) {
			m_ke = make_float3(1);
			m_intensity = 1.5;
			m_emissive = true;
		} else if (obj_path.find("OuterLight") != std::string::npos) {
			m_ke = make_float3(1, 0.95, 0.85);
			m_intensity = 3.0;
			m_emissive = true;
		} else if (obj_path.find("MainFloor") != std::string::npos) {
			m_diffuseMapFilename = "floor_COLOR.ppm";
			m_specularMapFilename = "floor_SPEC.ppm";
			m_krefl = make_float3(0.7);
			m_glossiness = 0.4;
		} else if (obj_path.find("ChairFrame") != std::string::npos) {
			m_kd = make_float3(1.0);
		} else if (obj_path.find("ChairTop") != std::string::npos) {
			m_kd = make_float3(0);
			m_krefl = make_float3(0.2);
			m_glossiness = 0.1;
		} else if (obj_path.find("MainStructure") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("CabinetBody") != std::string::npos) {
			m_kd = make_float3(0.8, 0.7, 0.4);
		} else if (obj_path.find("CabinetGlass") != std::string::npos) {
			m_alpha = make_float3(0.7);
			m_krefl = make_float3(0.5);
		} else if (obj_path.find("Drawers") != std::string::npos) {
			m_kd = make_float3(0.8, 0.7, 0.4);
		} else if (obj_path.find("Painting") != std::string::npos) {
			m_diffuseMapFilename = "painting.ppm";
		} else if (obj_path.find("WallWithPaper") != std::string::npos) {
			m_diffuseMapFilename = "congruent_pentagon.ppm";
		} else if (obj_path.find("WallLowerSide") != std::string::npos) {
			m_kd = make_float3(0.1);
			m_krefl = make_float3(0.3);
			m_glossiness = 0.2;
			m_diffuseMapFilename = "marble_wall_lower_COLOR.ppm";
		} else if (obj_path.find("WallPads") != std::string::npos) {
			m_kd = make_float3(0.85, 0.8, 0.6);
		} else if (obj_path.find("RoofTop") != std::string::npos) {
			m_kd = make_float3(1);
		} else if (obj_path.find("RoofStructure") != std::string::npos) {
			m_kd = make_float3(1, 0.9, 0.8);
		} else if (obj_path.find("HungLamps") != std::string::npos) {
			m_kd = make_float3(0);
			m_krefl = make_float3(0.7);
			m_glossiness = 0.2;
		} else if (obj_path.find("Bowl") != std::string::npos) {
			m_kd = make_float3(0.1);
			m_krefl = make_float3(0.8);
			m_glossiness = 0.1;
		} else if (obj_path.find("TableGlass") != std::string::npos) {
			m_alpha = make_float3(0.7);
			m_krefl = make_float3(0.5);
		} else if (obj_path.find("TablePillar") != std::string::npos) {
			m_kd = make_float3(0.1);
			m_krefl = make_float3(0.6);
			m_glossiness = 0.03;
		} else if (obj_path.find("AnisSurface") != std::string::npos) {
			m_kd = make_float3(0.1);
			m_krefl = make_float3(1.8);
			m_glossiness = 0.5;
			m_anisotropicMapFilename = "bottom_anis.ppm";
		}
	}

	// used for the bowling scene
	if (0) {
		if (obj_path.find("SideLight") != std::string::npos) {
			m_ke = make_float3(1);
			m_intensity = 15;
			m_emissive = true;
		} else if (obj_path.find("Pin") != std::string::npos) {
			m_diffuseMapFilename = "pin-diffuse.ppm";
		} else if (obj_path.find("BowlingBall") != std::string::npos) {
			m_krefl = make_float3(0.3);
			m_glossiness = 0.1;
			m_diffuseMapFilename = "cellgn.ppm";
		} else if (obj_path.find("MainFloor") != std::string::npos) {
			m_diffuseMapFilename = "wood_floor.ppm";
			m_krefl = make_float3(0.4);
		}
	}

	// used for the throw scene
	if (0) {
		if (obj_path.find("TopLight") != std::string::npos) {
			m_ke = make_float3(1);
			m_intensity = 1.2;
			m_emissive = true;
			m_attenuation_coeff = 0;
		} else if (obj_path.find("Base") != std::string::npos) {
			m_kd = make_float3(0.5);
		} else if (obj_path.find("Pole") != std::string::npos) {
			m_kd = make_float3(0.7);
		} else if (obj_path.find("Ground") != std::string::npos) {
			m_diffuseMapFilename = "marble_texture_milk.ppm";
			m_krefl = make_float3(0.4);
			m_glossiness = 0.1;
		} else if (obj_path.find("Glass") != std::string::npos) {
			m_krefl = make_float3(0.05);
			m_alpha = make_float3(1.0);
			m_glossiness = 0;
		} else if (obj_path.find("Milk1") != std::string::npos) {
			m_kd = make_float3(1);
			m_ss = make_float3(1);
		} else if (obj_path.find("Milk2") != std::string::npos) {
			m_kd = make_float3(1);
		}
	}

	// used for the anisotrophic scene
	if (0) {
		if (obj_path.find("Circle") != std::string::npos) {
			m_krefl = make_float3(0.7);
			m_glossiness = 3;
			m_anisotropicMapFilename = "anis.ppm";
		} else if (obj_path.find("LightSource") != std::string::npos) {
			m_emissive = true;
			m_ke = make_float3(1);
			m_intensity = 1;
		}
	}

	// used for testing ss
	if (0) {
		if (obj_path.find("Light") != std::string::npos) {
			m_ke = make_float3(1);
			m_intensity = 3;
			m_emissive = true;
		} else if (obj_path.find("SS_body") != std::string::npos) {
			m_kd = make_float3(0.3, 0.25, 0.05);
			m_ss = make_float3(0.9, 0.3, 0);
		}
	}

	// used for the lift scene
	if (0) {
		if (obj_path.find("Door") != std::string::npos) {
			m_diffuseMapFilename = "brushed_metal.ppm";
			m_krefl = make_float3(1);
			m_glossiness = 0.5;
			m_anisotropicMapFilename = "anis.ppm";
		} else if (obj_path.find("TopLight") != std::string::npos) {
			m_emissive = true;
			m_ke = make_float3(1);
			m_intensity = 0.6;
			m_attenuation_coeff = 0.03;
		} else if (obj_path.find("Monkey") != std::string::npos) {
			m_kd = make_float3(1, 0.7, 0);
		}
	}

	// used for the pot scene
	if (1) {
		if (obj_path.find("LightSource") != std::string::npos) {
			m_emissive = true;
			m_ke = make_float3(1);
			m_intensity = 1.5;
		} else if (obj_path.find("Pan") != std::string::npos) {
			m_diffuseMapFilename = "brushed_metal.ppm";
			m_krefl = make_float3(1);
			m_glossiness = 0.5;
			m_anisotropicMapFilename = "anis.ppm";
		}
	}

	/*
	Buffer areaLightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
	areaLightBuffer->setFormat(RT_FORMAT_USER);
	areaLightBuffer->setElementSize(sizeof(RectangleLight));
	areaLightBuffer->setSize(areaLights.size());
	memcpy(areaLightBuffer->map(), areaLightArray, sizeof(RectangleLight) * areaLights.size());
	areaLightBuffer->unmap();
	m_context["area_lights"]->set(areaLightBuffer);
	*/

	mat["is_emissive"]->setInt(m_emissive);
	mat["k_emission"]->setFloat(m_ke);
	mat["k_ambient"]->setFloat(m_ka);
	mat["k_diffuse"]->setFloat(m_kd);
	mat["k_specular"]->setFloat(m_ks);
	mat["k_reflective"]->setFloat(m_krefl);
	mat["alpha"]->setFloat(m_alpha);
	mat["subsurf_scatter_color"]->setFloat(m_ss);
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
	mat["anisotrophic_map"]->setTextureSampler(loadTexture(context, 
		tex_dir + m_anisotropicMapFilename, make_float3(1, 1, 1)));

	mat["has_diffuse_map"]->setInt(!m_diffuseMapFilename.empty());
	mat["has_normal_map"]->setInt(!m_normalMapFilename.empty());
	mat["has_specular_map"]->setInt(!m_specularMapFilename.empty());
	mat["has_anisotrophic_map"]->setInt(!m_anisotropicMapFilename.empty());
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

	light.color = m_ke;

	light.intensity = m_intensity;

	light.attenuation_coeff = m_attenuation_coeff;

	return light;
}