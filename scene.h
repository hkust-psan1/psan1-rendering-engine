#ifndef SCENE_
#define SCENE_

#include <Windows.h>
#include <stdio.h>
#include <conio.h>

#include <ctime>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <ImageLoader.h>
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <sutil.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include "common_structs.h"
#include <string.h>
#include "random.h"

#include "scene_object.h"
#include "collider.h"

#include "obj_file_processor.h"
#include "gui_control.h"

using namespace optix;

class Scene : public SampleScene {
public:
	Scene( const std::string& obj_path, int camera_type );

	// From SampleScene
	void initScene( InitialCameraData& camera_data );
	void trace( const RayGenCameraData& camera_data );
	void doResize( unsigned int width, unsigned int depth );
	Buffer getOutputBuffer();

	void initObjects();
	void resetObjects();

	btDiscreteDynamicsWorld* world;

	std::vector<SceneObject*> sceneObjects;
	// std::vector<BowlingPin*> pins;

	// Ball* ball;
	std::string	 m_obj_path;

	int m_dof_sample_num;

	int m_camera_type;

	void createContext( SampleScene::InitialCameraData& camera_data );
	void createMaterials(Material material[] );

	// Helper functions
	void makeMaterialPrograms( Material material, const char *filename, 
		const char *ch_program_name, const char *ah_program_name );

	int getEntryPoint() { return m_camera_type; }
	void genRndSeeds(unsigned int width, unsigned int height);

	enum {
		DOF = 0,
		AdaptivePinhole = 1,
		Pinhole = 2
	};

	void createGeometry();

	Buffer m_rnd_seeds;
	unsigned int m_frame_number;

	float distance_offset;

	static unsigned int WIDTH;
	static unsigned int HEIGHT;

	Group g;

	int currSampleInFrame;
};

#endif