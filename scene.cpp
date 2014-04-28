#include "gui_control.h"
#include "scene.h"
#include "utils.h"
#include "node_shading_system.h"

unsigned int Scene::WIDTH = 512u;
unsigned int Scene::HEIGHT = 384u;

Scene::Scene( const std::string& obj_path, int camera_type ) 
	: SampleScene(), m_obj_path( obj_path ), m_frame_number( 0u ), 
	m_camera_type( camera_type ) {

	m_jitter_grid_num = 4;
	m_jitter_base_x = 0;
	m_jitter_base_y = 0;

	m_dof_sample_num = 0;

	m_taking_snapshot = false;

	samplesPerFrame = 20;
}

void Scene::initScene( InitialCameraData& camera_data ) {
	try {
		optix::Material material[3];
		createContext( camera_data );

		initObjects();
		// resetObjects();

		m_context->validate();
		m_context->compile();

	} catch( Exception& e ) {
		sutilReportError( e.getErrorString().c_str() );
		exit( 2 );
	}
}

Buffer Scene::getOutputBuffer() {
	return m_context["output_buffer"]->getBuffer();
}

void Scene::trace(const RayGenCameraData& camera_data) {
	if (m_new_frame) {
		m_new_frame = false;
	}

	/* Optix rendering settings */
	if (m_camera_changed) {
		m_frame_number = 0u;
		m_camera_changed = false;
	}

	m_context["eye"]->setFloat(camera_data.eye);
	m_context["U"]->setFloat(camera_data.U);
	m_context["V"]->setFloat(camera_data.V);
	m_context["W"]->setFloat(camera_data.W);
	m_context["frame_number"]->setUint(m_frame_number++);

	m_context["soft_shadow_on"]->setInt(GUIControl::softShadowOn);
	m_context["aa_on"]->setInt(GUIControl::aaOn);
	m_context["dof_on"]->setInt(GUIControl::dofOn);
	m_context["glossy_on"]->setInt(GUIControl::glossyOn);
	m_context["gi_on"]->setInt(GUIControl::giOn);

	m_context["jitter_grid_size"]->setFloat(1.f / m_jitter_grid_num);

	float2 jitter_base = make_float2(
		-0.5 + m_jitter_base_x / (float) m_jitter_grid_num,
		-0.5 + m_jitter_base_y / (float) m_jitter_grid_num);
	m_context["jitter_base"]->setFloat(jitter_base);

	m_context["focal_scale"]->setFloat(GUIControl::cameraFocalScale);

	Buffer buffer = m_context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize( buffer_width, buffer_height );

	m_context->launch(getEntryPoint(),
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height));

	m_dof_sample_num++;

	bool dof_completed = true;
	if (GUIControl::dofOn) {
		if (m_dof_sample_num == 30) {
			m_dof_sample_num = 0;

			if (GUIControl::aaOn) {
				m_jitter_base_x++;
			}
		} else {
			dof_completed = false;
		}
	} else {
		if (GUIControl::aaOn) {
			m_jitter_base_x++;
		}
	}

	bool aa_completed = true;
	if (GUIControl::aaOn) {
		if (m_jitter_base_x == m_jitter_grid_num) { // row completed
			m_jitter_base_x = 0;
			m_jitter_base_y++;
		}

		if (m_jitter_base_y == m_jitter_grid_num) { // frame completed
			m_jitter_base_y = 0;
		} else { // supersampling not completed
			aa_completed = false;
		}
	}

	m_new_frame = dof_completed && aa_completed;

	if (m_new_frame && m_taking_snapshot) {
		m_taking_snapshot = false;
		GUIControl::snapShotButton->activate();
	}

	if (m_frame_number == samplesPerFrame)
	{
		if (GUIControl::onAnimation) 
		{ // when animation is on, step simulation
			world->stepSimulation(1 / 100.f, 10);
		}

		// re-position objects according to simulation result
		for (int i = 0; i < sceneObjects.size(); i++) 
		{
			SceneObject* so = sceneObjects[i];
			Collider* c = so->getCollider();

			if (c) 
			{
				c->step();
			}
		}

		// mark acceleration structure dirty
		g->getAcceleration()->markDirty();

		//reset
		m_camera_changed = true;

		//write picture
		cv::Mat img(WIDTH, HEIGHT, cv::DataType<cv::Vec3b>::type);

		writer << img;
		printf("hi");

		/*
		std::vector<unsigned char> pix(WIDTH * HEIGHT * 3);
		Buffer buffer = m_context["output_buffer"]->getBuffer(); 
		float* src = (float*)(&buffer);
				
		printf("hey");
		for (unsigned int i = 0u; i < WIDTH; i++)
		{			
			for (unsigned int j = 0u; j < HEIGHT; j++)
			{
				//float r = *(src++);
				//float g= *(src++);
				//float b = *(src++);
				//src++;
				//unsigned int R = r < 0 ? 0 : r > 0xff ? 0xff : r;
				//unsigned int G = g < 0 ? 0 : g > 0xff ? 0xff : g;
				//unsigned int B = b < 0 ? 0 : b > 0xff ? 0xff : b;
				img.at<cv::Vec3b>(i, j) = cv::Vec3b(i % 256, j % 256, (i * j) % 256);
			}
		}
		*/
		
		
	}
}

void Scene::doResize( unsigned int width, unsigned int height ) {
	// We need to update buffer sizes if resized (output_buffer handled in base class)
	m_context["variance_sum_buffer"]->getBuffer()->setSize( width, height );
	m_context["variance_sum2_buffer"]->getBuffer()->setSize( width, height );
	m_context["num_samples_buffer"]->getBuffer()->setSize( width, height );
	m_context["rnd_seeds"]->getBuffer()->setSize( width, height );
}

void Scene::createContext( InitialCameraData& camera_data ) {
	// Context
	m_context->setEntryPointCount( 3 );
	m_context->setRayTypeCount( 3 );
	m_context->setStackSize( 9600 );

	m_context["scene_epsilon"]->setFloat( 1.e-3f );
	m_context["radiance_ray_type"]->setUint( 0u );
	m_context["shadow_ray_type"]->setUint( 1u );
	m_context["max_depth"]->setInt( 8 );
	m_context["frame_number"]->setUint( 0u );
	
	m_context["focal_scale"]->setFloat( 0.0f ); // Value is set in trace()
	m_context["aperture_radius"]->setFloat(0.1f);
	m_context["frame_number"]->setUint(1);
	distance_offset = -1.5f;

	// Output buffer.
	Buffer buffer = createOutputBuffer( RT_FORMAT_FLOAT4, WIDTH, HEIGHT );
	m_context["output_buffer"]->set(buffer);
	
	// Pinhole Camera ray gen and exception program
	std::string	ptx_path = ptxpath( "tracer", "dof_camera.cu" );
	m_context->setRayGenerationProgram(DOF, m_context->createProgramFromPTXFile( ptx_path, "dof_camera" ) );
	m_context->setExceptionProgram(DOF, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

	// Adaptive Pinhole Camera ray gen and exception program
	ptx_path = ptxpath( "tracer", "adaptive_pinhole_camera.cu" );
	m_context->setRayGenerationProgram(AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
	m_context->setExceptionProgram(AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
	
	ptx_path = ptxpath("tracer", "pinhole_camera.cu" );
	m_context->setRayGenerationProgram(Pinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
	m_context->setExceptionProgram(Pinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
	
	// Setup lighting
	/*
	BasicLight lights[] = 
	{ 
		{ make_float3( 20.0f,	20.0f, 20.0f ), make_float3( 0.6f, 0.5f, 0.4f ), 3 },
	};

	Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();

	m_context["lights"]->set(light_buffer);
	*/

	m_context["ambient_light_color"]->setFloat( 0.1f, 0.1f, 0.1f );
	
	// Used by both exception programs
	m_context["bad_color"]->setFloat( 1.0f, 0.0f, 0.0f );

	// Miss program.
	ptx_path = ptxpath( "tracer", "gradientbg.cu" );
	// m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
	m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "envmap_miss" ) );
	const char* filename = "D:/OptiX SDK 3.0.1/SDK - Copy/tracer/CedarCity.hdr";
	m_context["envmap"]->setTextureSampler(loadTexture(m_context, filename, make_float3(0)));
	m_context["background_light"]->setFloat( 1.0f, 1.0f, 1.0f );
	m_context["background_dark"]->setFloat( 0.3f, 0.3f, 0.3f );

	// align background's up direction with camera's look direction
	float3 bg_up = make_float3(-14.0f, -14.0f, -7.0f);
	bg_up = normalize(bg_up);

	// tilt the background's up direction in the direction of the camera's up direction
	bg_up.y += 1.0f;
	bg_up = normalize(bg_up);
	m_context["up"]->setFloat( bg_up.x, bg_up.y, bg_up.z );
	
	// Set up camera position

	// for the dining room scene
	camera_data = InitialCameraData( make_float3( 6.65, 1.45, -1.29 ), // eye
		make_float3( 0.0, 1.45, 5 ),		// lookat
		make_float3( 0.0, 1.0, 0.0 ),		// up
		45.0 );	// vfov

	// for the kitchen scene
	/*
	camera_data = InitialCameraData( make_float3( 9.0, 2.93, -0.15 ), // eye
		make_float3( 0.0, 2.93, 0.0 ),		// lookat
		make_float3( 0.0, 1.0, 0.0 ),		// up
		45.0 );	// vfov
		*/

	// Declare camera variables. The values do not matter, they will be overwritten in trace.
	m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

	// Variance buffers
	Buffer variance_sum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
		RT_FORMAT_FLOAT4, WIDTH, HEIGHT );
	memset( variance_sum_buffer->map(), 0, WIDTH*HEIGHT*sizeof(float4) );
	variance_sum_buffer->unmap();
	m_context["variance_sum_buffer"]->set( variance_sum_buffer );

	Buffer variance_sum2_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
		RT_FORMAT_FLOAT4, WIDTH, HEIGHT );
	memset( variance_sum2_buffer->map(), 0, WIDTH*HEIGHT*sizeof(float4) );
	variance_sum2_buffer->unmap();
	m_context["variance_sum2_buffer"]->set( variance_sum2_buffer );

	// Sample count buffer
	Buffer num_samples_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
		RT_FORMAT_UNSIGNED_INT, WIDTH, HEIGHT );
	memset( num_samples_buffer->map(), 0, WIDTH*HEIGHT*sizeof(unsigned int) );
	num_samples_buffer->unmap();
	m_context["num_samples_buffer"]->set( num_samples_buffer);

	// RNG seed buffer
	m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
		RT_FORMAT_UNSIGNED_INT, WIDTH, HEIGHT );
	genRndSeeds( WIDTH, HEIGHT );
	m_context["rnd_seeds"]->set( m_rnd_seeds );
}

void Scene::genRndSeeds( unsigned int width, unsigned int height ) {
	unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
	fillRandBuffer(seeds, width*height);
	m_rnd_seeds->unmap();
}

void Scene::resetObjects() 
{
	const float pinRadius = 0.44; // from the obj file
	const float pinDistance = 0.44 * 12 / (4.75 / 2); // 12 inches distance with 4.75 inches diameter
	const float unitX = pinDistance * 1.73205 / 2;
	const float unitZ = pinDistance / 2;

	const float initY = 0;

	float3 pinBasePosition = make_float3(10, 0, 0);
	float3 pinPositions[10] = {
		make_float3(0, initY, 0) + pinBasePosition,
		make_float3(unitX, initY, unitZ) + pinBasePosition,
		make_float3(unitX, initY, - unitZ) + pinBasePosition,
		make_float3(2 * unitX, initY, 2 * unitZ) + pinBasePosition,
		make_float3(2 * unitX, initY, 0) + pinBasePosition,
		make_float3(2 * unitX, initY, - 2 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, initY, - 3 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, initY, - 1 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, initY, 1 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, initY, 3 * unitZ) + pinBasePosition,
	};

	world->stepSimulation(0.1, 1);
}

void Scene::initObjects() 
{
	std::string filename = "D:\\tracer_output.avi";
	const int format = CV_FOURCC('M', 'P', '4', '2');

	writer.open(filename, format, 10, cv::Size(WIDTH, HEIGHT), true);
	if (!writer.isOpened())
    {
        assert(false);
    }

	std::string prog_path = ptxpath("tracer", "triangle_mesh_iterative.cu");
	std::string mat_path = ptxpath("tracer", "phong.cu");

	SceneObject::closest_hit = m_context->createProgramFromPTXFile(mat_path, "closest_hit_radiance");
	SceneObject::any_hit = m_context->createProgramFromPTXFile(mat_path, "any_hit_shadow");

	SceneObject::mesh_intersect = m_context->createProgramFromPTXFile(prog_path, "mesh_intersect");
	SceneObject::mesh_bounds = m_context->createProgramFromPTXFile(prog_path, "mesh_bounds");

	SceneObject::context  = m_context;

	// process obj file
	ObjFileProcessor ofp;
	sceneObjects = ofp.processObject(m_obj_path + "lift", m_obj_path + "objs/");

	btDbvtBroadphase* broadPhase = new btDbvtBroadphase();

	btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

	world = new btDiscreteDynamicsWorld(dispatcher, broadPhase, solver, collisionConfiguration);

	for (int i = 0; i < sceneObjects.size(); i++) {
		SceneObject* so = sceneObjects[i];
		Collider* c = so->getCollider();
		if (c) {
			world->addRigidBody(c->getRigidBody());
		}
	}

	g = m_context->createGroup();
	g->setChildCount(sceneObjects.size());

	g->setAcceleration(m_context->createAcceleration("Bvh", "Bvh"));

	m_context["top_object"]->set(g);
	m_context["top_shadower"]->set(g);

	std::vector<RectangleLight> areaLights;
	std::vector<SpotLight> spotLights;
	std::vector<DirectionalLight> directionalLights;

	for (int i = 0; i < sceneObjects.size(); i++) {
		SceneObject* so = sceneObjects[i];
		if (so->m_emissive) {
			areaLights.push_back(so->createAreaLight());
		}
		g->setChild<Transform>(i, so->getTransform());
	}

	// add area lights to the scene
	RectangleLight* areaLightArray = NULL;
	if (!areaLights.empty()) {
		areaLightArray = &areaLights[0];
	}

	Buffer areaLightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
	areaLightBuffer->setFormat(RT_FORMAT_USER);
	areaLightBuffer->setElementSize(sizeof(RectangleLight));
	areaLightBuffer->setSize(areaLights.size());
	memcpy(areaLightBuffer->map(), areaLightArray, sizeof(RectangleLight) * areaLights.size());
	areaLightBuffer->unmap();
	m_context["area_lights"]->set(areaLightBuffer);

	SpotLight sl1 = {
		make_float3(2.56, 3.76, 2.89), // position
		make_float3(1, 0.95, 0.8), // color
		make_float3(0, -1, 0), // direction
		120 / 57.3 / 2, // angle
		4, // intensity
		8, // dropoff rate
		0.3 // attenuation factor
	};

	// spotLights.push_back(sl1);

	// add spot lights to the scene
	SpotLight* spotLightArray = NULL;
	if (!spotLights.empty()) {
		spotLightArray = &spotLights[0];
	}

	Buffer spotLightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
	spotLightBuffer->setFormat(RT_FORMAT_USER);
	spotLightBuffer->setElementSize(sizeof(SpotLight));
	spotLightBuffer->setSize(spotLights.size());
	memcpy(spotLightBuffer->map(), spotLightArray, sizeof(SpotLight) * spotLights.size());
	spotLightBuffer->unmap();
	m_context["spot_lights"]->set(spotLightBuffer);
	
	DirectionalLight dl1 = {
		make_float3(5, -3, 0), // direction
		make_float3(1, 1, 1), // color
		1, // intensity
		0.1 // attenuation factor
	};

	// directionalLights.push_back(dl1);

	// add directional lights to the scene
	DirectionalLight* directionalLightArray = NULL;
	if (!directionalLights.empty()) {
		directionalLightArray = &directionalLights[0];
	}

	Buffer directionalLightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
	directionalLightBuffer->setFormat(RT_FORMAT_USER);
	directionalLightBuffer->setElementSize(sizeof(DirectionalLight));
	directionalLightBuffer->setSize(directionalLights.size());
	memcpy(directionalLightBuffer->map(), directionalLightArray, sizeof(DirectionalLight) * directionalLights.size());
	directionalLightBuffer->unmap();
	m_context["directional_lights"]->set(directionalLightBuffer);
	
}

void Scene::makeMaterialPrograms( Material material, const char *filename, 
	const char *ch_program_name,
	const char *ah_program_name ) {
	Program ch_program = m_context->createProgramFromPTXFile( ptxpath("tracer", filename), ch_program_name );
	Program ah_program = m_context->createProgramFromPTXFile( ptxpath("tracer", filename), ah_program_name );

	material->setClosestHitProgram( 0, ch_program );
	material->setAnyHitProgram( 1, ah_program );
}