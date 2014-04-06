#include "gui_control.h"
#include "scene.h"
#include "utils.h"

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


		if (GUIControl::onAnimation) { // when animation is on, step simulation
			world->stepSimulation(1 / 30.f, 10);
		}

		// re-position objects according to simulation result
		for (int i = 0; i < sceneObjects.size(); i++) {
			SceneObject* so = sceneObjects[i];
			Collider* c = so->getCollider();

			if (c) {
				c->step();
			}
		}

		// mark acceleration structure dirty
		g->getAcceleration()->markDirty();
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
}

void Scene::doResize( unsigned int width, unsigned int height ) {
	// We need to update buffer sizes if resized (output_buffer handled in base class)
	m_context["variance_sum_buffer"]->getBuffer()->setSize( width, height );
	m_context["variance_sum2_buffer"]->getBuffer()->setSize( width, height );
	m_context["num_samples_buffer"]->getBuffer()->setSize( width, height );
	m_context["rnd_seeds"]->getBuffer()->setSize( width, height );
}

// Return whether we processed the key or not
bool Scene::keyPressed(unsigned char key, int x, int y) {
	float r = m_context["aperture_radius"]->getFloat();
	switch (key)
	{
	case 'z':		
		r += 0.01f;
		m_context["aperture_radius"]->setFloat(r);
		std::cout << "Aperture radius: " << r << std::endl;
			m_camera_changed = true;
		return true;
	case 'x':		
		r -= 0.01f;
		m_context["aperture_radius"]->setFloat(r);
		std::cout << "Aperture radius: " << r << std::endl;
			m_camera_changed = true;
		return true;
	case ',':		
		distance_offset -= 0.1f;
		std::cout << "Distance offset" << distance_offset << std::endl;
			m_camera_changed = true;
		return true;
	case '.':		
		distance_offset += 0.1f;
		std::cout << "Distance offset" << distance_offset << std::endl;
			m_camera_changed = true;
		return true;
	}
	return false;
}

void Scene::createContext( InitialCameraData& camera_data ) {
	// Context
	m_context->setEntryPointCount( 3 );
	m_context->setRayTypeCount( 2 );
	m_context->setStackSize( 2400 );

	m_context["scene_epsilon"]->setFloat( 1.e-3f );
	m_context["radiance_ray_type"]->setUint( 0u );
	m_context["shadow_ray_type"]->setUint( 1u );
	m_context["max_depth"]->setInt( 10 );
	m_context["frame_number"]->setUint( 0u );
	
	m_context["focal_scale"]->setFloat( 0.0f ); // Value is set in trace()
	m_context["aperture_radius"]->setFloat(0.1f);
	m_context["frame_number"]->setUint(1);
	m_context["jitter"]->setFloat(0.0f, 0.0f, 0.0f, 0.0f);
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

	m_context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );
	
	// Used by both exception programs
	m_context["bad_color"]->setFloat( 0.0f, 1.0f, 1.0f );

	// Miss program.
	ptx_path = ptxpath( "tracer", "gradientbg.cu" );
	m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
	m_context["background_light"]->setFloat( 1.0f, 1.0f, 1.0f );
	m_context["background_dark"]->setFloat( 0.3f, 0.3f, 0.3f );

	// align background's up direction with camera's look direction
	float3 bg_up = make_float3(-14.0f, -14.0f, -7.0f);
	bg_up = normalize(bg_up);

	// tilt the background's up direction in the direction of the camera's up direction
	bg_up.y += 1.0f;
	bg_up = normalize(bg_up);
	m_context["up"]->setFloat( bg_up.x, bg_up.y, bg_up.z );
	
	// Set up camera
	camera_data = InitialCameraData( make_float3( 30.0f, 15.0f, 7.5f ), // eye
		make_float3( 7.0f, .0f, 7.0f ),		// lookat
		make_float3( 0.0f, 1.0f, 0.0f ),		// up
		45.0f );	// vfov
	

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

void Scene::resetObjects() {
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

	/*
	for (int i = 0; i < pins.size(); i++) {
		pins[i]->setInitialPosition(pinPositions[i]);
		pins[i]->getRigidBody()->setLinearVelocity(btVector3(0, 0, 0));
		pins[i]->getRigidBody()->setAngularVelocity(btVector3(0, 0, 0));
	}

	ball->setInitialPosition(make_float3(-10, 0, 0));
	*/

	world->stepSimulation(0.1, 1);
}

void Scene::initObjects() {
	std::string prog_path = ptxpath("tracer", "triangle_mesh_iterative.cu");
	std::string mat_path = ptxpath("tracer", "phong.cu");

	SceneObject::closest_hit = m_context->createProgramFromPTXFile(mat_path, "closest_hit_radiance");
	SceneObject::any_hit = m_context->createProgramFromPTXFile(mat_path, "any_hit_shadow");

	SceneObject::mesh_intersect = m_context->createProgramFromPTXFile(prog_path, "mesh_intersect");
	SceneObject::mesh_bounds = m_context->createProgramFromPTXFile(prog_path, "mesh_bounds");

	SceneObject::context  = m_context;

	std::vector<RectangleLight> areaLights;

	// process obj file
	ObjFileProcessor ofp;
	sceneObjects = ofp.processObject(m_obj_path + "interior", m_obj_path + "objs/");

	/*
	GroundPlane* groundPlane = new GroundPlane(m_context);
	groundPlane->initGraphics(m_obj_path);
	groundPlane->initPhysics(m_obj_path);
	sceneObjects.push_back(groundPlane);

	for (int i = 0; i < 1; i++) {
		BowlingPin* pin = new BowlingPin(m_context);
		pin->initGraphics(m_obj_path);
		pin->initPhysics(m_obj_path);

		sceneObjects.push_back(pin); 
		pins.push_back(pin);
	}

	ball = new Ball(m_context);
	ball->initGraphics(m_obj_path);
	ball->initPhysics(m_obj_path);
	sceneObjects.push_back(ball);
	*/

	/*
	SceneObject* sphere = new SceneObject;
	sphere->initGraphics(m_obj_path + "cubes.obj");
	sceneObjects.push_back(sphere);
	*/

	btDbvtBroadphase* broadPhase = new btDbvtBroadphase();

	btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

	world = new btDiscreteDynamicsWorld(dispatcher, broadPhase, solver, collisionConfiguration);

	// add a ground plane
	/*
	btCollisionShape* groundShape = new btBoxShape(btVector3(10, 1, 10));
	btVector3 localInertia(0, 0, 0);
	btTransform groundTransform;
	groundTransform.setIdentity();
	groundTransform.setOrigin(btVector3(0, 0, 0));
	btDefaultMotionState* groundMotionState = new btDefaultMotionState(groundTransform);
	btRigidBody::btRigidBodyConstructionInfo info(0, groundMotionState, groundShape, localInertia);
	btRigidBody* groundBody = new btRigidBody(info);

	world->addRigidBody(groundBody);
	*/

	/*
	for (int i = 0; i < sceneObjects.size(); i++) {
		PhysicalObject* po = dynamic_cast<PhysicalObject*>(sceneObjects[i]);
		if (po) {
			world->addRigidBody(po->getRigidBody());
		}
	}
	*/

	for (int i = 0; i < sceneObjects.size(); i++) {
		SceneObject* so = sceneObjects[i];
		Collider* c = so->getCollider();
		if (c) {
			world->addRigidBody(c->getRigidBody());
		}
	}

	/*
	SceneObject* sample_light = new SceneObject;
	sample_light->m_emissive = true;
	sample_light->initGraphics(m_obj_path + "sample_light.obj", m_obj_path + "sample_light.mtl");
	areaLights.push_back(sample_light->createAreaLight());
	ofp.sceneObjects.push_back(sample_light);
	*/

	g = m_context->createGroup();
	g->setChildCount(sceneObjects.size());

	for (int i = 0; i < sceneObjects.size(); i++) {
		SceneObject* so = sceneObjects[i];
		if (so->m_emissive) {
			areaLights.push_back(so->createAreaLight());
		}
		g->setChild<Transform>(i, so->getTransform());
	}

	g->setAcceleration(m_context->createAcceleration("Bvh", "Bvh"));

	m_context["top_object"]->set(g);
	m_context["top_shadower"]->set(g);

	// add area lights to the scene
	RectangleLight* areaLightArray = &areaLights[0];
	Buffer areaLightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
	areaLightBuffer->setFormat(RT_FORMAT_USER);
	areaLightBuffer->setElementSize(sizeof(RectangleLight));
	areaLightBuffer->setSize(areaLights.size());
	memcpy(areaLightBuffer->map(), areaLightArray, sizeof(RectangleLight) * areaLights.size());
	areaLightBuffer->unmap();
	m_context["area_lights"]->set(areaLightBuffer);
}

void Scene::makeMaterialPrograms( Material material, const char *filename, 
	const char *ch_program_name,
	const char *ah_program_name ) {
	Program ch_program = m_context->createProgramFromPTXFile( ptxpath("tracer", filename), ch_program_name );
	Program ah_program = m_context->createProgramFromPTXFile( ptxpath("tracer", filename), ah_program_name );

	material->setClosestHitProgram( 0, ch_program );
	material->setAnyHitProgram( 1, ah_program );
}