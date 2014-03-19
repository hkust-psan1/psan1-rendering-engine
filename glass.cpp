
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

//------------------------------------------------------------------------------
//
//  A glass shader example.
//
//------------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "random.h"

#include <ImageLoader.h>
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <sutil.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include "commonStructs.h"
#include <string.h>

#include "bowling_pin.h"

using namespace optix;

//------------------------------------------------------------------------------
//
//  Glass scene 
//
//------------------------------------------------------------------------------

inline float random1()
{
  return (float)rand()/(float)RAND_MAX;
}

inline float2 random2()
{
  return make_float2( random1(), random1() );
}

inline float3 random3()
{
  return make_float3( random1(), random1(), random1() );
}

inline float4 random4()
{
  return make_float4( random1(), random1(), random1(), random1() );
}

class GlassScene : public SampleScene
{
public:
  GlassScene( const std::string& obj_path, bool aaa, bool gg ) 
    : SampleScene(), m_obj_path( obj_path ), m_frame_number( 0u ), m_adaptive_aa( aaa ), m_green_glass( gg ) {}

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int depth );
  Buffer getOutputBuffer();
  bool keyPressed(unsigned char key, int x, int y);

private:
	void initObjects(const std::string& path);

  void createContext( SampleScene::InitialCameraData& camera_data );
  void createMaterials(Material material[] );

  // Helper functions
  void makeMaterialPrograms( Material material, const char *filename,
                                                const char *ch_program_name,
                                                const char *ah_program_name );

  int getEntryPoint() { return m_adaptive_aa ? AdaptivePinhole: DOF; }
  void genRndSeeds(unsigned int width, unsigned int height);

  enum 
  {
    DOF = 0,
    AdaptivePinhole = 1
  };

  void createGeometry();

  Buffer        m_rnd_seeds;
  std::string   m_obj_path;
  unsigned int  m_frame_number;
  bool          m_adaptive_aa;
  bool          m_green_glass;

  float distance_offset;

  static unsigned int WIDTH;
  static unsigned int HEIGHT;

  btDiscreteDynamicsWorld* world;

  std::vector<SceneObject*> sceneObjects;

  Group g;
};

unsigned int GlassScene::WIDTH  = 512u;
unsigned int GlassScene::HEIGHT = 384u;


void GlassScene::genRndSeeds( unsigned int width, unsigned int height )
{
  unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
  fillRandBuffer(seeds, width*height);
  m_rnd_seeds->unmap();
}

void GlassScene::initScene( InitialCameraData& camera_data ) 
{
  try {
    optix::Material material[3];
    createContext( camera_data );
    createMaterials( material );
    // createGeometry( material, m_obj_path );

	initObjects(m_obj_path);

    m_context->validate();
    m_context->compile();

  } catch( Exception& e ) {
    sutilReportError( e.getErrorString().c_str() );
    exit( 2 );
  }
}


Buffer GlassScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void GlassScene::trace( const RayGenCameraData& camera_data )
{
	/* apply transformation obtained from bullet physics */
	world->stepSimulation(1 / 300.f, 10);

	for (int i = 0; i < sceneObjects.size(); i++) {
		PhysicalObject* po = dynamic_cast<PhysicalObject*>(sceneObjects[i]);
		if (po) {
			po->step();
		}
	}

	g->getAcceleration()->markDirty();

	/* Optix rendering settings */
  if ( m_camera_changed ) 
  {
    m_frame_number = 0u;
    m_camera_changed = false;
  }

  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );
  m_context["frame_number"]->setUint( m_frame_number++ );

  float focal_distance = length(camera_data.W) + distance_offset;
  focal_distance = fmaxf(focal_distance, m_context["scene_epsilon"]->getFloat());
  float focal_scale = focal_distance / length(camera_data.W);
  m_context["focal_scale"]->setFloat( focal_scale );
  
  m_context["jitter"]->setFloat( random4() );
  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( getEntryPoint(),
                   static_cast<unsigned int>(buffer_width),
                   static_cast<unsigned int>(buffer_height)
                   );
}


void GlassScene::doResize( unsigned int width, unsigned int height )
{
  // We need to update buffer sizes if resized (output_buffer handled in base class)
  m_context["variance_sum_buffer"]->getBuffer()->setSize( width, height );
  m_context["variance_sum2_buffer"]->getBuffer()->setSize( width, height );
  m_context["num_samples_buffer"]->getBuffer()->setSize( width, height );
  m_context["rnd_seeds"]->getBuffer()->setSize( width, height );
  genRndSeeds( width, height );
}


// Return whether we processed the key or not
bool GlassScene::keyPressed(unsigned char key, int x, int y)
{
  float r = m_context["aperture_radius"]->getFloat();
  switch (key)
  {
  case 'a':
    m_adaptive_aa = !m_adaptive_aa;
    m_camera_changed = true;
	GLUTDisplay::setContinuousMode( m_adaptive_aa ? GLUTDisplay::CDProgressive : GLUTDisplay::CDProgressive );
    return true;
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


void  GlassScene::createContext( InitialCameraData& camera_data )
{
  // Context
  m_context->setEntryPointCount( 2 );
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
  Variable output_buffer = m_context["output_buffer"];
  Buffer buffer = createOutputBuffer( RT_FORMAT_FLOAT4, WIDTH, HEIGHT );
  output_buffer->set(buffer);
  
  // Pinhole Camera ray gen and exception program
  std::string         ptx_path = ptxpath( "glass", "pinhole_camera.cu" );
  m_context->setRayGenerationProgram( DOF, m_context->createProgramFromPTXFile( ptx_path, "dof_camera" ) );
  m_context->setExceptionProgram(     DOF, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  // Adaptive Pinhole Camera ray gen and exception program
  ptx_path = ptxpath( "glass", "adaptive_pinhole_camera.cu" );
  m_context->setRayGenerationProgram( AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  
  // Setup lighting
  BasicLight lights[] = 
  { 
    { make_float3( -30.0f,  20.0f, -80.0f ), make_float3( 0.6f, 0.5f, 0.4f ), 1 },
    { make_float3( -30.0f,  -20.0f, -80.0f ), make_float3( 0.6f, 0.5f, 0.4f ), 1 },
    { make_float3(  10.5f,  30.0f, 20.5f ), make_float3( 0.65f, 0.65f, 0.6f ), 1 },
    /* { make_float3(  10.5f,  30.0f, 20.4f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.5f,  30.0f, 20.3f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.5f,  30.0f, 20.2f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.5f,  30.0f, 20.1f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.4f,  30.0f, 20.5f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.4f,  30.0f, 20.4f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.4f,  30.0f, 20.3f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.4f,  30.0f, 20.2f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.4f,  30.0f, 20.1f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.3f,  30.0f, 20.5f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.3f,  30.0f, 20.4f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.3f,  30.0f, 20.3f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.3f,  30.0f, 20.2f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.3f,  30.0f, 20.1f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.2f,  30.0f, 20.5f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.2f,  30.0f, 20.4f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.2f,  30.0f, 20.3f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.2f,  30.0f, 20.2f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.2f,  30.0f, 20.1f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.1f,  30.0f, 20.5f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.1f,  30.0f, 20.4f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.1f,  30.0f, 20.3f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.1f,  30.0f, 20.2f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 },
    { make_float3(  10.1f,  30.0f, 20.1f ), make_float3( 0.025f, 0.025f, 0.027f ), 1 }*/
  };

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);
  m_context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );
  
  // Used by both exception programs
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 1.0f );

  // Miss program.
  ptx_path = ptxpath( "glass", "gradientbg.cu" );
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
                                   make_float3( 7.0f, .0f, 7.0f ),    // lookat
                                   make_float3( 0.0f, 1.0f, 0.0f ),    // up
                                   45.0f );                            // vfov
  

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Variance buffers
  Buffer variance_sum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                       RT_FORMAT_FLOAT4,
                                                       WIDTH, HEIGHT );
  memset( variance_sum_buffer->map(), 0, WIDTH*HEIGHT*sizeof(float4) );
  variance_sum_buffer->unmap();
  m_context["variance_sum_buffer"]->set( variance_sum_buffer );

  Buffer variance_sum2_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                        RT_FORMAT_FLOAT4,
                                                        WIDTH, HEIGHT );
  memset( variance_sum2_buffer->map(), 0, WIDTH*HEIGHT*sizeof(float4) );
  variance_sum2_buffer->unmap();
  m_context["variance_sum2_buffer"]->set( variance_sum2_buffer );

  // Sample count buffer
  Buffer num_samples_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                      RT_FORMAT_UNSIGNED_INT,
                                                      WIDTH, HEIGHT );
  memset( num_samples_buffer->map(), 0, WIDTH*HEIGHT*sizeof(unsigned int) );
  num_samples_buffer->unmap();
  m_context["num_samples_buffer"]->set( num_samples_buffer);

  // RNG seed buffer
  m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                       RT_FORMAT_UNSIGNED_INT,
                                       WIDTH, HEIGHT );
  genRndSeeds( WIDTH, HEIGHT );
  m_context["rnd_seeds"]->set( m_rnd_seeds );
}


void GlassScene::createMaterials( Material material[] )
{
  material[0] = m_context->createMaterial();
  material[1] = m_context->createMaterial();
  material[2] = m_context->createMaterial();

  makeMaterialPrograms(material[0], "checkerboard.cu", "closest_hit_radiance", "any_hit_shadow");
  
  material[0]["ka" ]->setFloat( 0.2f, 0.2f, 0.2f );
  material[0]["ks" ]->setFloat( 0.5f, 0.5f, 0.5f );
  material[0]["kr" ]->setFloat( 0.8f, 0.8f, 0.8f );
  material[0]["ns" ]->setInt( 32 );
  material[0]["importance_cutoff"  ]->setFloat( 0.01f );
  material[0]["cutoff_color"       ]->setFloat( 0.2f, 0.2f, 0.2f );
  material[0]["reflection_maxdepth"]->setInt( 5 );
  material[0]["kd_map"]->setTextureSampler( loadTexture( m_context, "D:\\OptiX SDK 3.0.1\\SDK - Copy\\glass\\wood_floor.ppm", make_float3(1, 1, 1)) );

 

  // Checkerboard to aid positioning, not used in final setup.
  makeMaterialPrograms(material[1], "checkerboard.cu", "closest_hit_radiance", "any_hit_shadow");
  
  material[1]["ka" ]->setFloat( 0.6f, 0.6f, 0.6f );
  material[1]["ks" ]->setFloat( 0.5f, 0.5f, 0.5f );
  material[1]["kr" ]->setFloat( 0.0f, 0.0f, 0.0f );
  material[1]["ns" ]->setInt( 64 );
  material[1]["importance_cutoff"  ]->setFloat( 0.01f );
  material[1]["cutoff_color"       ]->setFloat( 0.2f, 0.2f, 0.2f );
  material[1]["reflection_maxdepth"]->setInt( 5 );
  material[1]["kd_map"]->setTextureSampler( loadTexture( m_context, "D:\\OptiX SDK 3.0.1\\SDK - Copy\\glass\\pin-diffuse.ppm", make_float3(1, 1, 1)) );

  makeMaterialPrograms(material[2], "checkerboard.cu", "closest_hit_radiance", "any_hit_shadow");
  
  material[2]["ka" ]->setFloat( 0.2f, 0.2f, 0.2f );
  material[2]["ks" ]->setFloat( 0.5f, 0.5f, 0.5f );
  material[2]["kr" ]->setFloat( 0.7f, 0.7f, 0.7f );
  material[2]["ns" ]->setInt( 64 );
  material[2]["importance_cutoff"  ]->setFloat( 0.01f );
  material[2]["cutoff_color"       ]->setFloat( 0.2f, 0.2f, 0.2f );
  material[2]["reflection_maxdepth"]->setInt( 5 );
  material[2]["kd_map"]->setTextureSampler( loadTexture( m_context, "D:\\OptiX SDK 3.0.1\\SDK - Copy\\glass\\cloth.ppm", make_float3(1, 1, 1)) );

}

void GlassScene::initObjects(const std::string& res_path) {
	const float pinRadius = 0.44; // from the obj file
	const float pinDistance = 0.44 * 12 / (4.75 / 2); // 12 inches distance with 4.75 inches diameter
	const float unitX = pinDistance * 1.73205 / 2;
	const float unitZ = pinDistance / 2;

	std::vector<RectangleLight> areaLights;

	std::string mesh_path = ptxpath( "glass", "triangle_mesh_iterative.cu" );
	std::string mat_path = ptxpath("glass", "checkerboard.cu");

	float3 pinBasePosition = make_float3(10, 0, 0);
	float3 pinPositions[10] = {
		make_float3(0, 0, 0) + pinBasePosition,
		make_float3(unitX, 0, unitZ) + pinBasePosition,
		make_float3(unitX, 0, - unitZ) + pinBasePosition,
		make_float3(2 * unitX, 0, 2 * unitZ) + pinBasePosition,
		make_float3(2 * unitX, 0, 0) + pinBasePosition,
		make_float3(2 * unitX, 0, - 2 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, 0, - 3 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, 0, - 1 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, 0, 1 * unitZ) + pinBasePosition,
		make_float3(3 * unitX, 0, 3 * unitZ) + pinBasePosition,
	};

	for (int i = 0; i < 10; i++) {
		BowlingPin* pin = new BowlingPin(m_context);
		pin->initGraphics(mesh_path, mat_path, res_path);
		pin->initPhysics(res_path);
		pin->setInitialPosition(pinPositions[i]);
		sceneObjects.push_back(pin); 
	}

	GroundPlane* groundPlane = new GroundPlane(m_context);
	groundPlane->initGraphics(mesh_path, mat_path, res_path);
	groundPlane->initPhysics(res_path);
	sceneObjects.push_back(groundPlane);

	Ball* ball = new Ball(m_context);
	ball->initGraphics(mesh_path, mat_path, res_path);
	ball->initPhysics(res_path);
	ball->setInitialPosition(make_float3(-10, 0, 0));
	ball->getRigidBody()->setLinearVelocity(btVector3(100, 0, 1));
	sceneObjects.push_back(ball);

	SceneObject* banner = new SceneObject(m_context);
	banner->m_renderObjFilename = "/banner.obj";
	banner->m_diffuseMapFilename = "/banner_diffuse.ppm";
	banner->initGraphics(mesh_path, mat_path, res_path);

	SceneObject* ditch = new SceneObject(m_context);
	ditch->m_renderObjFilename = "/ditch.obj";
	ditch->m_kd = make_float3(0.05, 0.05, 0.05);
	ditch->m_ks = make_float3(0.5, 0.5, 0.5);
	ditch->m_kr = make_float3(0.2, 0.2, 0.2);
	ditch->initGraphics(mesh_path, mat_path, res_path);

	SceneObject* ditch_bar = new SceneObject(m_context);
	ditch_bar->m_renderObjFilename = "/ditch_bar.obj";
	ditch_bar->initGraphics(mesh_path, mat_path, res_path);

	SceneObject* side_floor = new SceneObject(m_context);
	side_floor->m_renderObjFilename = "/side_floor.obj";
	side_floor->initGraphics(mesh_path, mat_path, res_path);

	EmissiveObject* sample_light = new EmissiveObject(m_context);
	sample_light->m_renderObjFilename = "/sample_light.obj";
	sample_light->initGraphics(mesh_path, mat_path, res_path);
	areaLights.push_back(sample_light->createAreaLight());

	btDbvtBroadphase* broadPhase = new btDbvtBroadphase();

	btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

	world = new btDiscreteDynamicsWorld(dispatcher, broadPhase, solver, collisionConfiguration);
	for (int i = 0; i < sceneObjects.size(); i++) {
		PhysicalObject* po = dynamic_cast<PhysicalObject*>(sceneObjects[i]);
		if (po) {
			world->addRigidBody(po->getRigidBody());
		}
	}

	g = m_context->createGroup();
	g->setChildCount(sceneObjects.size());

	for (int i = 0; i < sceneObjects.size(); i++) {
		g->setChild<Transform>(i, sceneObjects[i]->getTransform());
	}

	g->setAcceleration(m_context->createAcceleration("Bvh", "Bvh"));

	m_context["top_object"]->set(g);
	m_context["top_shadower"]->set(g);

	RectangleLight* areaLightArray = &areaLights[0];
	// add area lights to the scene
	Buffer areaLightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
	areaLightBuffer->setFormat(RT_FORMAT_USER);
	areaLightBuffer->setElementSize(sizeof(RectangleLight));
	areaLightBuffer->setSize(areaLights.size());
	memcpy(areaLightBuffer->map(), areaLightArray, sizeof(RectangleLight) * areaLights.size());
	areaLightBuffer->unmap();
	m_context["area_lights"]->set(areaLightBuffer);
}


void GlassScene::makeMaterialPrograms( Material material, const char *filename, 
                                                          const char *ch_program_name,
                                                          const char *ah_program_name )
{
  Program ch_program = m_context->createProgramFromPTXFile( ptxpath("glass", filename), ch_program_name );
  Program ah_program = m_context->createProgramFromPTXFile( ptxpath("glass", filename), ah_program_name );

  material->setClosestHitProgram( 0, ch_program );
  material->setAnyHitProgram( 1, ah_program );
}


//------------------------------------------------------------------------------
//
//  main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"

    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj-path <path>                    Specify path to OBJ files\n"
    << "  -A  | --adaptive-off                       Turn off adaptive AA\n"
    << "  -g  | --green                              Make the glass green\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  a Toggles adaptive pixel sampling on and off\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main(int argc, char* argv[])
{
	// btDbvtBroadphase* bp = new btDbvtBroadphase();
  GLUTDisplay::init( argc, argv );

  bool adaptive_aa = true;  // Default to true for now
  bool green_glass = false;
  std::string obj_path;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--adaptive-off" || arg == "-A" ) {
      adaptive_aa = false;
    } else if ( arg == "--green" || arg == "-g" ) {
      green_glass = true;
    } else if ( arg == "--obj-path" || arg == "-o" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      obj_path = argv[++i];
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  if( obj_path.empty() ) {
    obj_path = std::string( sutilSamplesDir() ) + "/glass";
  }

  try {
    GlassScene scene( obj_path, adaptive_aa, green_glass );
    GLUTDisplay::setTextColor( make_float3( 0.6f, 0.1f, 0.1f ) );
    GLUTDisplay::setTextShadowColor( make_float3( 0.9f ) );
	GLUTDisplay::run( "GlassScene", &scene, adaptive_aa ? GLUTDisplay::CDProgressive : GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
