#ifndef BOWLING_PIN_
#define BOWLING_PIN_

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <ObjLoader.h>

#include <btBulletDynamicsCommon.h>
#include <btBulletCollisionCommon.h>

#include <cd_wavefront.h>

#include <assert.h>

using namespace optix;

class SceneObject {
public:
	SceneObject(Context c) : m_context(c), m_mass(0) {
		m_initialTransformMtx = NULL;

		m_emissive = false;

		m_ke = make_float3(0, 0, 0);
		m_ka = make_float3(0.2, 0.2, 0.2);
		m_kd = make_float3(0.5, 0.5, 0.5);
		m_ks = make_float3(0.8, 0.8, 0.8);
		m_kr = make_float3(0, 0, 0);
		m_ns = 10;

		// m_normalMapFilename = "/default_normal.ppm";
		// m_specularMapFilename = "/default_specular.ppm";
	}

	virtual void initGraphics(std::string prog_path, std::string mat_path, std::string res_path) {
		m_objPath = res_path + m_renderObjFilename;

		Material mat = m_context->createMaterial();

		Program closestHit = m_context->createProgramFromPTXFile(mat_path, "closest_hit_radiance");
		Program anyHit = m_context->createProgramFromPTXFile(mat_path, "any_hit_shadow");

		mat->setClosestHitProgram(0, closestHit);
		mat->setAnyHitProgram(1, anyHit);

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

		mat["kd_map"]->setTextureSampler(loadTexture(m_context, 
			"D:/OptiX SDK 3.0.1/SDK - Copy/glass/" + m_diffuseMapFilename, 
			make_float3(1, 1, 1)));
		mat["ks_map"]->setTextureSampler(loadTexture(m_context, 
			"D:/OptiX SDK 3.0.1/SDK - Copy/glass/" + m_specularMapFilename, 
			make_float3(1, 1, 1)));
		mat["normal_map"]->setTextureSampler(loadTexture(m_context, 
			"D:/OptiX SDK 3.0.1/SDK - Copy/glass/" + m_normalMapFilename, 
			make_float3(1, 1, 1)));

		mat["has_diffuse_map"]->setInt(!m_diffuseMapFilename.empty());
		mat["has_normal_map"]->setInt(!m_normalMapFilename.empty());
		mat["has_specular_map"]->setInt(!m_specularMapFilename.empty());

		Program mesh_intersect = m_context->createProgramFromPTXFile(prog_path, "mesh_intersect");
		Program mesh_bounds = m_context->createProgramFromPTXFile(prog_path, "mesh_bounds");

		GeometryGroup group = m_context->createGeometryGroup();

		ObjLoader loader(m_objPath.c_str(), m_context, group, mat);
		loader.setIntersectProgram(mesh_intersect);
		loader.setBboxProgram(mesh_bounds);
		loader.load();

		m_transform = m_context->createTransform();
		/*
		if (m_initialTransformMtx != NULL) {
			m_transform->setMatrix(false, m_initialTransformMtx, NULL);
		}
		*/
		m_transform->setChild(group);
	}

	inline Transform getTransform() const { return m_transform; };

	/*
	inline void setInitialTransform(float* t) { m_initialTransformMtx = t; };

	void setInitialOriginPosition(float x, float y, float z) {
		float m[] = {
			1, 0, 0, x,
			0, 1, 0, y,
			0, 0, 0, z,
			0, 0, 0, 1,
		};
		setInitialTransform(m);
	}
	*/


	btScalar m_mass;

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

protected:
	Context m_context;
	Transform m_transform;
	float* m_initialTransformMtx;

	std::string m_objPath;
};

class EmissiveObject : public SceneObject {
public:
	EmissiveObject(Context c) : SceneObject(c) {
		m_emissive = true;
	};

	RectangleLight createAreaLight() {
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
};

class PhysicalObject : public SceneObject {
public:
	PhysicalObject(Context c) : SceneObject(c) {
	}

	virtual void initPhysics(std::string prog_path) {
		ConvexDecomposition::WavefrontObj wo;
		wo.loadObj((prog_path + m_physicsObjFilename).c_str());

		btTriangleMesh* triMesh = new btTriangleMesh();

		for (int i = 0; i < wo.mTriCount; i++) {
			int index0 = wo.mIndices[i * 3];
			int index1 = wo.mIndices[i * 3 + 1];
			int index2 = wo.mIndices[i * 3 + 2];

			btVector3 vertex0(wo.mVertices[index0 * 3], wo.mVertices[index0 * 3 + 1], wo.mVertices[index0 * 3 + 2]);
			btVector3 vertex1(wo.mVertices[index1 * 3], wo.mVertices[index1 * 3 + 1], wo.mVertices[index1 * 3 + 2]);
			btVector3 vertex2(wo.mVertices[index2 * 3], wo.mVertices[index2 * 3 + 1], wo.mVertices[index2 * 3 + 2]);

			float z1 = wo.mVertices[index0 * 3 + 1];
			float z2 = wo.mVertices[index1 * 3 + 1];
			float z3 = wo.mVertices[index2 * 3 + 1];

			triMesh->addTriangle(vertex0, vertex1, vertex2);
		}

		btCollisionShape* pinCollisionShape = new btBvhTriangleMeshShape(triMesh, true);

		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 10, 0)));

		btVector3 inertia(0, 0, 0);
		// pinCollisionShape->calculateLocalInertia(mass, inertia);
		btRigidBody::btRigidBodyConstructionInfo info(m_mass, state, pinCollisionShape, inertia);

		m_rigidBody = new btRigidBody(info);
	}
	
	void setInitialPosition(float x, float y, float z) {
		btTransform t;
		t.setIdentity();
		t.setOrigin(btVector3(x, y, z));
		m_rigidBody->setWorldTransform(t);
	}

	void step() {
		btTransform trans;
		m_rigidBody->getMotionState()->getWorldTransform(trans);

		btVector3 origin = trans.getOrigin();
		float tx = origin.getX();
		float ty = origin.getY();
		float tz = origin.getZ();

		btQuaternion quaternion = trans.getRotation();
		float qx = quaternion.getX();
		float qy = quaternion.getY();
		float qz = quaternion.getZ();
		float qw = quaternion.getW();

		float m[] = {
			1.0f - 2.0f*qy*qy - 2.0f*qz*qz,		2.0f*qx*qy - 2.0f*qz*qw,			2.0f*qx*qz + 2.0f*qy*qw,			tx,
			2.0f*qx*qy + 2.0f*qz*qw,			1.0f - 2.0f*qx*qx - 2.0f*qz*qz,		2.0f*qy*qz - 2.0f*qx*qw,			ty,
			2.0f*qx*qz - 2.0f*qy*qw,			2.0f*qy*qz + 2.0f*qx*qw,			1.0f - 2.0f*qx*qx - 2.0f*qy*qy,		tz,
			0.0f,								0.0f,								0.0f,								1.0f
		};

		m_transform->setMatrix(false, m, NULL);
	}

	inline btRigidBody* getRigidBody() const { return m_rigidBody; }

	std::string m_physicsObjFilename;
protected:
	btRigidBody* m_rigidBody;
};

class Ball : public PhysicalObject {
public:
	Ball(Context c) : PhysicalObject(c) {
		m_mass = 3;

		m_kr = make_float3(0.3, 0.3, 0.3);
		m_ns = 10;

		m_renderObjFilename = "/bowling_ball.obj";
	}

	virtual void initPhysics(std::string prog_path) {
		btCollisionShape* sphereShape = new btSphereShape(1);
		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btVector3 inertia(0, 0, 0);
		sphereShape->calculateLocalInertia(1, inertia);

		btRigidBody::btRigidBodyConstructionInfo info(1, state, sphereShape, inertia);

		m_rigidBody = new btRigidBody(info);
	}
};

class BowlingPin : public PhysicalObject {
public:
	BowlingPin(Context c) : PhysicalObject(c) {
		m_mass = 1;

		m_kr = make_float3(0, 0, 0);
		m_ns = 10;

		m_renderObjFilename = "/pin.obj";
		m_physicsObjFilename = "/pin-phy.obj";
		m_diffuseMapFilename = "/pin-diffuse.ppm";
		// m_normalMapFilename = "/brick_normal.ppm";
		// m_specularMapFilename = "/brick_specular.ppm";
	}

	virtual void initPhysics(std::string prog_path) {
		btCollisionShape* cylinderShape = new btCylinderShape(btVector3(0.44, 1.2, 1));
		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btVector3 inertia(0, 0, 0);
		cylinderShape->calculateLocalInertia(1, inertia);

		btRigidBody::btRigidBodyConstructionInfo info(1, state, cylinderShape, inertia);

		m_rigidBody = new btRigidBody(info);
	}

private:
};

class GroundPlane : public PhysicalObject {
public:
	GroundPlane(Context c) : PhysicalObject(c) {
		m_mass = 0;

		m_ka = make_float3(0.2, 0.2, 0.2);
		m_kr = make_float3(0.3, 0.3, 0.3);
		m_ns = 5;

		m_renderObjFilename = "/lane.obj";
		m_physicsObjFilename = "/bowling-floor.obj";
		m_diffuseMapFilename = "/wood_floor_diffuse.ppm";
		// m_normalMapFilename = "/wood_floor_normal.ppm";
		// m_specularMapFilename = "/wood_floor_specular.ppm";
	}

	virtual void initPhysics(std::string prog_path) {
		btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 0);

		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btRigidBody::btRigidBodyConstructionInfo info(0, state, groundShape, btVector3(0, 0, 0));

		m_rigidBody = new btRigidBody(info);
	}

private:
};



#endif