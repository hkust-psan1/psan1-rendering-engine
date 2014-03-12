#ifndef BOWLING_PIN_
#define BOWLING_PIN_

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <ObjLoader.h>

#include <btBulletDynamicsCommon.h>

#include <cd_wavefront.h>

using namespace optix;

class SceneObject {
public:
	SceneObject(Context c) : m_context(c), m_mass(0) {
	}

	void step() {
		btTransform trans;
		m_rigidBody->getMotionState()->getWorldTransform(trans);

		float m[] = {
			1, 0, 0, trans.getOrigin().getX(),
			0, 1, 0, trans.getOrigin().getY(),
			0, 0, 1, trans.getOrigin().getZ(),
			0, 0, 0, 1
		};

		m_transform->setMatrix(false, m, NULL);
	}

	virtual void initPhysics(std::string prog_path) {
		ConvexDecomposition::WavefrontObj wo;
		wo.loadObj((prog_path + m_physicsObjFilename).c_str());

		// printf("%d\n", wo.mTriCount);

		btTriangleMesh* triMesh = new btTriangleMesh();

		for (int i = 0; i < wo.mTriCount; i++) {
			int index0 = wo.mIndices[i * 3];
			int index1 = wo.mIndices[i * 3 + 1];
			int index2 = wo.mIndices[i * 3 + 2];

			btVector3 vertex0(wo.mVertices[index0 * 3], wo.mVertices[index0 * 3 + 1], wo.mVertices[index0 * 3 + 2]);
			btVector3 vertex1(wo.mVertices[index1 * 3], wo.mVertices[index1 * 3 + 1], wo.mVertices[index1 * 3 + 2]);
			btVector3 vertex2(wo.mVertices[index2 * 3], wo.mVertices[index2 * 3 + 1], wo.mVertices[index2 * 3 + 2]);

			triMesh->addTriangle(vertex0, vertex1, vertex2);
		}

		btCollisionShape* pinCollisionShape = new btBvhTriangleMeshShape(triMesh, true);

		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btVector3 inertia(0, 0, 0);
		// pinCollisionShape->calculateLocalInertia(mass, inertia);
		btRigidBody::btRigidBodyConstructionInfo info(m_mass, state, pinCollisionShape, inertia);

		m_rigidBody = new btRigidBody(info);
	}

	virtual void initGraphics(std::string prog_path, std::string mat_path, std::string res_path) {
		Material mat = m_context->createMaterial();

		Program closestHit = m_context->createProgramFromPTXFile(mat_path, "closest_hit_radiance");
		Program anyHit = m_context->createProgramFromPTXFile(mat_path, "any_hit_shadow");

		mat->setClosestHitProgram(0, closestHit);
		mat->setAnyHitProgram(1, anyHit);

		mat["ka"]->setFloat(0.2, 0.2, 0.2);
		mat["ks"]->setFloat(0.5, 0.5, 0.5);
		mat["kr"]->setFloat( 0.8f, 0.8f, 0.8f );
		mat["ns"]->setInt(32);
		mat["importance_cutoff"]->setFloat( 0.01f );
		mat["cutoff_color"]->setFloat( 0.2f, 0.2f, 0.2f );
		mat["reflection_maxdepth"]->setInt( 5 );
		mat["kd_map"]->setTextureSampler(loadTexture(m_context, 
			"D:/OptiX SDK 3.0.1/SDK - Copy/glass/" + m_diffuseMapFilename, 
			make_float3(1, 1, 1)));

		Program mesh_intersect = m_context->createProgramFromPTXFile(prog_path, "mesh_intersect");
		Program mesh_bounds = m_context->createProgramFromPTXFile(prog_path, "mesh_bounds");

		GeometryGroup group = m_context->createGeometryGroup();

		ObjLoader loader((res_path + m_renderObjFilename).c_str(), m_context, group, mat);
		loader.setIntersectProgram(mesh_intersect);
		loader.setBboxProgram(mesh_bounds);
		loader.load();

		m_transform = m_context->createTransform();
		m_transform->setChild(group);
	}

	inline btRigidBody* getRigidBody() const { return m_rigidBody; }

	inline Transform getTransform() const { return m_transform; };
protected:
	Context m_context;
	Transform m_transform;

	btRigidBody* m_rigidBody;

	btScalar m_mass;

	std::string m_renderObjFilename;
	std::string m_physicsObjFilename;
	std::string m_diffuseMapFilename;
};

class BowlingPin : public SceneObject {
public:
	BowlingPin(Context c) : SceneObject(c) {
		m_mass = 1;
		m_renderObjFilename = "/pin.obj";
		m_physicsObjFilename = "/pin-phy.obj";
		m_diffuseMapFilename = "/pin-diffuse.ppm";
	}

	virtual void initPhysics(std::string prog_path) {
		btCollisionShape* cylinderShape = new btCylinderShape(btVector3(1, 1, 1));
		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 5, 0)));

		btVector3 inertia(0, 0, 0);
		cylinderShape->calculateLocalInertia(1, inertia);

		btRigidBody::btRigidBodyConstructionInfo info(1, state, cylinderShape, inertia);

		m_rigidBody = new btRigidBody(info);
	}

private:
};

class GroundPlane : public SceneObject {
public:
	GroundPlane(Context c) : SceneObject(c) {
		m_mass = 0;
		m_renderObjFilename = "/bowling-floor.obj";
		m_physicsObjFilename = "/bowling-floor.obj";
		m_diffuseMapFilename = "/wood_floor.ppm";
	}

	virtual void initPhysics(std::string prog_path) {
		btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), -1);
		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -1, 0)));

		btRigidBody::btRigidBodyConstructionInfo info(0, state, groundShape, btVector3(0, 0, 0));

		m_rigidBody = new btRigidBody(info);
	}

private:
};

#endif