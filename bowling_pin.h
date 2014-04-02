#ifndef BOWLING_PIN_
#define BOWLING_PIN_

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <ObjLoader.h>

#include <btBulletDynamicsCommon.h>
#include <btBulletCollisionCommon.h>
#include <BulletCollision\CollisionShapes\btShapeHull.h>
#include <ConvexDecomposition\ConvexDecomposition.h>
#include <ConvexDecomposition\cd_wavefront.h>

#include "HACD\hacdCircularList.h"
#include "HACD\hacdVector.h"
#include "HACD\hacdICHull.h"
#include "HACD\hacdGraph.h"
#include "HACD\hacdHACD.h"

#include <assert.h>

#include <fstream>
#include <sstream>

using namespace optix;

void printFloat3(float3 f) {
	printf("%.3f\t%.3f\t%.3f\n", f.x, f.y, f.z);
}

void printBtVector(btVector3 v) {
	printf("%.3f\t%.3f\t%.3f\n", v.x(), v.y(), v.z());
}

class SceneObject {
public:
	SceneObject() {
		m_initialTransformMtx = NULL;

		m_emissive = false;

		m_ke = make_float3(0, 0, 0);
		m_ka = make_float3(0.2, 0.2, 0.2);
		m_kd = make_float3(0.5, 0.5, 0.5);
		m_ks = make_float3(0.3, 0.3, 0.3);
		m_kr = make_float3(0, 0, 0);
		m_ns = 10;
	}

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

	void parseMtlFile(Material mat, std::string mtl_path) {
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
			}
		}
	}

	virtual void initGraphics(std::string obj_path, std::string mtl_path) {
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

	inline Transform getTransform() const { return m_transform; };
	inline void setTransformMatrix(float m[]) { m_transform->setMatrix(false, m, NULL); };

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
};

Context SceneObject::context = NULL;
Program SceneObject::closest_hit = NULL;
Program SceneObject::any_hit= NULL;
Program SceneObject::mesh_intersect = NULL;
Program SceneObject::mesh_bounds = NULL;

class Collider {
public:
	Collider(Context c) : m_mass(0) {
	}

	virtual void initPhysics(std::string obj_path) {
		ConvexDecomposition::WavefrontObj wo;
		wo.loadObj((obj_path + m_physicsObjFilename).c_str());

		btTriangleMesh* triMesh = new btTriangleMesh();

		for (int i = 0; i < wo.mTriCount; i++) {
			int index0 = wo.mIndices[i * 3];
			int index1 = wo.mIndices[i * 3 + 1];
			int index2 = wo.mIndices[i * 3 + 2];

			btVector3 vertex0(wo.mVertices[index0 * 3], wo.mVertices[index0 * 3 + 1], 
				wo.mVertices[index0 * 3 + 2]);
			btVector3 vertex1(wo.mVertices[index1 * 3], wo.mVertices[index1 * 3 + 1], 
				wo.mVertices[index1 * 3 + 2]);
			btVector3 vertex2(wo.mVertices[index2 * 3], wo.mVertices[index2 * 3 + 1], 
				wo.mVertices[index2 * 3 + 2]);

			triMesh->addTriangle(vertex0, vertex1, vertex2);
		}

		ConvexDecomposition::DecompDesc desc;

		desc.mVcount = wo.mVertexCount;
		desc.mTcount = wo.mTriCount;

		desc.mVertices = wo.mVertices;
		desc.mIndices = (unsigned int *)wo.mIndices;

		desc.mDepth = 5;
		desc.mCpercent = 5;
		desc.mPpercent = 15;
		desc.mMaxVertices = 16;
		desc.mSkinWidth = 0;

		std::vector<HACD::Vec3<HACD::Real>> points;
		std::vector<HACD::Vec3<long>> triangles;

		for (int i = 0; i < wo.mVertexCount; i++) {
			HACD::Vec3<HACD::Real> v(wo.mVertices[i * 3], wo.mVertices[i * 3 + 1], wo.mVertices[i * 3 + 2]);
			points.push_back(v);
		}

		for (int i = 0; i < wo.mTriCount; i++) {
			HACD::Vec3<long> t(wo.mIndices[i * 3], wo.mIndices[i * 3 + 1], wo.mIndices[i * 3 + 2]);
			triangles.push_back(t);
		}

		HACD::HACD hacd;
		hacd.SetPoints(&points[0]);
		hacd.SetNPoints(points.size());
		hacd.SetTriangles(&triangles[0]);
		hacd.SetNTriangles(triangles.size());
		hacd.SetCompacityWeight(0.1);
		hacd.SetVolumeWeight(0);
		hacd.SetNClusters(2);
		hacd.SetNVerticesPerCH(100);
		hacd.SetConcavity(100);
		hacd.SetAddExtraDistPoints(false);
		hacd.SetAddNeighboursDistPoints(false);
		hacd.SetAddFacesPoints(false);
		hacd.Compute();

		btCompoundShape* compound = new btCompoundShape();

		for (int i = 0; i < hacd.GetNClusters(); i++) { // for each cluster from HACD
			int numPoints = hacd.GetNPointsCH(i);
			int numTriangles = hacd.GetNTrianglesCH(i);

			float* vertices = new float[numPoints * 3];
			unsigned int* triangles = new unsigned int[numTriangles * 3];

			HACD::Vec3<HACD::Real>* pointsCH = new HACD::Vec3<HACD::Real>[numPoints];
			HACD::Vec3<long>* trianglesCH = new HACD::Vec3<long>[numTriangles];
			hacd.GetCH(i, pointsCH, trianglesCH);

			for (int j = 0; j < numPoints; j++) {
				vertices[3 * j] = pointsCH[j].X();
				vertices[3 * j + 1] = pointsCH[j].Y();
				vertices[3 * j + 2] = pointsCH[j].Z();
			}

			for (int j = 0; j < numTriangles; j++) {
				triangles[3 * j] = trianglesCH[j].X();
				triangles[3 * j + 1] = trianglesCH[j].Y();
				triangles[3 * j + 2] = trianglesCH[j].Z();
			}

			ConvexDecomposition::ConvexResult result(numPoints, vertices, numTriangles, triangles);

			// find the center for each part
			btVector3 centeroid(0, 0, 0);

			// compute the center
			for (unsigned int j = 0; j < result.mHullVcount; j++) {
				btVector3 vertex(result.mHullVertices[j * 3], result.mHullVertices[j * 3 + 1],
					result.mHullVertices[j * 3 + 2]);
				centeroid += vertex;
			}

			centeroid /= float(result.mHullVcount);

			btAlignedObjectArray<btVector3> vArr;

			// computer relative position
			for (unsigned int j = 0; j < result.mHullVcount; j++) {
				btVector3 vertex(result.mHullVertices[j * 3], result.mHullVertices[j * 3 + 1],
					result.mHullVertices[j * 3 + 2]);
				vertex -= centeroid;
				vArr.push_back(vertex);
			}

			// create convex shape
			btConvexHullShape* shape = new btConvexHullShape(&(vArr[0].getX()), vArr.size());
			shape->setMargin(0.01f);

			btTransform trans;
			trans.setIdentity();
			trans.setOrigin(centeroid);

			compound->addChildShape(trans, shape);
		}

		btDefaultMotionState* state = new btDefaultMotionState(
			btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btVector3 inertia(0, 0, 0);
		if (m_mass > 0) {
			compound->calculateLocalInertia(m_mass, inertia);
		}

		btRigidBody::btRigidBodyConstructionInfo info(m_mass, state, compound, inertia);

		m_rigidBody = new btRigidBody(info);
	}
	
	void setInitialPosition(float3 pos) {
		btTransform t;
		t.setIdentity();
		t.setOrigin(btVector3(pos.x, pos.y, pos.z));
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

		// calculate transformation matrix according to quaternion
		float m[] = {
			1.0f - 2.0f*qy*qy - 2.0f*qz*qz,		2.0f*qx*qy - 2.0f*qz*qw,			2.0f*qx*qz + 2.0f*qy*qw,			tx,
			2.0f*qx*qy + 2.0f*qz*qw,			1.0f - 2.0f*qx*qx - 2.0f*qz*qz,		2.0f*qy*qz - 2.0f*qx*qw,			ty,
			2.0f*qx*qz - 2.0f*qy*qw,			2.0f*qy*qz + 2.0f*qx*qw,			1.0f - 2.0f*qx*qx - 2.0f*qy*qy,		tz,
			0.0f,								0.0f,								0.0f,								1.0f
		};

		parent->setTransformMatrix(m);
	}

	inline btRigidBody* getRigidBody() const { return m_rigidBody; }

	std::string m_physicsObjFilename;
protected:
	SceneObject* parent;
	btRigidBody* m_rigidBody;
	btScalar m_mass;
};

class Ball : public Collider {
public:
	Ball(Context c) : Collider(c) {
		m_mass = 6;

		/*
		m_kr = make_float3(0.3, 0.3, 0.3);
		m_ns = 10;

		m_renderObjFilename = "/bowling_ball.obj";
		*/
	}

	virtual void initPhysics(std::string obj_path) {
		btCollisionShape* sphereShape = new btSphereShape(1);
		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btVector3 inertia(0, 0, 0);
		sphereShape->calculateLocalInertia(m_mass, inertia);

		btRigidBody::btRigidBodyConstructionInfo info(m_mass, state, sphereShape, inertia);

		m_rigidBody = new btRigidBody(info);
	}
};

class BowlingPin : public Collider {
public:
	BowlingPin(Context c) : Collider(c) {
		m_mass = 1;

		/*
		m_kr = make_float3(0, 0, 0);
		m_ns = 10;

		m_renderObjFilename = "/pin-new.obj";
		m_physicsObjFilename = "/pin-phy.obj";
		m_diffuseMapFilename = "/pin-diffuse.ppm";
		// m_normalMapFilename = "/brick_normal.ppm";
		// m_specularMapFilename = "/brick_specular.ppm";
		*/
	}

	/*
	virtual void initPhysics(std::string obj_path) {
		btCollisionShape* cylinderShape = new btCylinderShape(btVector3(0.44, 1.2, 1));
		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btVector3 inertia(0, 0, 0);
		cylinderShape->calculateLocalInertia(1, inertia);

		btRigidBody::btRigidBodyConstructionInfo info(1, state, cylinderShape, inertia);

		m_rigidBody = new btRigidBody(info);
		m_rigidBody->setFriction(0.3);
	}
	*/

private:
};

class GroundPlane : public Collider {
public:
	GroundPlane(Context c) : Collider(c) {
		m_mass = 0;

		/*
		m_ka = make_float3(0.2, 0.2, 0.2);
		m_kr = make_float3(0.5, 0.5, 0.5);
		m_ns = 5;

		m_renderObjFilename = "/lane-4.obj";
		m_physicsObjFilename = "/lane-3.obj";
		m_diffuseMapFilename = "/wood_floor_diffuse.ppm";
		// m_normalMapFilename = "/wood_floor_normal.ppm";
		// m_specularMapFilename = "/wood_floor_specular.ppm";
		*/
	}

	/*
	virtual void initPhysics(std::string obj_path) {
		btCollisionShape* boxShape = new btBoxShape(btVector3(100, 1, 5));
		btDefaultMotionState* state = new btDefaultMotionState(
			btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btRigidBody::btRigidBodyConstructionInfo info(0, state, boxShape, btVector3(0, 0, 0));

		btTransform trans;
		trans.setIdentity();
		trans.setOrigin(btVector3(0, -0.5, 5));
		m_rigidBody = new btRigidBody(info);
		m_rigidBody->setWorldTransform(trans);
	}
	*/

private:
};

#endif