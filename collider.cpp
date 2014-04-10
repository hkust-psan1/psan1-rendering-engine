#include <assert.h>
#include <sstream>

#include "collider.h"
#include "scene_object.h"

Collider::Collider() : m_mass(0) {
}

void Collider::initPhysics(std::string obj_path) {
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
	m_rigidBody->setRestitution(0.6);
}

void Collider::setInitialPosition(btVector3 pos) {
	btTransform t;
	t.setIdentity();
	t.setOrigin(pos);
	m_rigidBody->setWorldTransform(t);
}

void Collider::step() {
	btTransform trans;
	m_rigidBody->getMotionState()->getWorldTransform(trans);

	btVector3 origin = trans.getOrigin();
	float tx = origin.getX();
	float ty = origin.getY();
	float tz = origin.getZ();

	// printf("%.3f\n", ty);

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

	m_parent->setTransformMatrix(m);
}

void SphereCollider::initPhysics(float radius) {
	btCollisionShape* sphereShape = new btSphereShape(1);
	btDefaultMotionState* state = new btDefaultMotionState(
		btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

	btVector3 inertia(0, 0, 0);

	sphereShape->calculateLocalInertia(m_mass, inertia);

	btRigidBody::btRigidBodyConstructionInfo info(m_mass, state, sphereShape, inertia);
	m_rigidBody = new btRigidBody(info);
}
