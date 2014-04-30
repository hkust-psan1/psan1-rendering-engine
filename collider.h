#ifndef COLLIDER_
#define COLLIDER_

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

class SceneObject;

class Collider {
public:
	Collider(bool f = false);

	void initPhysics(std::string obj_path);
	
	void setInitialPosition(btVector3 pos);

	void setInitialRotation(btQuaternion q);

	void step();

	inline btRigidBody* getRigidBody() const { return m_rigidBody; };

	inline void setParent(SceneObject* p) { m_parent = p; };

	std::string m_physicsObjFilename;

	inline void setMass(int m) { m_mass = m; };
	static float focus_x;
	static float focus_y;
	static float focus_z;
protected:
	SceneObject* m_parent;
	btRigidBody* m_rigidBody;
	btScalar m_mass;
	bool focus;
};

class SphereCollider : public Collider {
public:
	
	SphereCollider(bool f = false) : Collider(f){};
	void initPhysics(float radius);

protected:
	float m_radius;
};

class CylinderCollider : public Collider {
public:
	CylinderCollider() {
		m_mass = 1;
	}

	void initPhysics(float radius, float halfHeight) {
		btCollisionShape* cylinderShape = new btCylinderShape(btVector3(radius, halfHeight, radius));
		btDefaultMotionState* state = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));

		btVector3 inertia(0, 0, 0);
		cylinderShape->calculateLocalInertia(m_mass, inertia);

		btRigidBody::btRigidBodyConstructionInfo info(m_mass, state, cylinderShape, inertia);

		m_rigidBody = new btRigidBody(info);
	}
};

#endif