#ifndef BOWLING_PIN_
#define BOWLING_PIN_

#include "collider.h"

class Ball : public Collider {
public:
	Ball() : Collider() {
		m_mass = 15;
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

#endif