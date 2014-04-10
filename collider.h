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
	Collider();

	void initPhysics(std::string obj_path);
	
	void setInitialPosition(btVector3 pos);

	void step();

	inline btRigidBody* getRigidBody() const { return m_rigidBody; };

	inline void setParent(SceneObject* p) { m_parent = p; };

	std::string m_physicsObjFilename;

	inline void setMass(int m) { m_mass = m; };
protected:
	SceneObject* m_parent;
	btRigidBody* m_rigidBody;
	btScalar m_mass;
};

class SphereCollider : public Collider {
public:
	void initPhysics(float radius);

protected:
	float m_radius;
};

#endif