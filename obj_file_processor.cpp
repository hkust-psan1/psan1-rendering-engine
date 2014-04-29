#include "obj_file_processor.h"

const float pinRadius = 0.44; // from the obj file
const float pinDistance = 0.44 * 12 / (4.75 / 2); // 12 inches distance with 4.75 inches diameter
const float unitX = pinDistance * 1.73205 / 2;
const float unitZ = pinDistance / 2;

const float3 pinBasePosition = make_float3(10, 0, 0);
const float3 pinPositions[10] = {
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

std::vector<SceneObject*> ObjFileProcessor::processObject(std::string filename, std::string targetDir) {
	processObjFile(filename + ".obj", targetDir);
	processMtlFile(filename + ".mtl", targetDir);

	std::vector<SceneObject*> sceneObjects;

	int pin_index = 0;

	for (auto it = objMtlMap.begin(); it != objMtlMap.end(); it++) {
		std::cout << "Processing object: " << it->first << std::endl;

		SceneObject* so = new SceneObject();
		so->initGraphics(targetDir + it->first, targetDir + it->second, targetDir + "textures/");
		sceneObjects.push_back(so);

		// used for the kitchen scene
		/*
		if (it->first.find("Suzanne") != std::string::npos) {
			Collider* c = new Collider;
			c->setMass(1);
			c->initPhysics(targetDir + it->first);
			c->setInitialPosition(btVector3(0, 10, 0));
			so->attachCollider(c);
		} else if (it->first.find("Floor") != std::string::npos) {
			Collider* c = new Collider;
			c->setMass(0);
			c->initPhysics(targetDir + it->first);
			so->attachCollider(c);
		}
		*/

		// used for the bowling scene
		if (it->first.find("Pin") != std::string::npos) {
			Collider* c = new Collider;
			c->setMass(1);
			c->initPhysics(targetDir + "../pin-phy.obj");
			float3 pos = pinPositions[pin_index++];
			c->setInitialPosition(btVector3(pos.x, pos.y, pos.z));
			so->attachCollider(c);
		} else if (it->first.find("MainFloor") != std::string::npos) {
			Collider* c = new Collider;
			c->setMass(0);
			c->initPhysics(targetDir + it->first);
			so->attachCollider(c);
		} else if (it->first.find("BowlingBall") != std::string::npos) {
			SphereCollider* sc = new SphereCollider;
			sc->setMass(5);
			sc->initPhysics(1);
			sc->setInitialPosition(btVector3(-10, 2, 0));
			sc->getRigidBody()->setLinearVelocity(btVector3(10, 0, 0.1));
			so->attachCollider(sc);
		}

		// used for the throw scene
		/*
		if (it->first.find("Ground") != std::string::npos) {
			Collider* c = new Collider;
			c->setMass(0);
			c->initPhysics(targetDir + it->first);
			so->attachCollider(c);
		} else if (it->first.find("Base") != std::string::npos) {
			Collider* c = new Collider;
			c->setMass(10);
			c->initPhysics(targetDir + it->first);
			so->attachCollider(c);
		} else if (it->first.find("Pole") != std::string::npos) {
			Collider* c = new Collider;
			c->setMass(1);
			c->initPhysics(targetDir + it->first);
			c->setInitialPosition(btVector3(0, 2.1, 0));
			so->attachCollider(c);
		}
		*/
	}

	return sceneObjects;
}

void ObjFileProcessor::processMtlFile(std::string filename, std::string targetDir) {
	std::ifstream mtlInput(filename.c_str());
	std::ofstream output;
	std::string mtlName;

	for (std::string line; getline(mtlInput, line); ) {
		std::stringstream ss(line);
		std::string part;
		getline(ss, part, ' ');

		if (part == "newmtl") {
			getline(ss, part, '\n');
			mtlName = part + ".mtl";

			if (output) {
				output.close();
			}

			output.open((targetDir + mtlName).c_str());
		}

		if (output) {
			output << line << std::endl;
		}
	}
	output.close();
}

void ObjFileProcessor::processObjFile(std::string filename, std::string targetDir) {
	std::ifstream objInput(filename.c_str());
	std::ofstream output;
	std::string objName;

	int vCount = 0, nCount = 0, tCount = 0;
	int vOffset, nOffset, tOffset;

	for (std::string line; getline(objInput, line); ) {
		std::stringstream ss(line);
		std::string part;
		getline(ss, part, ' ');

		if (part == "o") {
			vOffset = vCount;
			nOffset = nCount;
			tOffset = tCount;

			getline(ss, part, '\n');
			objName = part + ".obj";

			printf("parsing: %s\n", part.c_str());

			if (output) { // output stream already initialized
				output.close();
			}

			output.open((targetDir + objName).c_str());
			output << line << std::endl;
		} else if (part == "v") {
			vCount++;
		} else if (part == "vn") {
			nCount++;
		} else if (part == "vt") {
			tCount++;
		} else if (part == "f") {
			// "part" is the line excluding leading "f"
			getline(ss, part, '\n');

			// each vSegment stands for a vertex in the format X/Y/Z
			std::string vSegment;

			// create a stringstream to handle "part", a series of vSegment
			std::stringstream vStream(part);

			// reset "line" to "f " and later add new face data to it
			line = "";
			std::stringstream newFaceStream;
			newFaceStream << "f ";

			while (getline(vStream, vSegment, ' ')) { // find all vertices of the face
				std::string vStr, tStr, nStr;

				// a stringstream for each vertex data item X/Y/Z
				std::stringstream segmentStream(vSegment);

				getline(segmentStream, vStr, '/');
				getline(segmentStream, tStr, '/');
				getline(segmentStream, nStr, '/');

				if (nStr.empty()) { // cannot find normal vector
					throw "normal vector not supplied";
				}

				int vIndex = atoi(vStr.c_str()) - vOffset;
				int nIndex = atoi(nStr.c_str()) - nOffset;
				int tIndex = -1;

				if (!tStr.empty()) { // texture coord provided
					tIndex = atoi(tStr.c_str()) - tOffset;
				}

				newFaceStream << vIndex << '/';
				if (tIndex != -1) { // texture coord provided
					newFaceStream << tIndex;
				}
				newFaceStream << '/';
				newFaceStream << nIndex << ' ';
			}

			line = newFaceStream.str();
		} else if (part == "usemtl") {
			getline(ss, part, '\n');
			std::string mtlFilename = targetDir + part + ".mtl";

			// insert into the hashmap
			objMtlMap.insert(std::pair<std::string, std::string>(objName, part + ".mtl"));
		}

		if (output) { // output stream already initialized
			output << line << std::endl;
		}
	}
	output.close();
}