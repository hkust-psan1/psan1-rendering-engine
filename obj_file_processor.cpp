#include "obj_file_processor.h"

std::vector<SceneObject*> ObjFileProcessor::processObject(std::string filename, std::string targetDir) {
	processObjFile(filename + ".obj", targetDir);
	processMtlFile(filename + ".mtl", targetDir);

	std::vector<SceneObject*> sceneObjects;

	for (auto it = objMtlMap.begin(); it != objMtlMap.end(); it++) {
		std::cout << "Processing object: " << it->first << std::endl;

		SceneObject* so = new SceneObject();
		so->initGraphics(targetDir + it->first, targetDir + it->second, targetDir);
		sceneObjects.push_back(so);

		if (it->first.find("Cube")) {
			Collider* c = new Collider;
			c->setMass(1);
			c->initPhysics(targetDir + it->first);
			so->attachCollider(c);
		}
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