#ifndef OBJ_FILE_PROCESSOR_
#define OBJ_FILE_PROCESSOR_

#include <fstream>
#include <sstream>
#include <map>

#include "scene_object.h"
#include "collider.h"

class ObjFileProcessor {
public:
	std::vector<SceneObject*> processObject(std::string filename, std::string targetDir);
	void processMtlFile(std::string filename, std::string targetDir);
	void processObjFile(std::string filename, std::string targetDir);

private:
	std::vector<std::string> objLines;
	std::map<std::string, std::string> objMtlMap;
};

#endif