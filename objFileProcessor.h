#ifndef OBJ_FILE_PROCESSOR_
#define OBJ_FILE_PROCESSOR_

#include <fstream>
#include <ObjLoader.h>

class ObjFileProcessor {
public:
	void processObjFile(std::string filename, std::string targetDir) {
		std::ifstream objInput(filename.c_str());


	}
};

#endif