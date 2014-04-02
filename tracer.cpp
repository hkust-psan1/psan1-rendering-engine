#include <Windows.h>
#include <stdio.h>
#include <conio.h>

#include "gui_control.h"
#include "scene.h"

Scene* scene;

void printUsageAndExit( const std::string& argv0, bool doExit = true ) {
	std::cerr
		<< "Usage	: " << argv0 << " [options]\n"
		<< "App options:\n"

		<< "	-h	| --help															 Print this usage message\n"
		<< "	-o	| --obj-path <path>										Specify path to OBJ files\n"
		<< "	-A	| --adaptive-off											 Turn off adaptive AA\n"
		<< "	-g	| --green															Make the glass green\n"
		<< std::endl;
	GLUTDisplay::printUsage();

	std::cerr
		<< "App keystrokes:\n"
		<< "	a Toggles adaptive pixel sampling on and off\n"
		<< std::endl;

	if ( doExit ) exit(1);
}

DWORD WINAPI displayGUIControlWindow(LPVOID lpParam) {
	GUIControl::scene = scene;
	GUIControl::showControlDialog();
	return 0;
}

int main(int argc, char* argv[]) {
	srand(time(0));

	GLUTDisplay::init( argc, argv );

	std::string obj_path;
	for ( int i = 1; i < argc; ++i ) {
		std::string arg( argv[i] );
		if ( arg == "--obj-path" || arg == "-o" ) {
			if ( i == argc-1 ) {
				printUsageAndExit( argv[0] );
			}
			obj_path = argv[++i];
		} else if ( arg == "--help" || arg == "-h" ) {
			printUsageAndExit( argv[0] );
		} else {
			std::cerr << "Unknown option: '" << arg << "'\n";
			printUsageAndExit( argv[0] );
		}
	}

	if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

	if( obj_path.empty() )
	{
		obj_path = std::string( sutilSamplesDir() ) + "/tracer/res/";
	}

	try {
		scene = new Scene(obj_path, 2); // pinhole camera

		HANDLE hThread;
		DWORD threadID;

		// GUIControl::scene = scene;
		hThread = CreateThread(NULL, 0, displayGUIControlWindow, NULL, 0, &threadID);

		GLUTDisplay::setTextColor( make_float3( 0.6f, 0.1f, 0.1f ) );
		GLUTDisplay::setTextShadowColor( make_float3( 0.9f ) );
		GLUTDisplay::run( "Scene", scene, GLUTDisplay::CDProgressive );

	} catch( Exception& e ){
		sutilReportError( e.getErrorString().c_str() );
		exit(1);
	}
}
