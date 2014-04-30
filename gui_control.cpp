#include "scene.h"
#include "gui_control.h"

#define MAX_SAMPLES_PER_FRAME 200

Scene* GUIControl::scene = NULL;

bool GUIControl::onAnimation = false;
bool GUIControl::initialStep = true;
float GUIControl::ballVelocityZ = 0;
float GUIControl::cameraFocalScale = 0.5;

bool GUIControl::dofOn = false;
bool GUIControl::softShadowOn = false;
bool GUIControl::glossyOn = false;
bool GUIControl::aaOn = false;
bool GUIControl::motionBlurOn = false;
bool GUIControl::giOn = false;

Fl_Button* GUIControl::startButton;
Fl_Button* GUIControl::pauseButton;
Fl_Button* GUIControl::recordButton;
Fl_Button* GUIControl::snapShotButton;

Fl_Value_Slider* GUIControl::ballVelocityZSlider;
Fl_Value_Slider* GUIControl::cameraFocalScaleSlider;

Fl_Light_Button* GUIControl::dofLightButton;
Fl_Light_Button* GUIControl::softShadowLightButton;
Fl_Light_Button* GUIControl::glossyLightButton;
Fl_Light_Button* GUIControl::aaLightButton;
Fl_Light_Button* GUIControl::motionBlurLightButton;
Fl_Light_Button* GUIControl::giLightButton;

Fl_Button* GUIControl::cameraChangeButton;
Fl_Value_Slider* GUIControl::samplesPerFrameSlider;

void GUIControl::startButtonPressed() {
	if (!scene) {
		return;
	}

	if (initialStep) { // at initial step, start the animation
		startButton->label("reset");
		startButton->deactivate();

		pauseButton->label("pause");
		pauseButton->activate();

		// ballVelocityZSlider->deactivate();

		onAnimation = true;
		initialStep = false;

		// scene->ball->getRigidBody()->setLinearVelocity(btVector3(20, 0, ballVelocityZ));
	} else { // reset
		startButton->label("start");

		pauseButton->deactivate();

		// ballVelocityZSlider->activate();

		onAnimation = false;
		initialStep = true;

		scene->resetObjects();

		// scene->ball->getRigidBody()->setLinearVelocity(btVector3(0, 0, 0));
	}
}

void GUIControl::pauseButtonPressed() {
	if (!scene) {
		return;
	}

	if (onAnimation) { // animation going on, pause the animation
		pauseButton->label("resume");
		startButton->activate();

		onAnimation = false;
	} else { // resume the animation
		pauseButton->label("pause");
		startButton->deactivate();

		onAnimation = true;
	}
}

void GUIControl::ballVelocityZSliderValueChanged() {
	ballVelocityZ = ballVelocityZSlider->value();
}

void GUIControl::cameraFocalScaleChanged() {
	cameraFocalScale = cameraFocalScaleSlider->value();
}

void GUIControl::recordButtonPressed() {
	if (!scene) {
		return;
	}

	printf("presseed");

	if (!scene->m_recording) {
		recordButton->label("stop recording");
		scene->m_recording = true;
		std::string filename = "D:\\tracer_output.avi";
		const int format = CV_FOURCC('D', 'I', 'V', 'X');

		scene->writer.open(filename, format, 100, cv::Size(scene->WIDTH, scene->HEIGHT), true);
		if (!scene->writer.isOpened())
		{
			assert(false);
		}
	} else {
		recordButton->label("record");
		scene->m_recording = false;
		scene->writer.release();
	}
}

void GUIControl::snapShotButtonPressed() {
	if (!scene) {
		return;
	}

	snapShotButton->deactivate();

	scene->m_taking_snapshot = true;

	scene->m_dof_sample_num = 0;
	scene->m_jitter_base_x = 0;
	scene->m_jitter_base_y = 0;
}

void GUIControl::dofLightButtonPressed() {
	dofOn = !dofOn;
	if (dofOn) {
		cameraFocalScaleSlider->activate();
	} else {
		cameraFocalScaleSlider->deactivate();
	}
}

void GUIControl::softShadowLightButtonPressed() {
	softShadowOn = !softShadowOn;
}

void GUIControl::glossyLightButtonPressed() {
	glossyOn = !glossyOn;
}

void GUIControl::aaLightButtonPressed() {
	aaOn = !aaOn;
}

void GUIControl::motionBlurLightButtonPressed() {
	motionBlurOn = !motionBlurOn;
}

void GUIControl::giLightButtonPressed() {
	giOn = !giOn;
}

void GUIControl::cameraChangeButtonPressed() {
	scene->m_camera_type = (scene->m_camera_type + 1) % 3;
	scene->m_camera_changed = true;
	printf("new camera type: %d\n", scene->m_camera_type);
}

void GUIControl::samplesPerFrameChanged() {
	scene->samplesPerFrame = samplesPerFrameSlider->value();

	if (scene->samplesPerFrame == MAX_SAMPLES_PER_FRAME) {
		scene->samplesPerFrame = 999999;
	}
}

void GUIControl::showControlDialog() {
	Fl_Window* window = new Fl_Window(300, 700);

	window->begin();

	startButton = new Fl_Button(0, 0, 300, 30, "start");
	startButton->callback((Fl_Callback*) startButtonPressed);

	pauseButton = new Fl_Button(0, 40, 300, 30, "pause");
	pauseButton->callback((Fl_Callback*) pauseButtonPressed);
	pauseButton->deactivate();

	/*
	Fl_Box* ballVelocityLabel = new Fl_Box(0, 80, 300, 20, "ball velocity on Z direction");

	ballVelocityZSlider = new Fl_Value_Slider(0, 110, 300, 30);
	ballVelocityZSlider->type(FL_HOR_SLIDER);
	ballVelocityZSlider->bounds(-1, 1);
	ballVelocityZSlider->value(0);
	ballVelocityZSlider->callback((Fl_Callback*) ballVelocityZSliderValueChanged);
	*/

	Fl_Box* cameraFocalLengthLabel = new Fl_Box(0, 150, 300, 20, "camera focal length");

	cameraFocalScaleSlider = new Fl_Value_Slider(0, 180, 300, 30);
	cameraFocalScaleSlider->type(FL_HOR_SLIDER);
	cameraFocalScaleSlider->bounds(0.2, 5);
	cameraFocalScaleSlider->value(0.5);
	cameraFocalScaleSlider->callback((Fl_Callback*) cameraFocalScaleChanged);
	cameraFocalScaleSlider->deactivate();

	recordButton = new Fl_Button(0, 220, 300, 30, "record");
	recordButton->callback((Fl_Callback*) recordButtonPressed);

	snapShotButton = new Fl_Button(0, 260, 300, 30, "take snapshot");
	snapShotButton->callback((Fl_Callback*) snapShotButtonPressed);

	dofLightButton = new Fl_Light_Button(0, 300, 300, 30, "depth of field");
	dofLightButton->callback((Fl_Callback*) dofLightButtonPressed);

	softShadowLightButton = new Fl_Light_Button(0, 340, 300, 30, "soft shadow");
	softShadowLightButton->callback((Fl_Callback*) softShadowLightButtonPressed);

	glossyLightButton = new Fl_Light_Button(0, 380, 300, 30, "glossiness");
	glossyLightButton->callback((Fl_Callback*) glossyLightButtonPressed);

	aaLightButton = new Fl_Light_Button(0, 420, 300, 30, "anti-aliasing");
	aaLightButton->callback((Fl_Callback*) aaLightButtonPressed);

	motionBlurLightButton = new Fl_Light_Button(0, 460, 300, 30, "motion blur");
	motionBlurLightButton->callback((Fl_Callback*) motionBlurLightButtonPressed);

	giLightButton = new Fl_Light_Button(0, 500, 300, 30, "global illumination");
	giLightButton->callback((Fl_Callback*) giLightButtonPressed);

	cameraChangeButton = new Fl_Button(0, 540, 300, 30, "change camera type");
	cameraChangeButton->callback((Fl_Callback*) cameraChangeButtonPressed);

	samplesPerFrameSlider = new Fl_Value_Slider(0, 580, 300, 30);
	samplesPerFrameSlider->type(FL_HOR_SLIDER);
	samplesPerFrameSlider->bounds(1, MAX_SAMPLES_PER_FRAME);
	samplesPerFrameSlider->value(20);
	samplesPerFrameSlider->callback((Fl_Callback*) samplesPerFrameChanged);

	window->end();
	window->show();

	Fl::run();
}