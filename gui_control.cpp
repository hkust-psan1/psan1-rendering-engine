#include "scene.h"
#include "gui_control.h"

Scene* GUIControl::scene = NULL;

bool GUIControl::onAnimation = false;
bool GUIControl::initialStep = true;
float GUIControl::ballVelocityZ = 0;
float GUIControl::cameraFocalScale = 0.5;

Fl_Button* GUIControl::startButton;
Fl_Button* GUIControl::pauseButton;
Fl_Button* GUIControl::recordButton;
Fl_Value_Slider* GUIControl::ballVelocityZSlider;
Fl_Value_Slider* GUIControl::cameraFocalScaleSlider;

/*
GUIControl::GUIControl(Scene* s) : scene(s) {
	onAnimation = false;
	initialStep = true;
	ballVelocityZ = 0;

	s->control = this;
}
*/

void GUIControl::startButtonPressed() {
	if (!scene) {
		return;
	}

	if (initialStep) { // at initial step, start the animation
		startButton->label("reset");
		startButton->deactivate();

		pauseButton->label("pause");
		pauseButton->activate();

		ballVelocityZSlider->deactivate();

		onAnimation = true;
		initialStep = false;

		// scene->ball->getRigidBody()->setLinearVelocity(btVector3(20, 0, ballVelocityZ));
	} else { // reset
		startButton->label("start");

		pauseButton->deactivate();

		ballVelocityZSlider->activate();

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

	if (!scene->m_recording) {
		recordButton->label("stop recording");
		scene->m_recording = true;
	} else {
		recordButton->label("record");
		scene->m_recording = false;
	}
}

void GUIControl::showControlDialog() {
	Fl_Window* window = new Fl_Window(300, 500);

	window->begin();

	startButton = new Fl_Button(0, 0, 300, 30, "start");
	startButton->callback((Fl_Callback*) startButtonPressed);

	pauseButton = new Fl_Button(0, 40, 300, 30, "pause");
	pauseButton->callback((Fl_Callback*) pauseButtonPressed);
	pauseButton->deactivate();

	Fl_Box* ballVelocityLabel = new Fl_Box(0, 80, 300, 20, "ball velocity on Z direction");

	ballVelocityZSlider = new Fl_Value_Slider(0, 110, 300, 30);
	ballVelocityZSlider->type(FL_HOR_SLIDER);
	ballVelocityZSlider->bounds(-1, 1);
	ballVelocityZSlider->value(0);
	ballVelocityZSlider->callback((Fl_Callback*) ballVelocityZSliderValueChanged);

	Fl_Box* cameraFocalLengthLabel = new Fl_Box(0, 150, 300, 20, "camera focal length");

	cameraFocalScaleSlider = new Fl_Value_Slider(0, 180, 300, 30);
	cameraFocalScaleSlider->type(FL_HOR_SLIDER);
	cameraFocalScaleSlider->bounds(0.2, 5);
	cameraFocalScaleSlider->value(0.5);
	cameraFocalScaleSlider->callback((Fl_Callback*) cameraFocalScaleChanged);

	recordButton = new Fl_Button(0, 220, 300, 30, "record");
	recordButton->callback((Fl_Callback*) recordButtonPressed);

	window->end();
	window->show();

	Fl::run();
}