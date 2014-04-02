#ifndef GUI_CONTROL_
#define GUI_CONTROL_

#include <FL\Fl.H>
#include <FL\Fl_Window.H>
#include <FL\Fl_Box.H>
#include <FL\Fl_Button.H>
#include <FL\Fl_Value_Slider.H>

class Scene;

class GUIControl {
public:
	GUIControl(Scene* s);

	static void startButtonPressed();
	static void pauseButtonPressed();
	static void ballVelocityZSliderValueChanged();
	static void cameraFocalLengthChanged();
	static void recordButtonPressed();
	static void showControlDialog();

	static Scene* scene;

	// indicating whether animation is on
	static bool onAnimation;

	// indicating whether at initial simulation step
	static bool initialStep;

	static float ballVelocityZ;

	static Fl_Button* startButton;
	static Fl_Button* pauseButton;
	static Fl_Button* recordButton;

	static Fl_Value_Slider* ballVelocityZSlider;
	static Fl_Value_Slider* cameraFocalLengthSlider;

};

#endif