#ifndef GUI_CONTROL_
#define GUI_CONTROL_

#include <FL\Fl.H>
#include <FL\Fl_Window.H>
#include <FL\Fl_Box.H>
#include <FL\Fl_Button.H>
#include <FL\Fl_Light_Button.H>
#include <FL\Fl_Value_Slider.H>

class Scene;

class GUIControl {
public:
	GUIControl(Scene* s);

	static void startButtonPressed();
	static void pauseButtonPressed();
	static void ballVelocityZSliderValueChanged();
	static void cameraFocalScaleChanged();
	static void recordButtonPressed();
	static void snapShotButtonPressed();
	static void showControlDialog();
	static void dofLightButtonPressed();
	static void softShadowLightButtonPressed();
	static void glossyLightButtonPressed();
	static void aaLightButtonPressed();
	static void motionBlurLightButtonPressed();
	static void giLightButtonPressed();
	static void cameraChangeButtonPressed();

	static Scene* scene;

	// indicating whether animation is on
	static bool onAnimation;

	// indicating whether at initial simulation step
	static bool initialStep;

	static float ballVelocityZ;

	static float cameraFocalScale;

	static bool dofOn;
	static bool softShadowOn;
	static bool glossyOn;
	static bool aaOn;
	static bool motionBlurOn;
	static bool giOn;

	static Fl_Button* startButton;
	static Fl_Button* pauseButton;
	static Fl_Button* recordButton;
	static Fl_Button* snapShotButton;

	static Fl_Value_Slider* ballVelocityZSlider;
	static Fl_Value_Slider* cameraFocalScaleSlider;

	static Fl_Light_Button* dofLightButton;
	static Fl_Light_Button* softShadowLightButton;
	static Fl_Light_Button* glossyLightButton;
	static Fl_Light_Button* aaLightButton;
	static Fl_Light_Button* motionBlurLightButton;
	static Fl_Light_Button* giLightButton;

	static Fl_Button* cameraChangeButton;
};

#endif