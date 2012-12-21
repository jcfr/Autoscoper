// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

// callbacks.cpp

#ifdef _WIN32
#include <windows.h>
#endif

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>

#ifdef _MACOSX
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <gtk/gtk.h>
#include <gtk/gtkgl.h>
#include <gdk/gdkkeysyms.h>

#include <glib.h>

#include <RadRenderer.hpp>
#include <Camera.hpp>
#include <CoordFrame.hpp>
#include <KeyCurve.hpp>
#include <Tracker.hpp>
#include <View.hpp>
#include <RayCaster.hpp>
#include <SobelFilter.hpp>
#include <ContrastFilter.hpp>
#include <SharpenFilter.hpp>
#include <GaussianFilter.hpp>
#include <Filter.hpp>
#include <Trial.hpp>
#include <Video.hpp>
#include <Volume.hpp>

#include "callbacks.hpp"
#include "interface.hpp"
#include "support.hpp"
#include "xromm_gtk_draw.hpp"
#include "xromm_gtk_tree_view.hpp"
#include "History.hpp"
#include "Manip3D.hpp"
#include "new_trial_dialog.hpp"

using namespace std;
using namespace xromm;

static Tracker tracker;
static Manip3D manip;

static const int DEFAULT_NUM_OF_FRAMES = 100;

// Global gl context
static GdkGLContext *glcontext = NULL;

// Default camera
static const double xyzypr[6] = {250.0f, 250.0f, 250.0f, 0.0f, 45.0f, -35.0f};
static CoordFrame defaultViewMatrix = CoordFrame::from_xyzypr(xyzypr);

// Volume orientation
//static CoordFrame manip_matrix;
static CoordFrame volume_matrix;

static bool drawGrid = true;
static bool movePivot = false;

static const float top_color[3] = {0.20f,0.35f,0.50f};
static const float bot_color[3] = {0.10f,0.17f,0.25f};

int max_viewport_dims[2];

enum CameraID
{
    DEFAULT_CAMERA = -1,
    CAMERA1 = 0,
    CAMERA2 = 1
};

enum
{
    LOCATION_ATTRIB_ROW = 0,
    LOOK_ATTRIB_ROW,
    UP_ATTRIB_ROW,
    RIGHT_ATTRIB_ROW
};

struct ViewData
{
    CameraID cameraid;

    int window_width;
    int window_height;

    float ratio;
    float fovy;
    float near_clip;
    float far_clip;

    float zoom;
    float zoom_x;
    float zoom_y;

    int viewport_x;
    int viewport_y;
    int viewport_width;
    int viewport_height;

    double scale;

    GLuint pbo;
};

struct GraphData
{
    bool show_x;
    bool show_y;
    bool show_z;
    bool show_yaw;
    bool show_pitch;
    bool show_roll;

    double min_frame;
    double max_frame;
    double min_value;
    double max_value;

    vector<bool> frame_locks;
};

enum Selection_type { NODE, IN_TANGENT, OUT_TANGENT };

vector<pair<pair<KeyCurve*,KeyCurve::iterator>,Selection_type> > selected_nodes;
vector<pair<KeyCurve*,KeyCurve::iterator> > copied_nodes;

bool modify_nodes = false;

bool draw_marquee = false;
float marquee[4];

// Drawing Area 1
static GtkWidget* drawingarea1;
static GtkWidget* drawingarea1_combobox;
static ViewData drawingarea1_view;

// Drawing Area 2
static GtkWidget* drawingarea2;
static GtkWidget* drawingarea2_combobox;
static ViewData drawingarea2_view;

//static float camera_scale = 25.0f;
static float pivot_size = 0.25f;

// Drawing Area 3
static GtkWidget* graph_drawingarea;
static ViewData graph_view;
static GraphData position_graph;
static GraphData error_graph;

// Widgets
static GtkWidget* timeline;

static bool spin_button_update = true;
static GtkWidget* x_spin_button;
static GtkWidget* y_spin_button;
static GtkWidget* z_spin_button;
static GtkWidget* yaw_spin_button;
static GtkWidget* pitch_spin_button;
static GtkWidget* roll_spin_button;
static GtkWidget* min_timeline_spin_button;
static GtkWidget* max_timeline_spin_button;

static GtkWidget* trial_tree_view;

// Dialog
static GtkWidget* tracking_dialog = NULL;

//static GtkWidget* volume_renderer_options = NULL;
static GtkWidget* export_tracking_dialog = NULL;

static GtkWidget* window;

static void reset_graph(GraphData*);

static vector<GLuint> textures;

static void update();
static void fill_notebook();
static void redraw_drawingarea(GtkWidget*);
static void save_tracking_results(const string& filename);
static string get_filename(bool save = false);

static double press_x;
static double press_y;

struct State
{
    KeyCurve x_curve;
    KeyCurve y_curve;
    KeyCurve z_curve;
    KeyCurve x_rot_curve;
    KeyCurve y_rot_curve;
    KeyCurve z_rot_curve;
};

History<State> history(10);
string trial_filename;
bool is_trial_saved = true;
bool is_tracking_saved = true;

static void
save_trial_prompt()
{
    if (is_trial_saved) {
        return;
    }

    GtkWidget* dialog = gtk_message_dialog_new(NULL,
        GtkDialogFlags(GTK_DIALOG_MODAL|GTK_DIALOG_DESTROY_WITH_PARENT),
        GTK_MESSAGE_QUESTION,GTK_BUTTONS_YES_NO,
        "Would you like to save the current trial?");

    gint result = gtk_dialog_run(GTK_DIALOG(dialog));
    switch (result) {
        case GTK_RESPONSE_YES: {
            on_save_trial1_activate(NULL,NULL);
            break;
        }
        default: {
            break;
        }
    }

    gtk_widget_destroy(dialog);
}

static void
save_tracking_prompt()
{
    if (is_tracking_saved) {
        return;
    }

    GtkWidget* dialog = gtk_message_dialog_new(NULL,
        GtkDialogFlags(GTK_DIALOG_MODAL|GTK_DIALOG_DESTROY_WITH_PARENT),
        GTK_MESSAGE_QUESTION,GTK_BUTTONS_YES_NO,
        "Would you like to export the unsaved tracking data?");

    gint result = gtk_dialog_run(GTK_DIALOG(dialog));
    switch (result) {
        case GTK_RESPONSE_YES: {
            string filename = get_filename(true);
            if (filename.compare("") != 0) {
                save_tracking_results(filename);
            }
            break;
        }
        default: {
            break;
        }
    }

    gtk_widget_destroy(dialog);
}

static CoordFrame
manip_matrix()
{
    return CoordFrame::from_matrix(trans(manip.transform()));
}

static void
set_manip_matrix(const CoordFrame& frame)
{
    double m[16];
    frame.to_matrix_row_order(m);
    manip.set_transform(Mat4d(m));
}

static void
mouse_to_graph(double mouse_x,
               double mouse_y,
               double& graph_x,
               double& graph_y)
{
    double frame_offset = 48.0*(position_graph.max_frame-
                                position_graph.min_frame)/
                                graph_view.viewport_width;
    double min_frame = position_graph.min_frame-frame_offset;
    double max_frame = position_graph.max_frame-1.0;
    double value_offset = 24.0*(position_graph.max_value-
                                position_graph.min_value)/
                                graph_view.viewport_height;
    double value_offset_top = 8.0*(position_graph.max_value-
                                   position_graph.min_value)/
                                   graph_view.viewport_height;
    double min_value = position_graph.min_value-value_offset;
    double max_value = position_graph.max_value+value_offset_top;

    double ndcx = mouse_x/graph_view.viewport_width;
    double ndcy = 1.0-mouse_y/graph_view.viewport_height;

    graph_x = min_frame+ndcx*(max_frame+1-min_frame);
    graph_y = min_value+ndcy*(max_value-min_value);
}

// Initialization function. Sets up the opengl state and other default params.
static void init()
{
	cerr << "Initializing OpenGL..." << endl;

	glewInit();

#ifndef _WIN32
	cerr << "Initializing GLUT..." << endl;

	int argc = 0;
	char argv[1] = {'\n'};
	char* pargv = &argv[0];

	glutInit(&argc,&pargv);
#endif

    glDisable(GL_LIGHTING);

    glEnable(GL_DEPTH_TEST);

    //glEnable(GL_LINE_SMOOTH);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE);

    glClearColor(0.5,0.5,0.5,1.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Save the maximum size of the viewport.
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_viewport_dims);

    reset_graph(&position_graph);

	cerr << "Initializing OpenCL-OpenGL interoperability..." << endl;
	opencl::opencl_global_gl_context();
}

static
string
get_filename(bool save)
{
    GtkWidget* chooser =
        gtk_file_chooser_dialog_new((save? "Save As": "Select File"),
                                    NULL,
                                    (save? GTK_FILE_CHOOSER_ACTION_SAVE:
                                           GTK_FILE_CHOOSER_ACTION_OPEN),
                                    GTK_STOCK_CANCEL,
                                    GTK_RESPONSE_CANCEL,
                                    GTK_STOCK_OK,
                                    GTK_RESPONSE_OK,
                                    NULL);
    if (save) {
        gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(chooser),TRUE);
    }

    gint response = gtk_dialog_run(GTK_DIALOG(chooser));
    string filename;
    if (response == GTK_RESPONSE_OK) {
        filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(chooser));
    }

    gtk_widget_destroy(chooser);

    return filename;
}

static
void
load_trial(const string& filename)
{
    try {
        Trial trial(filename.c_str());
        tracker.load(trial);

        trial_filename = filename;
        is_trial_saved = true;
        is_tracking_saved = true;

        manip.set_transform(Mat4d());
        volume_matrix = CoordFrame();

        gtk_range_set_value(
            GTK_RANGE(lookup_widget(window,
                "xromm_markerless_tracking_timeline")),0);

        update();
        fill_notebook();
        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
    }
    catch (exception& e) {
        cerr << e.what() << endl;
    }
}

// Automatically updates the graph's minimum and maximum values to stretch the
// data the full height of the viewport.
static void update_graph_min_max(GraphData* graph, int frame = -1)
{
    if (!tracker.trial() || tracker.trial()->x_curve.empty()) {
        graph->max_value = 180.0;
        graph->min_value = -180.0;
    }
    // If a frame is specified then only check that frame for a new minimum and
    // maximum.
    else if (frame != -1) {
        if (graph->show_x) {
            float x_value = tracker.trial()->x_curve(frame);
            if (x_value > graph->max_value) {
                graph->max_value = x_value;
            }
            if (x_value < graph->min_value) {
                graph->min_value = x_value;
            }
        }
        if (graph->show_y) {
            float y_value = tracker.trial()->y_curve(frame);
            if (y_value > graph->max_value) {
                graph->max_value = y_value;
            }
            if (y_value < graph->min_value) {
                graph->min_value = y_value;
            }
        }
        if (graph->show_z) {
            float z_value = tracker.trial()->z_curve(frame);
            if (z_value > graph->max_value) {
                graph->max_value = z_value;
            }
            if (z_value < graph->min_value) {
                graph->min_value = z_value;
            }
        }
        if (graph->show_yaw) {
            float yaw_value = tracker.trial()->yaw_curve(frame);
            if (yaw_value > graph->max_value) {
                graph->max_value = yaw_value;
            }
            if (yaw_value < graph->min_value) {
                graph->min_value = yaw_value;
            }
        }
        if (graph->show_pitch) {
            float pitch_value = tracker.trial()->pitch_curve(frame);
            if (pitch_value > graph->max_value) {
                graph->max_value = pitch_value;
            }
            if (pitch_value < graph->min_value) {
                graph->min_value = pitch_value;
            }
        }
        if (graph->show_roll) {
            float roll_value = tracker.trial()->roll_curve(frame);
            if (roll_value > graph->max_value) {
                graph->max_value = roll_value;
            }
            if (roll_value < graph->min_value) {
                graph->min_value = roll_value;
            }
        }
    }
    // Otherwise we need to check all the frames.
    else {

        graph->min_value = 1e6;
        graph->max_value = -1e6;

        if (graph->show_x) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float x_value = tracker.trial()->x_curve(frame);
                if (x_value > graph->max_value) {
                    graph->max_value = x_value;
                }
                if (x_value < graph->min_value) {
                    graph->min_value = x_value;
                }
            }
        }
        if (graph->show_y) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float y_value = tracker.trial()->y_curve(frame);
                if (y_value > graph->max_value) {
                    graph->max_value = y_value;
                }
                if (y_value < graph->min_value) {
                    graph->min_value = y_value;
                }
            }
        }
        if (graph->show_z) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float z_value = tracker.trial()->z_curve(frame);
                if (z_value > graph->max_value) {
                    graph->max_value = z_value;
                }
                if (z_value < graph->min_value) {
                    graph->min_value = z_value;
                }
            }
        }
        if (graph->show_yaw) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float yaw_value = tracker.trial()->yaw_curve(frame);
                if (yaw_value > graph->max_value) {
                    graph->max_value = yaw_value;
                }
                if (yaw_value < graph->min_value) {
                    graph->min_value = yaw_value;
                }
            }
        }
        if (graph->show_pitch) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float pitch_value = tracker.trial()->pitch_curve(frame);
                if (pitch_value > graph->max_value) {
                    graph->max_value = pitch_value;
                }
                if (pitch_value < graph->min_value) {
                    graph->min_value = pitch_value;
                }
            }
        }
        if (graph->show_roll) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float roll_value = tracker.trial()->roll_curve(frame);
                if (roll_value > graph->max_value) {
                    graph->max_value = roll_value;
                }
                if (roll_value < graph->min_value) {
                    graph->min_value = roll_value;
                }
            }
        }

        graph->min_value -= 1.0;
        graph->max_value += 1.0;
    }
}

// Helper functions that invalidate the drawing areas so that they will be
// redrawn when the program returns to the main loop.
static void redraw_drawingarea(GtkWidget* drawingarea)
{
    if (drawingarea == NULL) {
        return;
    }

    GdkRectangle rectangle = drawingarea->allocation;
    rectangle.x = 0;
    rectangle.y = 0;

    gdk_window_invalidate_rect(drawingarea->window, &rectangle, FALSE);
}

// Updates the size and location of the viewport based on the zoom, zoom_x,
// zoom_y, window_width, and window_height variables in the view. This mimics
// camera zooming and panning but does not change the perspective of the view.
// This function should be called when any one of the above parameters changes.
static void update_viewport(ViewData* view)
{
    if (view->cameraid == DEFAULT_CAMERA) {

        view->zoom = 1.0f;
        view->zoom_x = 0.0f;
        view->zoom_y = 0.0f;

        view->viewport_x = 0;
        view->viewport_y = 0;
        view->viewport_width = view->window_width;
        view->viewport_height = view->window_height;

        return;
    }

    // A zoom of 1 corresponds to a viewport that is the same size as the
    // viewing window. A zoom of 2 corrseponds to a viewport that is twice the
    // size of the window in each dimension. We don't ever want the viewport to
    // be smaller than the window so we clamp the min to 1.
    if (view->zoom < 1.0f) {
        view->zoom = 1.0f;
    }

    view->viewport_width = (int)(view->window_width*view->zoom);
    view->viewport_height = (int)(view->window_height*view->zoom);

    // Clamp the viewport width and height to the maximum viewport dimensions
    // supported by this opengl implementation.
    /* TODO: This effectively limits the maximum amount of zoom. If we do not do
     * this then the manipulator will not be drawn correctly at extreme scales.
     * We need a between solution to this problem.
    if (view->viewport_width > max_viewport_dims[0]) {
        view->viewport_width = max_viewport_dims[0];
        view->zoom = (float)view->viewport_width/(float)view->window_width;
        view->viewport_height = (int)(view->window_height*view->zoom);
    }
    if (view->viewport_height > max_viewport_dims[1]) {
        view->viewport_height = max_viewport_dims[1];
        view->zoom = (float)view->viewport_height/(float)view->window_height;
        view->viewport_width = (int)(view->window_width*view->zoom);
    }
    */

    // The zoom_x and zoom_y parameters should be normalized between -1 and 1.
    // They determine the location of the window in the viewport. They are
    // clamped so that the window never moves outside of the viewport.
    view->viewport_x = -(int)(view->viewport_width/2.0f*
                             (1.0+view->zoom_x-1.0/view->zoom));
    view->viewport_y = -(int)(view->viewport_height/2.0f*
                             (1.0+view->zoom_y-1.0/view->zoom));

    int min_viewport_x = view->window_width-view->viewport_width;
    int max_viewport_x = 0;
    int min_viewport_y = view->window_height-view->viewport_height;
    int max_viewport_y = 0;

    if (view->viewport_x < min_viewport_x) {
        view->viewport_x = min_viewport_x;
        view->zoom_x = 1.0f/(float)view->zoom-
                       2.0f*view->viewport_x/(float)view->viewport_width-1.0f;
    }
    if (view->viewport_x > max_viewport_x) {
        view->viewport_x = max_viewport_x;
        view->zoom_x = 1.0f/(float)view->zoom-
                       2.0f*view->viewport_x/(float)view->viewport_width-1.0f;
    }
    if (view->viewport_y < min_viewport_y) {
        view->viewport_y = min_viewport_y;
        view->zoom_y = 1.0f/(float)view->zoom-
                       2.0f*view->viewport_y/(float)view->viewport_height-1.0f;
    }
    if (view->viewport_y > max_viewport_y) {
        view->viewport_y = max_viewport_y;
        view->zoom_y = 1.0f/(float)view->zoom-
                       2.0f*view->viewport_y/(float)view->viewport_height-1.0f;
    }
}

// Resizes the specified view. Note that this function calls update_viewport
// when it is completed.
static void resize_view(ViewData* view, int width, int height)
{
    view->window_width = width;
    view->window_height = height;

    // Prevent divie by 0.
    if (view->window_height == 0) {
        view->window_height = 1;
    }

    view->ratio = (float)view->window_width/(float)view->window_height;

    // Unregister and delete the pixel buffer if it already exists.
    if (!glIsBufferARB(view->pbo)) {
        //glDeleteBuffersARB(1, &view->pbo);
        glGenBuffersARB(1, &view->pbo);
    }

    // Create a pixel buffer object.
    //glGenBuffersARB(1, &view->pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, view->pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                3*view->window_width*view->window_height*sizeof(float),
                0,
                GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    if (view->viewport_width < view->window_width) {
        view->viewport_width = view->window_width;
    }

    if (view->viewport_height < view->window_height) {
        view->viewport_height = view->window_height;
    }

    // The viewport needs to be recalculated.
    update_viewport(view);
}

// Changes the size of the manipulator based on the view. The goal is to keep
// the manipulator/pivot the same size relative to the screen.
static void update_scale_in_view(ViewData* view)
{
    // Determine the distance from the center of the pivot point to the
    // center of the view.

    CoordFrame mat = manip_matrix();
    double dist_vec[3];
    if (view->cameraid == DEFAULT_CAMERA) {
        dist_vec[0] = mat.translation()[0]-
                      defaultViewMatrix.translation()[0];
        dist_vec[2] = mat.translation()[1]-
                      defaultViewMatrix.translation()[1];
        dist_vec[1] = mat.translation()[2]-
                      defaultViewMatrix.translation()[2];
    }
    else {
        dist_vec[0] = mat.translation()[0]-
            tracker.trial()->cameras.at(view->cameraid).coord_frame().translation()[0];
        dist_vec[1] = mat.translation()[1]-
            tracker.trial()->cameras.at(view->cameraid).coord_frame().translation()[1];
        dist_vec[2] = mat.translation()[2]-
            tracker.trial()->cameras.at(view->cameraid).coord_frame().translation()[2];
    }
    double dist = sqrt(dist_vec[0]*dist_vec[0]+
                       dist_vec[1]*dist_vec[1]+
                       dist_vec[2]*dist_vec[2]);

    // Adjust the size of the pivot based on the distance.
    view->scale = 2.0*dist*tan(view->fovy*M_PI/360.0)*view->near_clip/view->zoom;
}

// Selects the axis of translation or rotation of the manipulator that is under
// the mouse located at pixel coordinates x,y.
static void select_manip_in_view(ViewData* view, double x, double y, int button)
{
    // Setup the view from this perspective so that we can simply call set_view
    // on the manipulator

    glViewport(view->viewport_x,
               view->viewport_y,
               view->viewport_width,
               view->viewport_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(view->fovy,view->ratio,view->near_clip,view->far_clip);

    CoordFrame viewMatrix = defaultViewMatrix;
    if (view->cameraid != DEFAULT_CAMERA) {
        viewMatrix = tracker.trial()->cameras.at(view->cameraid).coord_frame();
    }

    double m[16];
    viewMatrix.inverse().to_matrix(m);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(m);

    manip.set_view();
    manip.set_size(view->scale*pivot_size);
    manip.on_mouse_press(x,view->window_height-y);
}

// Moves the manipulator and volume based on the view, the selected axis, and
// the direction of the motion.
static void move_manip_in_view(const ViewData* view, double x, double y, bool out_of_plane=false)
{
    if (position_graph.frame_locks.at(tracker.trial()->frame)) {
        return;
    }

    CoordFrame frame;
    if (movePivot) {
        frame = (manip_matrix()*volume_matrix);
    }

    if (!out_of_plane) {
        manip.set_size(view->scale*pivot_size);
        manip.on_mouse_move(x,view->window_height-y);
    }
    else if (manip.selection() == Manip3D::VIEW_PLANE) {
        CoordFrame mmat = manip_matrix();
        CoordFrame viewMatrix = defaultViewMatrix;
        if (view->cameraid != DEFAULT_CAMERA) {
            viewMatrix = tracker.trial()->cameras.at(view->cameraid).coord_frame();
        }

        double zdir[3] = { mmat.translation()[0]-viewMatrix.translation()[0],
                           mmat.translation()[1]-viewMatrix.translation()[1],
                           mmat.translation()[2]-viewMatrix.translation()[2]};
        double mag = sqrt(zdir[0]*zdir[0]+zdir[1]*zdir[1]+zdir[2]*zdir[2]);
        zdir[0] /= mag;
        zdir[1] /= mag;
        zdir[2] /= mag;

        double ztrans[3] = { (x-y)/2.0*zdir[0],(x-y)/2.0*zdir[1],(x-y)/2.0*zdir[2] };

        mmat.translate(ztrans);
        set_manip_matrix(mmat);

        manip.set_selection(Manip3D::VIEW_PLANE);
    }

    if (movePivot) {
        CoordFrame new_manip_matrix = manip_matrix();
        volume_matrix = new_manip_matrix.inverse()*frame;
    }
}

// Updates the current position and rotation values after the volume has been
// moved. This is done for both the graph and spin buttons.
static void update_xyzypr()
{
    double xyzypr[6];
    (manip_matrix()*volume_matrix).to_xyzypr(xyzypr);

    // We do not update the graph because the graph only updates when values
    // have been keyed.

    //position_graph.x_values[tracker.trial()->frame] = xyzypr[0];
    //position_graph.y_values[tracker.trial()->frame] = xyzypr[1];
    //position_graph.z_values[tracker.trial()->frame] = xyzypr[2];
    //position_graph.yaw_values[tracker.trial()->frame] = xyzypr[3];
    //position_graph.pitch_values[tracker.trial()->frame] = xyzypr[4];
    //position_graph.roll_values[tracker.trial()->frame] = xyzypr[5];

    //Update the spin buttons.
    spin_button_update = false;
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(x_spin_button), xyzypr[0]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(y_spin_button), xyzypr[1]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(z_spin_button), xyzypr[2]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(yaw_spin_button), xyzypr[3]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(pitch_spin_button), xyzypr[4]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(roll_spin_button), xyzypr[5]);
    spin_button_update = true;
}

// Updates the coordinate frames position after the spin buttons values have
// been changed.
static void update_coord_frame()
{
    double xyzypr[6];
    xyzypr[0] = gtk_spin_button_get_value(GTK_SPIN_BUTTON(x_spin_button));
    xyzypr[1] = gtk_spin_button_get_value(GTK_SPIN_BUTTON(y_spin_button));
    xyzypr[2] = gtk_spin_button_get_value(GTK_SPIN_BUTTON(z_spin_button));
    xyzypr[3] = gtk_spin_button_get_value(GTK_SPIN_BUTTON(yaw_spin_button));
    xyzypr[4] = gtk_spin_button_get_value(GTK_SPIN_BUTTON(pitch_spin_button));
    xyzypr[5] = gtk_spin_button_get_value(GTK_SPIN_BUTTON(roll_spin_button));

    CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
    CoordFrame mat = newCoordFrame*volume_matrix.inverse();
    set_manip_matrix(newCoordFrame*volume_matrix.inverse());

    // We do not update the graph because the graph only updates when values
    // have been keyed.

    //if (tracker.trial()->coordFrames.size() > 0) {
    //    tracker.trial()->coordFrames[tracker.trial()->frame] = newCoordFrame;
    //    position_graph.x_values[tracker.trial()->frame] = xyzypr[0];
    //    position_graph.y_values[tracker.trial()->frame] = xyzypr[1];
    //    position_graph.z_values[tracker.trial()->frame] = xyzypr[2];
    //    position_graph.yaw_values[tracker.trial()->frame] = xyzypr[3];
    //    position_graph.pitch_values[tracker.trial()->frame] = xyzypr[4];
    //    position_graph.roll_values[tracker.trial()->frame] = xyzypr[5];
    //}
}

// Updates the coordinate frames position after the spin buttons values have
// been changed.
static void update_xyzypr_and_coord_frame()
{
    if (tracker.trial()->x_curve.empty()) {
        return;
    }

    double xyzypr[6];
    xyzypr[0] = tracker.trial()->x_curve(tracker.trial()->frame);
    xyzypr[1] = tracker.trial()->y_curve(tracker.trial()->frame);
    xyzypr[2] = tracker.trial()->z_curve(tracker.trial()->frame);
    xyzypr[3] = tracker.trial()->yaw_curve(tracker.trial()->frame);
    xyzypr[4] = tracker.trial()->pitch_curve(tracker.trial()->frame);
    xyzypr[5] = tracker.trial()->roll_curve(tracker.trial()->frame);

    CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
    set_manip_matrix(newCoordFrame*volume_matrix.inverse());

    spin_button_update = false;
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(x_spin_button), xyzypr[0]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(y_spin_button), xyzypr[1]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(z_spin_button), xyzypr[2]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(yaw_spin_button), xyzypr[3]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(pitch_spin_button), xyzypr[4]);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(roll_spin_button), xyzypr[5]);
    spin_button_update = true;
}


// Updates the position of the volume based on the frame. Also updates the spin
// buttons to reflect this new position and redraws the drawing area.
static void frame_changed()
{
    // Lock or unlock the position

    if (position_graph.frame_locks.at(tracker.trial()->frame)) {
        gtk_widget_set_sensitive(x_spin_button,false);
        gtk_widget_set_sensitive(y_spin_button,false);
        gtk_widget_set_sensitive(z_spin_button,false);
        gtk_widget_set_sensitive(yaw_spin_button,false);
        gtk_widget_set_sensitive(pitch_spin_button,false);
        gtk_widget_set_sensitive(roll_spin_button,false);
    }
    else {
        gtk_widget_set_sensitive(x_spin_button,true);
        gtk_widget_set_sensitive(y_spin_button,true);
        gtk_widget_set_sensitive(z_spin_button,true);
        gtk_widget_set_sensitive(yaw_spin_button,true);
        gtk_widget_set_sensitive(pitch_spin_button,true);
        gtk_widget_set_sensitive(roll_spin_button,true);
    }

    update_xyzypr_and_coord_frame();

    for (unsigned int i = 0; i < tracker.trial()->cameras.size(); ++i) {
        tracker.trial()->videos.at(i).set_frame(tracker.trial()->frame);
        tracker.view(i)->radRenderer()->set_rad(
            tracker.trial()->videos.at(i).data(),
            tracker.trial()->videos.at(i).width(),
            tracker.trial()->videos.at(i).height(),
            tracker.trial()->videos.at(i).bps());

        glBindTexture(GL_TEXTURE_2D,textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     1,
                     tracker.trial()->videos.at(i).width(),
                     tracker.trial()->videos.at(i).height(),
                     0,
                     GL_LUMINANCE,
                    (tracker.trial()->videos.at(i).bps() == 8? GL_UNSIGNED_BYTE:
                                                               GL_UNSIGNED_SHORT),
                     tracker.trial()->videos.at(i).data());
        glBindTexture(GL_TEXTURE_2D,0);
    }

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);
    gdk_window_process_all_updates();
}

static
void
save_tracking_results(const string& filename)
{
    if (export_tracking_dialog == NULL) {
        export_tracking_dialog = create_export_tracking_options_dialog();
    }

    gint response = gtk_dialog_run(GTK_DIALOG(export_tracking_dialog));
    gtk_widget_hide(export_tracking_dialog);

    if (response != GTK_RESPONSE_OK) {
        return;
    }

    bool save_as_matrix = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_type_matrix")));
    bool save_as_rows = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_orientation_row")));
    bool save_with_commas = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_separator_comma")));
    bool convert_to_cm = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_translation_cm")));
    bool convert_to_rad = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_rotation_radians")));
    bool interpolate = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_interpolate_spline")));

    const char* s = save_with_commas? "," :" ";

    ofstream file(filename.c_str(), ios::out);

    file.precision(16);
    file.setf(ios::fixed,ios::floatfield);

    for (int i = 0; i < tracker.trial()->num_frames; ++i) {

        if (!interpolate) {
            if (tracker.trial()->x_curve.find(i) ==
                    tracker.trial()->x_curve.end() &&
                tracker.trial()->y_curve.find(i) ==
                    tracker.trial()->y_curve.end() &&
                tracker.trial()->z_curve.find(i) ==
                    tracker.trial()->z_curve.end() &&
                tracker.trial()->yaw_curve.find(i) ==
                    tracker.trial()->yaw_curve.end() &&
                tracker.trial()->pitch_curve.find(i) ==
                    tracker.trial()->pitch_curve.end() &&
                tracker.trial()->roll_curve.find(i) ==
                    tracker.trial()->roll_curve.end()) {
                if (save_as_matrix) {
                    file << "NaN";
                    for (int j = 0; j < 15; j++) { file << s << "NaN"; }
                    file << endl;
                }
                else {
                    file << "NaN";
                    for (int j = 0; j < 5; j++) { file << s << "NaN"; }
                    file << endl;
                }
                continue;
            }
        }

        double xyzypr[6];
        xyzypr[0] = tracker.trial()->x_curve(i);
        xyzypr[1] = tracker.trial()->y_curve(i);
        xyzypr[2] = tracker.trial()->z_curve(i);
        xyzypr[3] = tracker.trial()->yaw_curve(i);
        xyzypr[4] = tracker.trial()->pitch_curve(i);
        xyzypr[5] = tracker.trial()->roll_curve(i);

        if (save_as_matrix) {
            double m[16];
            CoordFrame::from_xyzypr(xyzypr).to_matrix(m);

            if (convert_to_cm) {
                m[12] /= 10.0;
                m[13] /= 10.0;
                m[14] /= 10.0;
            }

            if (save_as_rows) {
                file << m[0] << s << m[4] << s << m[8] << s << m[12] << s
                     << m[1] << s << m[5] << s << m[9] << s << m[13] << s
                     << m[2] << s << m[6] << s << m[10] << s << m[14] << s
                     << m[3] << s << m[7] << s << m[11] << s<< m[15]
                     << endl;
            }
            else {
                file << m[0] << s << m[1] << s << m[2] << s << m[3] << s
                     << m[4] << s << m[5] << s << m[6] << s << m[7] << s
                     << m[8] << s << m[9] << s << m[10] << s << m[11] << s
                     << m[12] << s << m[13] << s << m[14] << s<< m[15]
                     << endl;
            }
        }
        else {
            if (convert_to_cm) {
                xyzypr[0] /= 10.0;
                xyzypr[1] /= 10.0;
                xyzypr[2] /= 10.0;
            }
            if (convert_to_rad) {
                xyzypr[3] *= M_PI/180.0;
                xyzypr[4] *= M_PI/180.0;
                xyzypr[5] *= M_PI/180.0;
            }

            file << xyzypr[0] << s << xyzypr[1] << s << xyzypr[2] << s
                 << xyzypr[3] << s << xyzypr[4] << s << xyzypr[5] << endl;
        }
    }
    file.close();

    is_tracking_saved = true;
}

static
void
load_tracking_results(const string& filename)
{
    save_tracking_prompt();

    if (export_tracking_dialog == NULL) {
        export_tracking_dialog = create_export_tracking_options_dialog();
    }

    gint response = gtk_dialog_run(GTK_DIALOG(export_tracking_dialog));
    gtk_widget_hide(export_tracking_dialog);

    if (response != GTK_RESPONSE_OK) {
        return;
    }

    bool save_as_matrix = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_type_matrix")));
    bool save_as_rows = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_orientation_row")));
    bool save_with_commas = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_separator_comma")));
    bool convert_to_cm = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_translation_cm")));
    bool convert_to_rad = (bool)gtk_toggle_button_get_active(
            GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
                                            "export_tracking_rotation_radians")));
    //bool interpolate = gtk_toggle_button_get_active(
    //        GTK_TOGGLE_BUTTON(lookup_widget(export_tracking_dialog,
    //                                        "export_tracking_interpolate_spline")));

    char s = save_with_commas? ',': ' ';

    ifstream file(filename.c_str(), ios::in);

    tracker.trial()->x_curve.clear();
    tracker.trial()->y_curve.clear();
    tracker.trial()->z_curve.clear();
    tracker.trial()->yaw_curve.clear();
    tracker.trial()->pitch_curve.clear();
    tracker.trial()->roll_curve.clear();

    double m[16];
    string line, value;
    for (int i = 0; i < tracker.trial()->num_frames && getline(file,line); ++i) {
        istringstream lineStream(line);
        for (int j = 0; j < (save_as_matrix? 16: 6) && getline(lineStream, value, s); ++j) {
            istringstream valStream(value);
            valStream >> m[j];
        }

        if (value.compare(0,3,"NaN") == 0) {
            continue;
        }

        if (save_as_matrix && save_as_rows) {
            double n[16];
            memcpy(n,m,16*sizeof(double));
            m[1] = n[4];
            m[2] = n[8];
            m[3] = n[12];
            m[4] = n[1];
            m[6] = n[9];
            m[7] = n[13];
            m[8] = n[2];
            m[9] = n[6];
            m[11] = n[14];
            m[12] = n[3];
            m[13] = n[7];
            m[14] = n[11];
        }

        if (convert_to_cm) {
            if (save_as_matrix) {
                m[12] *= 10.0;
                m[13] *= 10.0;
                m[14] *= 10.0;
            }
            else {
                m[0] *= 10.0;
                m[1] *= 10.0;
                m[2] *= 10.0;
            }
        }

        if (convert_to_rad) {
            if (!save_as_matrix) {
                m[3] *= 180.0/M_PI;
                m[4] *= 180.0/M_PI;
                m[5] *= 180.0/M_PI;
            }
        }

        if (save_as_matrix) {
            CoordFrame::from_matrix(m).to_xyzypr(m);
        }

        tracker.trial()->x_curve.insert(i,m[0]);
        tracker.trial()->y_curve.insert(i,m[1]);
        tracker.trial()->z_curve.insert(i,m[2]);
        tracker.trial()->yaw_curve.insert(i,m[3]);
        tracker.trial()->pitch_curve.insert(i,m[4]);
        tracker.trial()->roll_curve.insert(i,m[5]);
    }
    file.close();

    is_tracking_saved = true;

    frame_changed();
    update_graph_min_max(&position_graph);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);
}

static void reset_graph(GraphData* graph)
{
    tracker.trial()->x_curve.clear();
    tracker.trial()->y_curve.clear();
    tracker.trial()->z_curve.clear();
    tracker.trial()->yaw_curve.clear();
    tracker.trial()->pitch_curve.clear();
    tracker.trial()->roll_curve.clear();

    copied_nodes.clear();
}

// Renders a bitmap string at the specified position using glut.
static void render_bitmap_string(double x,
                                 double y,
                                 void* font,
                                 const char* string)
{
    glRasterPos2d(x,y);
    for (const char* c = string; *c != '\0'; ++c) {
        glutBitmapCharacter(font, *c);
    }
}

static void draw_curve(const KeyCurve& curve)
{
    // Get the minimum and maximum x-values

    float min_x, max_x;
    KeyCurve::const_iterator it = curve.begin();
    if (it == curve.end()) {
        return;
    }

    min_x = curve.time(it);
    it = curve.end(); it--;
    max_x = curve.time(it);

    // Clamp the values to the extents of the graph

    if (min_x < position_graph.min_frame) {
        min_x = position_graph.min_frame;
    }

    if (max_x > position_graph.max_frame) {
        max_x = position_graph.max_frame;
    }

    // Calculate the number of curve segments to draw

    int num_segments = graph_view.window_width/8;
    float dx = (max_x-min_x)/num_segments;
    dx = 1.0f/(int)(1.0f+1.0f/dx);

    // Draw the curve

    glBegin(GL_LINE_STRIP);
    for (float x = min_x; x < max_x; x += dx) {
        glVertex2f(x,curve(x));
    }
    glVertex2f(max_x,curve(max_x));
    glEnd();

    // Draw the curve points

    glPushAttrib(GL_CURRENT_BIT);

    float current_color[4];
    glGetFloatv(GL_CURRENT_COLOR,current_color);

    glBegin(GL_POINTS);
    it = curve.begin();
    while (it != curve.end()) {
        if (curve.time(it) < min_x || curve.time(it) > max_x) {
            it++;
            continue;
        }

        if (position_graph.frame_locks.at((int)curve.time(it))) {
            glColor3fv(current_color);
        }
        else {
            glColor3f(0.0f,0.0f,0.0f);
        }

        glVertex2f(curve.time(it),curve.value(it));
        it++;
    }
    glEnd();

    glPopAttrib();
}

// Draws the specified graph from the specified veiw. This function does not
// use many of the variables defined in the ViewData struct. This is because
// ViewData is primarily used to define a 3-dimensional view and the graph is
// 2-dimensional.
static void draw_graph(const ViewData* view, const GraphData* graph)
{
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);

    glPushAttrib(GL_POINT_BIT);
    glPointSize(3.0);

    glPushAttrib(GL_LINE_BIT);
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0);

    // Calculate how much space needs to be left on the left and bottom of the
    // graph in order to accomodate the labels.
    double frame_offset = 48.0*(graph->max_frame-graph->min_frame)/
                          view->viewport_width;
    double min_frame = graph->min_frame-frame_offset;
    double max_frame = graph->max_frame-1.0;
    double value_offset = 24.0*(graph->max_value-graph->min_value)/
                          view->viewport_height;
    double value_offset_top = 8.0*(graph->max_value-graph->min_value)/
                              view->viewport_height;
    double min_value = graph->min_value-value_offset;
    double max_value = graph->max_value+value_offset_top;

    glViewport(view->viewport_x,
               view->viewport_y,
               view->viewport_width,
               view->viewport_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(min_frame,max_frame+1,min_value,max_value);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Clear the buffers.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    double frame_dist = (int)ceil(frame_offset);
    double value_dist = 3.0*value_offset;

    if (frame_dist < 1.0) {
        frame_dist = 1.0;
    }

    // Draw grid with grid lines separated by the above frame_dist and
    // value_dist distances. Those distances are calculated each time this
    // fucntion is called and are based on the size of the window.
    glColor3f(0.25f,0.25f,0.25f);
    glBegin(GL_LINES);
    for (double x = graph->min_frame; x <= max_frame; x += frame_dist) {
        glVertex2d(x,min_value);
        glVertex2d(x,max_value);
    }
    glEnd();
    glBegin(GL_LINES);
    for (double y = 0; y < max_value; y += value_dist) {
        glVertex2d(min_frame,y);
        glVertex2d(max_frame+1,y);
    }
    for (double y = 0; y > min_value; y -= value_dist) {
        glVertex2d(min_frame,y);
        glVertex2d(max_frame+1,y);
    }
    glEnd();

    // Draw the x and y axes.
	glColor3f(0.75f,0.75f,0.75f);
    glBegin(GL_LINES);
    glVertex2d(min_frame,0.0);
    glVertex2d(max_frame+1,0.0);
    glVertex2d(0.0,min_value);
    glVertex2d(0.0,max_value);
    glEnd();

    // Draw grid labels.
    double char_width = 8.0*(graph->max_frame-graph->min_frame-frame_offset)/
                        view->viewport_width;
    double char_height = 13.0*(graph->max_value-graph->min_value-value_offset)/
                         view->viewport_height;

    glColor3f(0.0f,0.0f,0.0f);
    for (double x = graph->min_frame; x <= max_frame; x += frame_dist) {
        stringstream ss; ss << (int)x;
        render_bitmap_string(x-char_width*ss.str().length()/2.0,
                             min_value+char_height/2.0,
                             GLUT_BITMAP_8_BY_13,
                             ss.str().c_str());
    }
    for (double y = 0; y < max_value; y += value_dist) {
        stringstream ss; ss << (int)(y+0.5);
        render_bitmap_string(min_frame+char_width/2.0,
                             y-char_height/2.0,
                             GLUT_BITMAP_8_BY_13,
                             ss.str().c_str());
    }
    for (double y = 0; y > min_value-value_offset; y -= value_dist) {
        stringstream ss; ss << (int)(y+0.5);
        render_bitmap_string(min_frame+char_width/2.0,
                             y-char_height/2.0,
                             GLUT_BITMAP_8_BY_13,
                             ss.str().c_str());
    }

    // Draw current frame
    glBegin(GL_LINES);
    glVertex2d((double)tracker.trial()->frame,min_value);
    glVertex2d((double)tracker.trial()->frame,max_value);
    glEnd();

    // XXX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (draw_marquee) {

        glBegin(GL_LINES);
        glVertex2f(marquee[0],marquee[1]);
        glVertex2f(marquee[0],marquee[3]);

        glVertex2f(marquee[0],marquee[1]);
        glVertex2f(marquee[2],marquee[1]);

        glVertex2f(marquee[0],marquee[3]);
        glVertex2f(marquee[2],marquee[3]);

        glVertex2f(marquee[2],marquee[1]);
        glVertex2f(marquee[2],marquee[3]);
        glEnd();

        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2,0x3333);

        glColor3f(1.0f,1.0f,1.0f);
        glBegin(GL_LINES);
        glVertex2f(marquee[0],marquee[1]);
        glVertex2f(marquee[0],marquee[3]);

        glVertex2f(marquee[0],marquee[1]);
        glVertex2f(marquee[2],marquee[1]);

        glVertex2f(marquee[0],marquee[3]);
        glVertex2f(marquee[2],marquee[3]);

        glVertex2f(marquee[2],marquee[1]);
        glVertex2f(marquee[2],marquee[3]);
        glEnd();

        glLineStipple(1,0);
        glDisable(GL_LINE_STIPPLE);
    }

    // Draw the key frame curves

    if (graph->show_x) {
        glColor3f(1.0f,0.0f,0.0f);
        draw_curve(tracker.trial()->x_curve);
    }

    if (graph->show_y) {
        glColor3f(0.0f,1.0f,0.0f);
        draw_curve(tracker.trial()->y_curve);
    }

    if (graph->show_z) {
        glColor3f(0.0f,0.0f,1.0f);
        draw_curve(tracker.trial()->z_curve);
    }

    if (graph->show_yaw) {
        glColor3f(1.0f,1.0f,0.0f);
        draw_curve(tracker.trial()->yaw_curve);
    }

    if (graph->show_pitch) {
        glColor3f(1.0f,0.0f,1.0f);
        draw_curve(tracker.trial()->pitch_curve);
    }

    if (graph->show_roll) {
        glColor3f(0.0f,1.0f,1.0f);
        draw_curve(tracker.trial()->roll_curve);
    }

    float a = (max_frame+1-min_frame)/(max_value-min_value)*
              view->viewport_height/view->viewport_width;
    float tan_scale = 40.0f*(max_frame+1-min_frame)/graph_view.viewport_width;

    for (unsigned i = 0; i < selected_nodes.size(); i++) {
        KeyCurve& curve = *selected_nodes[i].first.first;
        KeyCurve::iterator it = selected_nodes[i].first.second;
        Selection_type type = selected_nodes[i].second;

        float s_in = tan_scale/sqrt(1.0f+a*a*curve.in_tangent(it)*curve.in_tangent(it));
        float s_out = tan_scale/sqrt(1.0f+a*a*curve.out_tangent(it)*curve.out_tangent(it));

        glBegin(GL_LINES);

        if (type == NODE || type == IN_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
        else { glColor3f(0.0f,0.0f,0.0f); }

        glVertex2f(curve.time(it)-s_in,curve.value(it)-s_in*curve.in_tangent(it));
        glVertex2f(curve.time(it),curve.value(it));

        if (type == NODE || type == OUT_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
        else { glColor3f(0.0f,0.0f,0.0f); }

        glVertex2f(curve.time(it),curve.value(it));
        glVertex2f(curve.time(it)+s_out,curve.value(it)+s_out*curve.out_tangent(it));

        glEnd();

        glBegin(GL_POINTS);

        if (type == NODE || type == IN_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
        else { glColor3f(0.0f,0.0f,0.0f); }
        glVertex2f(curve.time(it)-s_in,curve.value(it)-s_in*curve.in_tangent(it));

        if (type == NODE) { glColor3f(1.0f,1.0f,0.0f); }
        else { glColor3f(0.0f,0.0f,0.0f); }
        glVertex2f(curve.time(it),curve.value(it));

        if (type == NODE || type == OUT_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
        else { glColor3f(0.0f,0.0f,0.0f); }
        glVertex2f(curve.time(it)+s_out,curve.value(it)+s_out*curve.out_tangent(it));

        glEnd();
    }

    glPopMatrix();
    glPopAttrib(); // GL_LINE_BIT
    glPopAttrib(); // GL_POINT_BIT
    glPopAttrib(); // GL_ENABLE_BIT
}

///////////////////////////////////////////////////////////////////////////////

static void
update()
{
    // Remove previous cameras
    for (unsigned int i = 0; i < tracker.trial()->cameras.size(); i++) {
        gtk_combo_box_remove_text(GTK_COMBO_BOX(drawingarea1_combobox), 1);
        gtk_combo_box_remove_text(GTK_COMBO_BOX(drawingarea2_combobox), 1);
    }

    // Add the new cameras
    for (unsigned int i = 1; i <= tracker.trial()->cameras.size(); i++) {
        std::stringstream ss;
        ss << "Camera " << i;
        gtk_combo_box_append_text(GTK_COMBO_BOX(drawingarea1_combobox),
                                  ss.str().c_str());
        gtk_combo_box_append_text(GTK_COMBO_BOX(drawingarea2_combobox),
                                  ss.str().c_str());
    }

    textures.resize(tracker.trial()->cameras.size());
    for (unsigned i = 0; i < textures.size(); i++) {
        glGenTextures(1,&textures[i]);
        glBindTexture(GL_TEXTURE_2D,textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     1,
                     tracker.trial()->videos.at(i).width(),
                     tracker.trial()->videos.at(i).height(),
                     0,
                     GL_LUMINANCE,
                    (tracker.trial()->videos.at(i).bps() == 8? GL_UNSIGNED_BYTE:
                                                               GL_UNSIGNED_SHORT),
                     tracker.trial()->videos.at(i).data());
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D,0);
    }

	// Setup the default view

	gtk_combo_box_set_active(GTK_COMBO_BOX(drawingarea1_combobox),0);
	gtk_combo_box_set_active(GTK_COMBO_BOX(drawingarea2_combobox),0);

    // Update the number of frames
    gtk_range_set_range(GTK_RANGE(timeline),0,tracker.trial()->num_frames);

    reset_graph(&position_graph);

    // Update the coordinate frames

    position_graph.min_frame = 0;
    position_graph.max_frame = tracker.trial()->num_frames-1;
    position_graph.frame_locks = vector<bool>(tracker.trial()->num_frames,false);

    GtkAdjustment* min_adj = (GtkAdjustment*)gtk_adjustment_new (0, 0, tracker.trial()->num_frames-1, 1, 1, 0);
    GtkAdjustment* max_adj = (GtkAdjustment*)gtk_adjustment_new (tracker.trial()->num_frames-1, 0, tracker.trial()->num_frames-1, 1, 1, 0);
    gtk_spin_button_set_adjustment(GTK_SPIN_BUTTON(min_timeline_spin_button), min_adj);
    gtk_spin_button_set_adjustment(GTK_SPIN_BUTTON(max_timeline_spin_button), max_adj);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(max_timeline_spin_button),tracker.trial()->num_frames-1);

    update_graph_min_max(&position_graph);

    frame_changed();
}

static void
enable_headlight()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);

    float position[4] = {0.0f,0.0f,0.0f,1.0f};
    glLightfv(GL_LIGHT0,GL_POSITION,position);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,1);

    float ambient[4] = {0.7f,0.7f,0.7f,1.0f};
    glMaterialfv(GL_FRONT,GL_AMBIENT,ambient);

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// Draws the pivot from the specifed view.
static void
draw_manip_from_view(const ViewData* view)
{
    if (movePivot) {
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2,0x3333);
    }

    glLineWidth(1.0);
    manip.set_size(view->scale*pivot_size);
    manip.draw();

    if (movePivot) {
        glLineStipple(1,0);
        glDisable(GL_LINE_STIPPLE);
    }
}

// Sets up the Viewport, Projection Matrix, and Modelview Matrix and draws the
// scene from the view specified in the ViewData.
static void
draw_view(const ViewData* view)
{
    update_viewport(const_cast<ViewData*>(view));

    glViewport(view->viewport_x,
               view->viewport_y,
               view->viewport_width,
               view->viewport_height);

    double m[16];
    if (view->cameraid == DEFAULT_CAMERA) {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glViewport(view->viewport_x,
                   view->viewport_y,
                   view->viewport_width,
                   view->viewport_height);

        defaultViewMatrix.inverse().to_matrix(m);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(view->fovy,view->ratio,view->near_clip,view->far_clip);
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixd(m);

        // Draw background
        draw_gradient(top_color,bot_color);

        // Draw image planes
        for (unsigned int i = 0; i < tracker.trial()->cameras.size(); ++i) {
            draw_textured_quad(tracker.trial()->cameras.at(i).image_plane(),
                               textures[i]);
        }

        // Draw cameras
        enable_headlight();
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        for (unsigned int i = 0; i < tracker.trial()->cameras.size(); ++i) {

            glPushMatrix();

            double m1[16];
            tracker.trial()->cameras.at(i).coord_frame().to_matrix(m1);
            glMultMatrixd(m1);

            float scale = 0.05*sqrt(m1[12]*m1[12]+m1[13]*m1[13]+m1[14]*m1[14]);
            glScalef(scale,scale,scale);

            glColor3f(0.5f, 0.5f, 0.5f);
            draw_camera();

            glPopMatrix();
        }

        draw_manip_from_view(view);
        glDisable(GL_LIGHTING);

        // Draw grid
        if (drawGrid == true) {
            draw_xz_grid(24, 24, 10.0f);
        }

        if (!tracker.views().empty()) {

            CoordFrame modelview = defaultViewMatrix.inverse()*manip_matrix()*volume_matrix;

            double imv[16];
            modelview.inverse().to_matrix_row_order(imv);
            tracker.view(0)->drrRenderer()->setInvModelView(imv);

            float width = 2.0f/view->zoom, height = 2.0f/view->zoom;
            float x = view->zoom_x-width/2.0f, y = view->zoom_y-height/2.0f;

            tracker.view(0)->drrRenderer()->setViewport(
                view->ratio*x, y, view->ratio*width, height);
            //tracker.view(view->cameraid)->radRenderer()->viewport(
            //    view->ratio*x, y, view->ratio*width, height);

            tracker.view(0)->renderDrr(view->pbo,view->window_width,view->window_height);

            glViewport(0, 0, view->window_width, view->window_height);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE,GL_ONE);

            glRasterPos2i(0, 0);
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, view->pbo);
            glDrawPixels(view->window_width,
                         view->window_height,
                         GL_RGB, GL_FLOAT, 0);
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
        }
        return;
    }
    else {

        CoordFrame modelview =
            tracker.trial()->cameras.at(view->cameraid).coord_frame().inverse()*
            manip_matrix()*volume_matrix;

        double imv[16];
        modelview.inverse().to_matrix_row_order(imv);
        tracker.view(view->cameraid)->drrRenderer()->setInvModelView(imv);

        float temp = 2.0f*sqrt(5.0)*sin(M_PI*view->fovy/360.0);
        float width = temp/view->zoom, height = temp/view->zoom;
        float x = view->zoom_x-width/2.0f, y = view->zoom_y-height/2.0f;

        tracker.view(view->cameraid)->drrRenderer()->setViewport(
            view->ratio*x, y, view->ratio*width, height);
        tracker.view(view->cameraid)->radRenderer()->set_viewport(
            view->ratio*x, y, view->ratio*width, height);

        tracker.view(view->cameraid)->render(view->pbo,
                                             view->window_width,
                                             view->window_height);

        glViewport(0, 0, view->window_width, view->window_height);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(0, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, view->pbo);
        glDrawPixels(view->window_width,
                     view->window_height,
                     GL_RGB, GL_FLOAT, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glEnable(GL_DEPTH_TEST);


        glViewport(view->viewport_x,
                   view->viewport_y,
                   view->viewport_width,
                   view->viewport_height);
        tracker.trial()->cameras.at(view->cameraid).
            coord_frame().inverse().to_matrix(m);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(view->fovy,view->ratio,view->near_clip,view->far_clip);
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixd(m);

        enable_headlight();
        draw_manip_from_view(view);
        glDisable(GL_LIGHTING);

        return;
    }
}

void trial_tree_view_update()
{
    GtkTreeModel* tree_model;
    GtkTreeIter camera_row_iter, camera_attrib_iter;

    tree_model = gtk_tree_view_get_model(GTK_TREE_VIEW(trial_tree_view));

    bool camera_flag =
        gtk_tree_model_get_iter_first(tree_model, &camera_row_iter);
    int camera_index = -1;
    while (camera_flag == true) {

        CoordFrame camera_view_matrix;
        if (camera_index == -1) {
            camera_view_matrix = defaultViewMatrix;
        }
        else {
            camera_view_matrix = tracker.trial()->cameras.at(camera_index).coord_frame();
        }


        stringstream ss_location;
        stringstream ss_look;
        stringstream ss_up;
        stringstream ss_right;

        ss_location.precision(3);
        ss_look.precision(3);
        ss_up.precision(3);
        ss_right.precision(3);

        ss_location << "Location ("
                    << camera_view_matrix.translation()[0] << ", "
                    << camera_view_matrix.translation()[1] << ", "
                    << camera_view_matrix.translation()[2] << ")";

        ss_look << "Look (" << camera_view_matrix.rotation()[6] << ", "
                            << camera_view_matrix.rotation()[7] << ", "
                            << camera_view_matrix.rotation()[8] << ")";

        ss_up << "Up (" << camera_view_matrix.rotation()[3] << ", "
                        << camera_view_matrix.rotation()[4] << ", "
                        << camera_view_matrix.rotation()[5] << ")";

        ss_right << "Right (" << camera_view_matrix.rotation()[0] << ", "
                              << camera_view_matrix.rotation()[1] << ", "
                              << camera_view_matrix.rotation()[2] << ")";

        gtk_tree_model_iter_nth_child(tree_model,
                                      &camera_attrib_iter,
                                      &camera_row_iter,
                                      LOCATION_ATTRIB_ROW);
        gtk_tree_store_set(GTK_TREE_STORE(tree_model),
                           &camera_attrib_iter,
                           0, ss_location.str().c_str(), -1);

        gtk_tree_model_iter_nth_child(tree_model,
                                      &camera_attrib_iter,
                                      &camera_row_iter,
                                      LOOK_ATTRIB_ROW);
        gtk_tree_store_set(GTK_TREE_STORE(tree_model),
                           &camera_attrib_iter,
                           0, ss_look.str().c_str(), -1);

        gtk_tree_model_iter_nth_child(tree_model,
                                      &camera_attrib_iter,
                                      &camera_row_iter,
                                      UP_ATTRIB_ROW);
        gtk_tree_store_set(GTK_TREE_STORE(tree_model),
                           &camera_attrib_iter,
                           0, ss_up.str().c_str(), -1);

        gtk_tree_model_iter_nth_child(tree_model,
                                      &camera_attrib_iter,
                                      &camera_row_iter,
                                      RIGHT_ATTRIB_ROW);
        gtk_tree_store_set(GTK_TREE_STORE(tree_model),
                           &camera_attrib_iter,
                           0, ss_right.str().c_str(), -1);

        camera_flag = gtk_tree_model_iter_next(tree_model, &camera_row_iter);
        camera_index++;
    }
}

void fill_notebook()
{
    GtkWidget* notebook =
        lookup_widget(window, "xromm_markerless_tracking_notebook");

    // Remove all the pages from the notebook.
    for (int i = 0; i < gtk_notebook_get_n_pages(GTK_NOTEBOOK(notebook)); ++i) {
        gtk_notebook_remove_page(GTK_NOTEBOOK(notebook),i);
    }

    // Insert the correct number of pages.
    for (int i = 0; i < 1; ++i) {

        // Create a scrolled window.
        GtkWidget* scrolled_window = gtk_scrolled_window_new(NULL, NULL);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window),
                                       GTK_POLICY_AUTOMATIC,
                                       GTK_POLICY_AUTOMATIC);
        gtk_widget_show(scrolled_window);

        GtkWidget* tree_view = xromm_gtk_tree_view_new_from_views(tracker.views());
        gtk_container_add(GTK_CONTAINER(scrolled_window), tree_view);

        // Create a label for the notebook tab.
        GtkWidget* label = gtk_label_new("View");
        gtk_widget_show(label);

        gtk_notebook_append_page(GTK_NOTEBOOK(notebook),
                                 scrolled_window,
                                 label);
    }
}

static bool first_undo = true;

void push_state()
{
    State current_state;
    current_state.x_curve = tracker.trial()->x_curve;
    current_state.y_curve = tracker.trial()->y_curve;
    current_state.z_curve = tracker.trial()->z_curve;
    current_state.x_rot_curve = tracker.trial()->yaw_curve;
    current_state.y_rot_curve = tracker.trial()->pitch_curve;
    current_state.z_rot_curve = tracker.trial()->roll_curve;

    history.push(current_state);

    first_undo = true;
    is_tracking_saved = false;
}

void undo_state()
{
    if (history.can_undo()) {

        if (first_undo) {
            push_state();
            history.undo();
            first_undo = false;
        }

        State undo_state = history.undo();

        tracker.trial()->x_curve = undo_state.x_curve;
        tracker.trial()->y_curve = undo_state.y_curve;
        tracker.trial()->z_curve = undo_state.z_curve;
        tracker.trial()->yaw_curve = undo_state.x_rot_curve;
        tracker.trial()->pitch_curve = undo_state.y_rot_curve;
        tracker.trial()->roll_curve = undo_state.z_rot_curve;

        selected_nodes.clear();

        update_graph_min_max(&position_graph);
        update_xyzypr_and_coord_frame();
        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}

void redo_state()
{
    if (history.can_redo()) {
        State redo_state = history.redo();

        tracker.trial()->x_curve = redo_state.x_curve;
        tracker.trial()->y_curve = redo_state.y_curve;
        tracker.trial()->z_curve = redo_state.z_curve;
        tracker.trial()->yaw_curve = redo_state.x_rot_curve;
        tracker.trial()->pitch_curve = redo_state.y_rot_curve;
        tracker.trial()->roll_curve = redo_state.z_rot_curve;

        selected_nodes.clear();

        update_graph_min_max(&position_graph);
        update_xyzypr_and_coord_frame();
        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ My Callbacks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

void
on_xromm_drr_renderer_properties_dialog_sample_distance_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
    double value = exp(7*gtk_range_get_value(range)-5);

	opencl::RayCaster* rayCaster = (opencl::RayCaster*)data;
    rayCaster->setSampleDistance(value);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_xromm_drr_renderer_properties_dialog_intensity_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
    double value = exp(15*gtk_range_get_value(range)-5);

    opencl::RayCaster* rayCaster = (opencl::RayCaster*)data;
    rayCaster->setRayIntensity(value);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}


void
on_xromm_drr_renderer_properties_dialog_cutoff_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
    opencl::RayCaster* rayCaster = (opencl::RayCaster*)data;
    float value = gtk_range_get_value(range);

    rayCaster->setCutoff(value*(rayCaster->getMaxCutoff()-
                                rayCaster->getMinCutoff())+
                         rayCaster->getMinCutoff());

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_xromm_sobel_properties_dialog_scale_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
	opencl::SobelFilter* sobelFilter = (opencl::SobelFilter*)data;
    sobelFilter->setScale(gtk_range_get_value(range));

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_xromm_sobel_properties_dialog_blend_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
	opencl::SobelFilter* sobelFilter = (opencl::SobelFilter*)data;
    sobelFilter->setBlend(gtk_range_get_value(range));

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_xromm_contrast_properties_dialog_alpha_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
	opencl::ContrastFilter* contrastFilter = (opencl::ContrastFilter*)data;
    contrastFilter->set_alpha(gtk_range_get_value(range));

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_xromm_gaussian_properties_dialog_radius_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
    opencl::GaussianFilter* gaussianFilter = (opencl::GaussianFilter*)data;
    gaussianFilter->set_radius(gtk_range_get_value(range));

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_xromm_sharpen_properties_dialog_radius_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
    opencl::SharpenFilter* sharpenFilter = (opencl::SharpenFilter*)data;
    sharpenFilter->set_radius(gtk_range_get_value(range));

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_xromm_sharpen_properties_dialog_contrast_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
    opencl::SharpenFilter* sharpenFilter = (opencl::SharpenFilter*)data;
    sharpenFilter->set_contrast(gtk_range_get_value(range));

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}


void
on_xromm_contrast_properties_dialog_beta_scale_value_changed
                                        (GtkRange*      range,
                                         gpointer       data)
{
    opencl::ContrastFilter* contrastFilter = (opencl::ContrastFilter*)data;
    contrastFilter->set_beta(gtk_range_get_value(range));

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_toggle_filter(gpointer filter, bool toggled)
{
    ((opencl::Filter*)filter)->set_enabled(toggled);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_toggle_renderer(gpointer view, gint type, bool toggled)
{
    if (type == 0) {
        ((opencl::View*)view)->drr_enabled = toggled;
    }
    else if (type == 1) {
        ((opencl::View*)view)->rad_enabled = toggled;
    }

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_remove_filter_activate(GtkWidget* menu_item, gpointer data)
{
    Args* args = static_cast<Args*>(data);
    vector<opencl::Filter*>::iterator it = args->filters->begin();
    for (; it != args->filters->end(); ++it) {
        if (*it == args->filter) {
            it = args->filters->erase(it);
            break;
        }
    }

    fill_notebook();
    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_new_filter_activate(GtkWidget* menu_item, gpointer data)
{
    vector<opencl::Filter*>* filters =
								static_cast<vector<opencl::Filter*>*>(data);
    string new_filter_name(gtk_label_get_text(
        GTK_LABEL(gtk_bin_get_child(GTK_BIN(menu_item)))));
    if (new_filter_name.compare("Sobel") == 0) {
        filters->push_back(new opencl::SobelFilter());
    }
    else if (new_filter_name.compare("Contrast") == 0) {
        filters->push_back(new opencl::ContrastFilter());
    }
    else if (new_filter_name.compare("Gaussian") == 0) {
        filters->push_back(new opencl::GaussianFilter());
    }
    else if (new_filter_name.compare("Sharpen") == 0) {
        filters->push_back(new opencl::SharpenFilter());
    }

    fill_notebook();
    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

void
on_export_view_activate                 (GtkWidget*     menu_item,
                                         gpointer       data)
{
    string filename = get_filename(true);
    if (filename.compare("") == 0) {
        return;
    }

    ofstream file(filename.c_str(), ios::out);
    if (!file) {
        cerr << "Export: Unable to open file for writing" << endl;
        return;
    }

    opencl::View* view = (opencl::View*)data;
    vector<opencl::Filter*>::const_iterator iter;

    file << "DrrRenderer_begin" << endl;
    file << "SampleDistance " << view->drrRenderer()->getSampleDistance() << endl;
    file << "RayIntensity " << view->drrRenderer()->getRayIntensity() << endl;
    file << "Cutoff " << view->drrRenderer()->getCutoff() << endl;
    file << "DrrRenderer_end" << endl;
    file << "DrrFilters_begin" << endl;
    for (iter = view->drrFilters().begin();
         iter != view->drrFilters().end();
         ++iter) {
        switch ((*iter)->type()) {
            case opencl::Filter::XROMM_OPENCL_SOBEL_FILTER:
                file << "SobelFilter_begin" << endl;
                file << "Scale "
				     << ((opencl::SobelFilter*)(*iter))->scale() << endl;
                file << "Blend "
				     << ((opencl::SobelFilter*)(*iter))->blend() << endl;
                file << "SobelFilter_end" << endl;
                break;
            case opencl::Filter::XROMM_OPENCL_CONTRAST_FILTER:
                file << "ContrastFilter_begin" << endl;
                file << "Alpha "
				     << ((opencl::ContrastFilter*)(*iter))->alpha() << endl;
                file << "Beta "
				     << ((opencl::ContrastFilter*)(*iter))->beta() << endl;
                file << "ContrastFilter_end" << endl;
                break;
            default: break;
        }
    }
    file << "DrrFilters_end" << endl;
    file << "RadFilters_begin" << endl;
    for (iter = view->radFilters().begin();
         iter != view->radFilters().end();
         ++iter) {
        switch ((*iter)->type()) {
            case opencl::Filter::XROMM_OPENCL_SOBEL_FILTER:
                file << "SobelFilter_begin" << endl;
                file << "Scale "
				     << ((opencl::SobelFilter*)(*iter))->scale() << endl;
                file << "Blend "
				     << ((opencl::SobelFilter*)(*iter))->blend() << endl;
                file << "SobelFilter_end" << endl;
                break;
            case opencl::Filter::XROMM_OPENCL_CONTRAST_FILTER:
                file << "ContrastFilter_begin" << endl;
                file << "Alpha "
				     << ((opencl::ContrastFilter*)(*iter))->alpha() << endl;
                file << "Beta "
				     << ((opencl::ContrastFilter*)(*iter))->beta() << endl;
                file << "ContrastFilter_end" << endl;
                break;
            default: break;
        }
    }
    file << "RadFilters_end" << endl;

    file.close();
}



void
on_import_view_activate                 (GtkWidget*     menu_item,
                                         gpointer       data)
{
    string filename = get_filename();
    if (filename.compare("") == 0) {
        return;
    }

    ifstream file(filename.c_str(), ios::in);
    if (!file) {
        cerr << "Import: Unable to open file for writing" << endl;
        return;
    }

    opencl::View* view = (opencl::View*)data;

    string line, key;
    while (getline(file,line)) {
        if (line.compare("DrrRenderer_begin") == 0) {
            while (getline(file,line) && line.compare("DrrRenderer_end") != 0) {
                istringstream lineStream(line);
                lineStream >> key;
                if (key.compare("SampleDistance") == 0) {
                    float sampleDistance;
                    lineStream >> sampleDistance;
                    view->drrRenderer()->setSampleDistance(sampleDistance);
                }
                else if (key.compare("RayIntensity") == 0) {
                    float rayIntensity;
                    lineStream >> rayIntensity;
                    view->drrRenderer()->setRayIntensity(rayIntensity);
                }
                else if (key.compare("Cutoff") == 0) {
                    float cutoff;
                    lineStream >> cutoff;
                    view->drrRenderer()->setCutoff(cutoff);
                }
            }
        }
        else if (line.compare("DrrFilters_begin") == 0) {
            view->drrFilters().clear(); //XXX Memory Leak
            while (getline(file,line) && line.compare("DrrFilters_end") != 0) {
                if (line.compare("SobelFilter_begin") == 0) {
                    opencl::SobelFilter* filter = new opencl::SobelFilter();
                    while (getline(file,line) && line.compare("SobelFilter_end") != 0) {
                        istringstream lineStream(line);
                        lineStream >> key;
                        if (key.compare("Scale") == 0) {
                            float scale;
                            lineStream >> scale;
                            filter->setScale(scale);
                        }
                        else if (key.compare("Blend") == 0) {
                            float blend;
                            lineStream >> blend;
                            filter->setBlend(blend);
                        }
                    }
                    view->drrFilters().push_back(filter);
                }
                else if (line.compare("ContrastFilter_begin") == 0) {
                    opencl::ContrastFilter* filter = new opencl::ContrastFilter();
                    while (getline(file,line) && line.compare("ContrastFilter_end") != 0) {
                        istringstream lineStream(line);
                        lineStream >> key;
                        if (key.compare("Alpha") == 0) {
                            float alpha;
                            lineStream >> alpha;
                            filter->set_alpha(alpha);
                        }
                        else if (key.compare("Beta") == 0) {
                            float beta;
                            lineStream >> beta;
                            filter->set_beta(beta);
                        }
                    }
                    view->drrFilters().push_back(filter);
                }
            }
        }
        else if (line.compare("RadFilters_begin") == 0) {
            view->radFilters().clear(); //XXX Memory Leak
            while (getline(file,line) && line.compare("RadFilters_end") != 0) {
                if (line.compare("SobelFilter_begin") == 0) {
                    opencl::SobelFilter* filter = new opencl::SobelFilter();
                    while (getline(file,line) && line.compare("SobelFilter_end") != 0) {
                        istringstream lineStream(line);
                        lineStream >> key;
                        if (key.compare("Scale") == 0) {
                            float scale;
                            lineStream >> scale;
                            filter->setScale(scale);
                        }
                        else if (key.compare("Blend") == 0) {
                            float blend;
                            lineStream >> blend;
                            filter->setBlend(blend);
                        }
                    }
                    view->radFilters().push_back(filter);
                }
                else if (line.compare("ContrastFilter_begin") == 0) {
                    opencl::ContrastFilter* filter = new opencl::ContrastFilter();
                    while (getline(file,line) && line.compare("ContrastFilter_end") != 0) {
                        istringstream lineStream(line);
                        lineStream >> key;
                        if (key.compare("Alpha") == 0) {
                            float alpha;
                            lineStream >> alpha;
                            filter->set_alpha(alpha);
                        }
                        else if (key.compare("Beta") == 0) {
                            float beta;
                            lineStream >> beta;
                            filter->set_beta(beta);
                        }
                    }
                    view->radFilters().push_back(filter);
                }
            }
        }
    }
    file.close();

    fill_notebook();
    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Glade Callbacks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

gboolean
on_xromm_markerless_tracking_window_delete_event
                                       (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data)
{
    gtk_main_quit();

    return FALSE;
}


void
on_xromm_markerless_tracking_drawingarea1_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    // Save a pointer to the drawing area.
    drawingarea1 = widget;

    // Initialize the drawing area view
    drawingarea1_view.cameraid = DEFAULT_CAMERA;

    drawingarea1_view.ratio = 1.0f;
    drawingarea1_view.fovy = 53.13f;
    drawingarea1_view.near_clip = 1.0f;
    drawingarea1_view.far_clip = 10000.0f;

    drawingarea1_view.zoom = 1.0f;
    drawingarea1_view.zoom_x = 0.0f;
    drawingarea1_view.zoom_y = 0.0f;

    drawingarea1_view.pbo = 0;

    // Initialize the opengl context and state
    if (glcontext == NULL)
	{
        glcontext = gtk_widget_get_gl_context(widget);
        GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);

        // Begin opengl calls.
        if (gdk_gl_drawable_gl_begin(gldrawable, glcontext) != TRUE) {
            return;
        }

        init();

        // End opengl calls.
        gdk_gl_drawable_gl_end(gldrawable);

		redraw_drawingarea(drawingarea1);
		redraw_drawingarea(drawingarea2);
		redraw_drawingarea(graph_drawingarea);
    }

	// XXX: This is necessary because on windows configure is called before
	// realize, so the opengl context has not been set up.
	on_xromm_markerless_tracking_drawingarea1_configure_event(widget,0,0);
}


gboolean
on_xromm_markerless_tracking_drawingarea1_configure_event
                                       (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data)
{
    // Return if the widget is not drawable.
    GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);
    if (!GDK_IS_GL_DRAWABLE(gldrawable)) {
        return FALSE;
    }

    // Resize the view. Note that this function calls
    // update viewport.
    resize_view(&drawingarea1_view,
                widget->allocation.width,
                widget->allocation.height);

    update_scale_in_view(&drawingarea1_view);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea1_expose_event
                                       (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data)
{
    GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);

    // glBegin
    if (!gdk_gl_drawable_gl_begin(gldrawable, glcontext)) {
        return FALSE;
    }

    update_scale_in_view(&drawingarea1_view);
    draw_view(&drawingarea1_view);

    if (gdk_gl_drawable_is_double_buffered(gldrawable)) {
        gdk_gl_drawable_swap_buffers(gldrawable);
    }

    // glEnd
    gdk_gl_drawable_gl_end(gldrawable);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea1_button_press_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
    press_x = event->x;
    press_y = event->y;

    select_manip_in_view(&drawingarea1_view,event->x,event->y,event->button);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea1_button_release_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
    redraw_drawingarea(drawingarea1);
    manip.on_mouse_release(event->x,event->y);

    update_graph_min_max(&position_graph,tracker.trial()->frame);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea1_motion_notify_event
                                       (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data)
{
    static double prevx = event->x;
    static double prevy = event->y;

    double dx = event->x - prevx;
    double dy = event->y - prevy;

    if ((event->state & GDK_CONTROL_MASK) == GDK_CONTROL_MASK) {
        if (drawingarea1_view.cameraid == DEFAULT_CAMERA) {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                CoordFrame rotationMatrix;
                rotationMatrix.rotate(defaultViewMatrix.rotation()+3,
                                             -dx/2.0);
                rotationMatrix.rotate(defaultViewMatrix.rotation()+0,
                                             -dy/2.0);

                defaultViewMatrix = rotationMatrix*defaultViewMatrix;
            }
            else if ((event->state & GDK_BUTTON2_MASK) == GDK_BUTTON2_MASK) {
                double xtrans[3] = {-dx*defaultViewMatrix.rotation()[0],
                                    -dx*defaultViewMatrix.rotation()[1],
                                    -dx*defaultViewMatrix.rotation()[2]};
                double ytrans[3] = {dy*defaultViewMatrix.rotation()[3],
                                    dy*defaultViewMatrix.rotation()[4],
                                    dy*defaultViewMatrix.rotation()[5]};

                defaultViewMatrix.translate(xtrans);
                defaultViewMatrix.translate(ytrans);
            }
            else if ((event->state & GDK_BUTTON3_MASK) == GDK_BUTTON3_MASK) {
                double ztrans[3] =
                    { (dx-dy)/2.0*defaultViewMatrix.rotation()[6],
                      (dx-dy)/2.0*defaultViewMatrix.rotation()[7],
                      (dx-dy)/2.0*defaultViewMatrix.rotation()[8] };

                defaultViewMatrix.translate(ztrans);
            }
        }
        else {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                drawingarea1_view.zoom_x -= dx/200/drawingarea1_view.zoom;
                drawingarea1_view.zoom_y += dy/200/drawingarea1_view.zoom;

                update_viewport(&drawingarea1_view);
            }
        }
        update_scale_in_view(&drawingarea1_view);
    }
    else {
        if ((event->state & GDK_SHIFT_MASK) == GDK_SHIFT_MASK) {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                // Only display in one direction
                if (abs(event->x-press_x) > abs(event->y-press_y)) {
                    event->y = press_y;
                }
                else {
                    event->x = press_x;
                }
                move_manip_in_view(&drawingarea2_view,event->x,event->y);
            }
        }
        else {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                move_manip_in_view(&drawingarea1_view,event->x,event->y);
            }
            else if ((event->state & GDK_BUTTON3_MASK) == GDK_BUTTON3_MASK) {
                move_manip_in_view(&drawingarea1_view,dx,dy,true);
            }
        }
    }

    update_xyzypr();
    //update_graph();

    prevx = event->x;
    prevy = event->y;

    // Invalidate both windows.
    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);

    return TRUE;
}


void
on_xromm_markerless_tracking_drawingarea2_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    // Save a pointer to the drawingarea.
    drawingarea2 = widget;

    // Initialize the drawing area view
    drawingarea2_view.cameraid = DEFAULT_CAMERA;

    drawingarea2_view.ratio = 1.0f;
    drawingarea2_view.fovy = 53.13f;
    drawingarea2_view.near_clip = 1.0f;
    drawingarea2_view.far_clip = 10000.0f;

    drawingarea2_view.zoom = 1.0f;
    drawingarea2_view.zoom_x = 0.0f;
    drawingarea2_view.zoom_y = 0.0f;

    drawingarea2_view.pbo = 0;

    // Initialize the opengl context and state if it has not already been done.
    if (glcontext == NULL) {

        glcontext = gtk_widget_get_gl_context(widget);
        GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);

        // Begin opengl calls.
        if (gdk_gl_drawable_gl_begin(gldrawable, glcontext) != TRUE) {
            return;
        }

        init();

        // End opengl calls.
        gdk_gl_drawable_gl_end(gldrawable);

		redraw_drawingarea(drawingarea1);
		redraw_drawingarea(drawingarea2);
		redraw_drawingarea(graph_drawingarea);
    }

	// XXX: This is necessary because on windows configure is called before
	// realize, so the opengl context has not been set up.
	on_xromm_markerless_tracking_drawingarea2_configure_event(widget,0,0);
}


gboolean
on_xromm_markerless_tracking_drawingarea2_configure_event
                                       (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data)
{
    GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);
    if (!GDK_IS_GL_DRAWABLE(gldrawable)) {
        return FALSE;
    }

    // Resize the view. Note that this function calls
    // update viewport.
    resize_view(&drawingarea2_view,
                widget->allocation.width,
                widget->allocation.height);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea2_expose_event
                                       (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data)
{
    GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);

    // glBegin
    if (!gdk_gl_drawable_gl_begin(gldrawable, glcontext)) {
        return FALSE;
    }

    update_scale_in_view(&drawingarea2_view);
    draw_view(&drawingarea2_view);

    if (gdk_gl_drawable_is_double_buffered(gldrawable)) {
        gdk_gl_drawable_swap_buffers(gldrawable);
    }

    // glEnd
    gdk_gl_drawable_gl_end(gldrawable);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea2_button_press_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
    press_x = event->x;
    press_y = event->y;

    select_manip_in_view(&drawingarea2_view,event->x,event->y,event->button);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea2_button_release_event
                                       (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
    redraw_drawingarea(drawingarea2);
    manip.on_mouse_release(event->x,event->y);

    update_graph_min_max(&position_graph,tracker.trial()->frame);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea2_motion_notify_event
                                       (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data)
{
    CoordFrame viewMatrix = defaultViewMatrix;
    if (drawingarea2_view.cameraid != DEFAULT_CAMERA) {
        viewMatrix = tracker.trial()->cameras.at(drawingarea2_view.cameraid).coord_frame();
    }

    static double prevx = event->x;
    static double prevy = event->y;

    double dx = event->x - prevx;
    double dy = event->y - prevy;

    if ((event->state & GDK_CONTROL_MASK) == GDK_CONTROL_MASK) {
        if (drawingarea2_view.cameraid == DEFAULT_CAMERA) {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                CoordFrame rotationMatrix;
                rotationMatrix.rotate(defaultViewMatrix.rotation()+3,
                                             -dx/2.0);
                rotationMatrix.rotate(defaultViewMatrix.rotation()+0,
                                             -dy/2.0);

                defaultViewMatrix = rotationMatrix*defaultViewMatrix;
            }
            else if ((event->state & GDK_BUTTON2_MASK) == GDK_BUTTON2_MASK) {
                double xtrans[3] = {-dx*defaultViewMatrix.rotation()[0],
                                    -dx*defaultViewMatrix.rotation()[1],
                                    -dx*defaultViewMatrix.rotation()[2]};
                double ytrans[3] = {dy*defaultViewMatrix.rotation()[3],
                                    dy*defaultViewMatrix.rotation()[4],
                                    dy*defaultViewMatrix.rotation()[5]};

                defaultViewMatrix.translate(xtrans);
                defaultViewMatrix.translate(ytrans);
            }
            else if ((event->state & GDK_BUTTON3_MASK) == GDK_BUTTON3_MASK) {
                double ztrans[3] =
                    { (dx-dy)/2.0*defaultViewMatrix.rotation()[6],
                      (dx-dy)/2.0*defaultViewMatrix.rotation()[7],
                      (dx-dy)/2.0*defaultViewMatrix.rotation()[8] };

                defaultViewMatrix.translate(ztrans);
            }
        }
        else {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                drawingarea2_view.zoom_x -= dx/200/drawingarea2_view.zoom;
                drawingarea2_view.zoom_y += dy/200/drawingarea2_view.zoom;

                update_viewport(&drawingarea2_view);
            }
        }
    }
    else {
        if ((event->state & GDK_SHIFT_MASK) == GDK_SHIFT_MASK) {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                // Only display in one direction
                if (abs(event->x-press_x) > abs(event->y-press_y)) {
                    event->y = press_y;
                }
                else {
                    event->x = press_x;
                }
                move_manip_in_view(&drawingarea2_view,event->x,event->y);
            }
        }
        else {
            if ((event->state & GDK_BUTTON1_MASK) == GDK_BUTTON1_MASK) {
                move_manip_in_view(&drawingarea2_view,event->x,event->y);
            }
            else if ((event->state & GDK_BUTTON3_MASK) == GDK_BUTTON3_MASK) {
                move_manip_in_view(&drawingarea2_view,dx,dy,true);
            }
        }
    }

    update_xyzypr();
    //update_graph();

    prevx = event->x;
    prevy = event->y;

    // Invalidate both windows.
    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);

    return TRUE;
}


void
on_xromm_markerless_tracking_graph_drawingarea_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    graph_drawingarea = widget;

    // Initialize the drawing area view
    graph_view.zoom = 1.0f;
    graph_view.zoom_x = 0.0f;
    graph_view.zoom_y = 0.0f;

    position_graph.show_x = true;
    position_graph.show_y = true;
    position_graph.show_z = true;
    position_graph.show_yaw = true;
    position_graph.show_pitch = true;
    position_graph.show_roll = true;
    position_graph.min_frame = 0.0;
    position_graph.max_frame = 100.0;
    position_graph.min_value = -180.0;
    position_graph.max_value = 180.0;
    position_graph.frame_locks.resize(100,false);

    // Initialize the opengl context and state if it has not already been done.
    if (glcontext == NULL) {

        glcontext = gtk_widget_get_gl_context(widget);
        GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);

        // Begin opengl calls.
        if (gdk_gl_drawable_gl_begin(gldrawable, glcontext) != TRUE) {
            return;
        }

        init();

        // End opengl calls.
        gdk_gl_drawable_gl_end(gldrawable);

		redraw_drawingarea(drawingarea1);
		redraw_drawingarea(drawingarea2);
		redraw_drawingarea(graph_drawingarea);
    }

	// XXX: This is necessary because on windows configure is called before
	// realize, so the opengl context has not been set up.
	on_xromm_markerless_tracking_graph_drawingarea_configure_event(widget,0,0);
}


gboolean
on_xromm_markerless_tracking_graph_drawingarea_configure_event
                                       (GtkWidget       *widget,
                                        GdkEventConfigure *event,
                                        gpointer         user_data)
{
    // Return if the widget is not drawable.
    GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);
    if (!GDK_IS_GL_DRAWABLE(gldrawable)) {
        return FALSE;
    }

    // Resize the view. Note that this function calls
    // update viewport.
    resize_view(&graph_view,
                widget->allocation.width,
                widget->allocation.height);

    return TRUE;
}

gboolean
on_xromm_markerless_tracking_graph_drawingarea_expose_event
                                       (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data)
{
    GdkGLDrawable *gldrawable = gtk_widget_get_gl_drawable(widget);

    // glBegin
    if (!gdk_gl_drawable_gl_begin(gldrawable, glcontext)) {
        return FALSE;
    }

    draw_graph(&graph_view,&position_graph);

    if (gdk_gl_drawable_is_double_buffered(gldrawable)) {
        gdk_gl_drawable_swap_buffers(gldrawable);
    }

    // glEnd
    gdk_gl_drawable_gl_end(gldrawable);

    return TRUE;
}


void
on_xromm_markerless_tracking_toolbar_new_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{

}


void
on_xromm_markerless_tracking_toolbar_save_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
    string filename = get_filename(true);
    if (filename.compare("") != 0) {
        save_tracking_results(filename);
    }
}


void
on_xromm_markerless_tracking_saveas_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
    string filename = get_filename();
    if (filename.compare("") != 0) {
        load_tracking_results(filename);
    }
}


void
on_xromm_markerless_tracking_toolbar_add_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{

}


void
on_xromm_markerless_tracking_drawingarea1_toolbar_combo_box_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    drawingarea1_combobox = widget;
    gtk_combo_box_set_active(GTK_COMBO_BOX(widget),0);
}


void
on_xromm_markerless_tracking_drawingarea1_toolbar_combo_box_changed
                                       (GtkComboBox     *combobox,
                                        gpointer         user_data)
{
    drawingarea1_view.cameraid = CameraID(gtk_combo_box_get_active(combobox)-1);
	if (drawingarea1_view.cameraid < DEFAULT_CAMERA) {
		drawingarea1_view.cameraid = DEFAULT_CAMERA;
	}

    // Reset the viewport
    drawingarea1_view.viewport_x = 0;
    drawingarea1_view.viewport_y = 0;
    drawingarea1_view.viewport_width = drawingarea1_view.window_width;
    drawingarea1_view.viewport_height = drawingarea1_view.window_height;

    // XXX: The field of view for the drr is pretty small--we should decrease the
    // default accordingly, but this causes problems with the zoom.
    //drawingarea1_view.fovy = drawingarea1_view.cameraid == DEFAULT_CAMERA? 53.13: 15;

    redraw_drawingarea(drawingarea1);
}


void
on_xromm_markerless_tracking_drawingarea2_toolbar_combo_box_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    drawingarea2_combobox = widget;
    gtk_combo_box_set_active(GTK_COMBO_BOX(widget),0);
}


void
on_xromm_markerless_tracking_drawingarea2_toolbar_combo_box_changed
                                       (GtkComboBox     *combobox,
                                        gpointer         user_data)
{
    drawingarea2_view.cameraid = CameraID(gtk_combo_box_get_active(combobox)-1);
	if (drawingarea2_view.cameraid < DEFAULT_CAMERA) {
		drawingarea2_view.cameraid = DEFAULT_CAMERA;
	}

    // Reset the viewport
    drawingarea2_view.viewport_x = 0;
    drawingarea2_view.viewport_y = 0;
    drawingarea2_view.viewport_width = drawingarea2_view.window_width;
    drawingarea2_view.viewport_height = drawingarea2_view.window_height;

    // XXX: The field of view for the drr is pretty small--we should decrease the
    // default accordingly, but this causes problems with the zoom.
    //drawingarea2_view.fovy = drawingarea2_view.cameraid == DEFAULT_CAMERA? 53.13: 15;

    redraw_drawingarea(drawingarea2);
}


gboolean
on_xromm_markerless_tracking_drawingarea1_scroll_event
                                       (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data)
{
    if (drawingarea1_view.cameraid == DEFAULT_CAMERA) {
        return FALSE;
    }

    if ((event->scroll.state & GDK_CONTROL_MASK) == GDK_CONTROL_MASK) {
        if (event->scroll.direction == GDK_SCROLL_UP) {
            drawingarea1_view.zoom *= 1.1f;
        }
        else if (event->scroll.direction == GDK_SCROLL_DOWN) {
            drawingarea1_view.zoom /= 1.1f;
        }

        update_viewport(&drawingarea1_view);
        redraw_drawingarea(drawingarea1);
    }

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_drawingarea2_scroll_event
                                       (GtkWidget       *widget,
                                        GdkEvent        *event,
                                        gpointer         user_data)
{
    if (drawingarea2_view.cameraid == DEFAULT_CAMERA) {
        return FALSE;
    }

    if ((event->scroll.state & GDK_CONTROL_MASK) == GDK_CONTROL_MASK) {
        if (event->scroll.direction == GDK_SCROLL_UP) {
            drawingarea2_view.zoom *= 1.1f;
        }
        else if (event->scroll.direction == GDK_SCROLL_DOWN) {
            drawingarea2_view.zoom /= 1.1f;
        }

        update_viewport(&drawingarea2_view);
        redraw_drawingarea(drawingarea2);
    }

    return TRUE;
}


void
on_xromm_markerless_tracking_toolbar_translate_radiobutton_toggled
                                       (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data)
{
    manip.set_mode(Manip3D::TRANSLATION);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}


void
on_xromm_markerless_tracking_toolbar_rotate_radiobutton_toggled
                                       (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data)
{
    manip.set_mode(Manip3D::ROTATION);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}


// TODO: Remove this function
void
on_xromm_markerless_tracking_drawingarea1_toolbar_global_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    //XXX drawingarea1_view.interaction_mode = GLOBAL;

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

// TODO: Remove this function
void
on_xromm_markerless_tracking_drawingarea1_toolbar_inplane_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    //XXX drawingarea1_view.interaction_mode = IN_PLANE;

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

// TODO: Remove this function
void
on_xromm_markerless_tracking_drawingarea2_toolbar_global_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    //XXX drawingarea2_view.interaction_mode = GLOBAL;

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}

// TODO: Remove this function
void
on_xromm_markerless_tracking_drawingarea2_toolbar_inplane_radiobutton_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    //XXX drawingarea2_view.interaction_mode = IN_PLANE;

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}


void
on_show_grid_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    if (gtk_check_menu_item_get_active(GTK_CHECK_MENU_ITEM(menuitem)) == TRUE) {
        drawGrid = true;
    }
    else {
        drawGrid = false;
    }

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}


void
on_toggletoolbutton1_toggled           (GtkToggleToolButton *toggletoolbutton,
                                        gpointer         user_data)
{
    movePivot = gtk_toggle_tool_button_get_active(toggletoolbutton) == TRUE?
                true: false;

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
}


void
on_xromm_markerless_tracking_timeline_value_changed
                                       (GtkRange        *range,
                                        gpointer         user_data)
{
    tracker.trial()->frame = (int)gtk_range_get_value(range);
    frame_changed();
}


void
on_xromm_markerless_tracking_timeline_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    timeline = widget;
}

void
on_xromm_markerless_tracking_graph_toolbar_x_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    position_graph.show_x = gtk_toggle_button_get_active(togglebutton);

    update_graph_min_max(&position_graph);

    redraw_drawingarea(graph_drawingarea);
}


void
on_xromm_markerless_tracking_graph_toolbar_y_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    position_graph.show_y = gtk_toggle_button_get_active(togglebutton);

    update_graph_min_max(&position_graph);

    redraw_drawingarea(graph_drawingarea);
}


void
on_xromm_markerless_tracking_graph_toolbar_z_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    position_graph.show_z = gtk_toggle_button_get_active(togglebutton);

    update_graph_min_max(&position_graph);

    redraw_drawingarea(graph_drawingarea);
}


void
on_xromm_markerless_tracking_graph_toolbar_yaw_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    position_graph.show_yaw = gtk_toggle_button_get_active(togglebutton);

    update_graph_min_max(&position_graph);

    redraw_drawingarea(graph_drawingarea);
}


void
on_xromm_markerless_tracking_graph_toolbar_pitch_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    position_graph.show_pitch = gtk_toggle_button_get_active(togglebutton);

    update_graph_min_max(&position_graph);

    redraw_drawingarea(graph_drawingarea);
}


void
on_xromm_markerless_tracking_graph_toolbar_roll_check_button_toggled
                                       (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    position_graph.show_roll = gtk_toggle_button_get_active(togglebutton);

    update_graph_min_max(&position_graph);

    redraw_drawingarea(graph_drawingarea);
}



void
on_xromm_markerless_tracking_graph_toolbar_x_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    x_spin_button = widget;
}


void
on_xromm_markerless_tracking_graph_toolbar_x_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    if (spin_button_update == true) {
        update_coord_frame();
        update_graph_min_max(&position_graph);

        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}


void
on_xromm_markerless_tracking_graph_toolbar_y_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    y_spin_button = widget;
}


void
on_xromm_markerless_tracking_graph_toolbar_y_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    if (spin_button_update == true) {
        update_coord_frame();
        update_graph_min_max(&position_graph);

        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}


void
on_xromm_markerless_tracking_graph_toolbar_z_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    z_spin_button = widget;
}


void
on_xromm_markerless_tracking_graph_toolbar_z_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    if (spin_button_update == true) {
        update_coord_frame();
        update_graph_min_max(&position_graph);

        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }

}


void
on_xromm_markerless_tracking_graph_toolbar_yaw_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    yaw_spin_button = widget;
}


void
on_xromm_markerless_tracking_graph_toolbar_yaw_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    if (spin_button_update == true) {
        update_coord_frame();
        update_graph_min_max(&position_graph);

        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}


void
on_xromm_markerless_tracking_graph_toolbar_pitch_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    pitch_spin_button = widget;
}


void
on_xromm_markerless_tracking_graph_toolbar_pitch_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    if (spin_button_update == true) {
        update_coord_frame();
        update_graph_min_max(&position_graph);

        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}


void
on_xromm_markerless_tracking_graph_toolbar_roll_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    roll_spin_button = widget;

}


void
on_xromm_markerless_tracking_graph_toolbar_roll_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    if (spin_button_update == true) {
        update_coord_frame();
        update_graph_min_max(&position_graph);

        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}


void
on_xromm_markerless_tracking_timeline_min_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    gdouble new_min = gtk_spin_button_get_value(spinbutton);
    gtk_spin_button_set_range(
        GTK_SPIN_BUTTON(lookup_widget(window,
                        "xromm_markerless_tracking_timeline_max_spin_button")),
        new_min+1, tracker.trial()->num_frames);

    position_graph.min_frame = new_min;
    update_graph_min_max(&position_graph);
    redraw_drawingarea(graph_drawingarea);
}


void
on_xromm_markerless_tracking_timeline_min_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    min_timeline_spin_button = widget;
}


void
on_xromm_markerless_tracking_timeline_max_spin_button_value_changed
                                       (GtkSpinButton   *spinbutton,
                                        gpointer         user_data)
{
    gdouble new_max = gtk_spin_button_get_value(spinbutton);
    gtk_spin_button_set_range(
        GTK_SPIN_BUTTON(lookup_widget(window,
                        "xromm_markerless_tracking_timeline_min_spin_button")),
        0, new_max-1);

    position_graph.max_frame = new_max;
    update_graph_min_max(&position_graph);
    redraw_drawingarea(graph_drawingarea);
}


void
on_xromm_markerless_tracking_timeline_max_spin_button_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    max_timeline_spin_button = widget;
}

void
on_xromm_markerless_tracking_notebook_views_treeview_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
    trial_tree_view = widget;
/*
    GtkTreeView* tree_view = GTK_TREE_VIEW(widget);
    gtk_tree_view_set_headers_visible(tree_view, false);

    // Create a column for names
    GtkTreeViewColumn* name_column = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(name_column, "Name");
    gtk_tree_view_append_column(tree_view, name_column);

    // Setup a text renderer for the name column
    GtkCellRenderer* name_renderer = gtk_cell_renderer_text_new();
    gtk_tree_view_column_pack_start(name_column, name_renderer, TRUE);
    gtk_tree_view_column_add_attribute(name_column, name_renderer,
                                       "text", 0);

    // Create a column for locations
    GtkTreeViewColumn* location_column = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(location_column, "Location");
    gtk_tree_view_append_column(tree_view, location_column);

    // Setup a text renderer for the location column
    GtkCellRenderer* location_renderer = gtk_cell_renderer_text_new();
    gtk_tree_view_column_pack_start(location_column, location_renderer,
                                    TRUE);
    gtk_tree_view_column_add_attribute(location_column, location_renderer,
                                       "text", 1);

    // Create a new tree store and model for the view
    GtkTreeStore* tree_store = gtk_tree_store_new(2, G_TYPE_STRING,
                                                  G_TYPE_STRING);
    GtkTreeModel* tree_model = GTK_TREE_MODEL(tree_store);
    gtk_tree_view_set_model(tree_view, tree_model);

    // destroy model automatically with view
    g_object_unref(tree_model);

    //views_treeview_reset();

    gtk_tree_selection_set_mode(gtk_tree_view_get_selection(tree_view),
                                GTK_SELECTION_NONE);
*/
}


void
on_xromm_markerless_tracking_notebook_volumes_treeview_realize
                                       (GtkWidget       *widget,
                                        gpointer         user_data)
{
/*
    trial_tree_view = widget;

    GtkTreeViewColumn* column;
    GtkCellRenderer* renderer;
    GtkTreeStore* treestore;
    GtkTreeModel* model;

    // Column 1
    column = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(column, "Views");
    gtk_tree_view_append_column(GTK_TREE_VIEW(widget), column);

    renderer = gtk_cell_renderer_text_new();
    gtk_tree_view_column_pack_start(column, renderer, TRUE);
    gtk_tree_view_column_add_attribute(column, renderer, "text", 0);

    treestore = gtk_tree_store_new(1, G_TYPE_STRING);

    model = GTK_TREE_MODEL(treestore);

    gtk_tree_view_set_model(GTK_TREE_VIEW(widget), model);

    g_object_unref(model); // destroy model automatically with view

    views_treeview_reset();

    gtk_tree_selection_set_mode(
        gtk_tree_view_get_selection(GTK_TREE_VIEW(widget)), GTK_SELECTION_NONE);
*/
}

gint idle_frame_optimize_id = 0;
int idle_frame, idle_from_frame, idle_to_frame, d_frame;
bool idle_exit = false;
int idle_num_repeats = 1;

void
idle_frame_optimize(gpointer* data)
{
    if (idle_frame == -1) {
        return;
    }

    GtkWidget* pbar =
        lookup_widget(tracking_dialog,
                      "xromm_markerless_tracking_tracking_progressbar");
    if (idle_frame != idle_to_frame+d_frame && !idle_exit) {

         if (position_graph.frame_locks.at(idle_frame)) {
            idle_frame += d_frame;
            return;
         }

         tracker.optimize(idle_frame, d_frame, idle_num_repeats);

         update_graph_min_max(&position_graph, tracker.trial()->frame);

         frame_changed();

         //update progress bar
         gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(pbar),
                 (abs(idle_frame-idle_from_frame)+1.0f)/
                 (abs(idle_to_frame-idle_from_frame)+1.0f));

         idle_frame += d_frame;
    }
    else {
        gtk_idle_remove(idle_frame_optimize_id);
        idle_frame_optimize_id = 0;

        gtk_widget_hide(tracking_dialog);
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(pbar),0.0f);
    }
}

void
on_xromm_markerless_tracking_toolbar_track_button_clicked
                                       (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
    if (tracking_dialog == NULL) {
        tracking_dialog = create_xromm_markerless_tracking_tracking_dialog();
        gtk_spin_button_set_value(
            GTK_SPIN_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_from_spinbutton")),
            position_graph.min_frame);
        gtk_spin_button_set_value(
            GTK_SPIN_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_to_spinbutton")),
            position_graph.max_frame);
    }

    GtkWidget* from_spin_button = lookup_widget(tracking_dialog,"xromm_markerless_tracking_tracking_from_spinbutton");
    GtkWidget* to_spin_button = lookup_widget(tracking_dialog,"xromm_markerless_tracking_tracking_to_spinbutton");

    GtkAdjustment* from_adj = gtk_spin_button_get_adjustment(GTK_SPIN_BUTTON(from_spin_button));
    GtkAdjustment* to_adj = gtk_spin_button_get_adjustment(GTK_SPIN_BUTTON(to_spin_button));

    from_adj->upper = tracker.trial()->num_frames-1;
    gtk_adjustment_changed(from_adj);

    if (from_adj->value > from_adj->upper) {
        from_adj->value = from_adj->upper;
        gtk_adjustment_value_changed(from_adj);
    }

    to_adj->upper = tracker.trial()->num_frames-1;
    gtk_adjustment_changed(to_adj);

    if (to_adj->value > to_adj->upper) {
        to_adj->value = to_adj->upper;
        gtk_adjustment_value_changed(to_adj);
    }

    gint response = gtk_dialog_run(GTK_DIALOG(tracking_dialog));

    if (response == GTK_RESPONSE_OK) {

        int from_frame =
            (int)gtk_spin_button_get_value(GTK_SPIN_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_from_spinbutton")));
        int to_frame =
            (int)gtk_spin_button_get_value(GTK_SPIN_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_to_spinbutton")));

        int num_repititions =
            (int)gtk_spin_button_get_value(GTK_SPIN_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_repeats_spinbutton")));

        bool reverse =
            gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_reverse_check_button")));

        //GtkWidget* pbar =
        //    lookup_widget(tracking_dialog,
        //                  "xromm_markerless_tracking_tracking_progressbar");

        tracker.trial()->volumeTrans = volume_matrix;

        push_state();

        if (tracker.trial()->guess == 0) {
            double xyzypr[6];
            (manip_matrix()*volume_matrix).to_xyzypr(xyzypr);

            tracker.trial()->x_curve.insert(from_frame,xyzypr[0]);
            tracker.trial()->y_curve.insert(from_frame,xyzypr[1]);
            tracker.trial()->z_curve.insert(from_frame,xyzypr[2]);
            tracker.trial()->yaw_curve.insert(from_frame,xyzypr[3]);
            tracker.trial()->pitch_curve.insert(from_frame,xyzypr[4]);
            tracker.trial()->roll_curve.insert(from_frame,xyzypr[5]);
        }

        if (!idle_frame_optimize_id) {
            if (reverse) {
                idle_frame = to_frame;
                idle_from_frame = to_frame;
                idle_to_frame = from_frame;
                d_frame = from_frame > to_frame? 1: -1;
            }
            else {
                idle_frame = from_frame;
                idle_from_frame = from_frame;
                idle_to_frame = to_frame;
                d_frame = from_frame > to_frame? -1: 1;
            }
            idle_num_repeats = num_repititions;
            idle_exit = false;
            idle_frame_optimize_id =
                gtk_idle_add((GtkFunction)idle_frame_optimize, NULL);
        }
    }
}

void
on_xromm_markerless_toolbar_retrack_button_clicked
                                        (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
    if (tracking_dialog == NULL) {
        tracking_dialog = create_xromm_markerless_tracking_tracking_dialog();
        gtk_spin_button_set_value(
            GTK_SPIN_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_from_spinbutton")),
            position_graph.min_frame);
        gtk_spin_button_set_value(
            GTK_SPIN_BUTTON(
                lookup_widget(tracking_dialog,
                "xromm_markerless_tracking_tracking_to_spinbutton")),
            position_graph.max_frame);
    }

    if (idle_from_frame != idle_to_frame) {
        gtk_widget_show(GTK_WIDGET(tracking_dialog));
    }

    if (!idle_frame_optimize_id) {
        idle_frame = idle_from_frame;
        idle_exit = false;
        idle_frame_optimize_id =
            gtk_idle_add((GtkFunction)idle_frame_optimize, NULL);
    }
}


void
on_xromm_markerless_tracking_timeline_prev_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    if (tracker.trial()->frame > 0) {
        gtk_range_set_value(
            GTK_RANGE(lookup_widget(window,
                "xromm_markerless_tracking_timeline")),
            --tracker.trial()->frame);
    }
}

gint play_tag = 0;

void play(gpointer* data)
{
    int next_frame = ((int)(tracker.trial()->frame+1-position_graph.min_frame)%
                      (int)(position_graph.max_frame-position_graph.min_frame))+
                      (int)position_graph.min_frame;
    gtk_range_set_value(
        GTK_RANGE(lookup_widget(window,
            "xromm_markerless_tracking_timeline")),
        next_frame);
}

void
on_xromm_markerless_tracking_timeline_stop_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    if (play_tag) {
        g_source_remove(play_tag);
        play_tag = 0;
    }
}

void
on_xromm_markerless_tracking_timeline_play_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    if (!play_tag) {
        //gtk_range_set_value(
        //    GTK_RANGE(lookup_widget(window,
        //        "xromm_markerless_tracking_timeline")),
        //    position_graph.min_frame);
        play_tag = g_timeout_add(100, (GtkFunction)play, NULL);
    }
}


void
on_xromm_markerless_tracking_timeline_next_button_clicked
                                        (GtkButton       *button,
                                        gpointer         user_data)
{
    if (tracker.trial()->frame < tracker.trial()->num_frames-1) {
        gtk_range_set_value(
            GTK_RANGE(lookup_widget(window,
                "xromm_markerless_tracking_timeline")),
            ++tracker.trial()->frame);
    }
}


void
on_xromm_markerless_tracking_notebook_trial_treeview_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data)
{
}


void
on_xromm_markerless_tracking_window_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data)
{
    window = widget;
}


void
on_xromm_markerless_tracking_current_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    if (gtk_toggle_button_get_active(togglebutton)) {
        tracker.trial()->guess = 0;
    }
}


void
on_xromm_markerless_tracking_tracking_previous_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    if (gtk_toggle_button_get_active(togglebutton)) {
        tracker.trial()->guess = 1;
    }
}


void
on_xromm_markerless_tracking_tracking_extrap_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    if (gtk_toggle_button_get_active(togglebutton)) {
        tracker.trial()->guess = 2;
    }
}


void
on_xromm_markerless_tracking_graph_toolbar_realize
                                        (GtkWidget       *widget,
                                        gpointer         user_data)
{
    GtkWidget* x_spin_button = lookup_widget(widget,
        "xromm_markerless_tracking_graph_toolbar_x_spin_button");
    GtkWidget* y_spin_button = lookup_widget(widget,
        "xromm_markerless_tracking_graph_toolbar_y_spin_button");
    GtkWidget* z_spin_button = lookup_widget(widget,
        "xromm_markerless_tracking_graph_toolbar_z_spin_button");
    GtkWidget* yaw_spin_button = lookup_widget(widget,
        "xromm_markerless_tracking_graph_toolbar_yaw_spin_button");
    GtkWidget* pitch_spin_button = lookup_widget(widget,
        "xromm_markerless_tracking_graph_toolbar_pitch_spin_button");
    GtkWidget* roll_spin_button = lookup_widget(widget,
        "xromm_markerless_tracking_graph_toolbar_roll_spin_button");

    GList* focusable_widgets = NULL;
    focusable_widgets = g_list_append(focusable_widgets, x_spin_button);
    focusable_widgets = g_list_append(focusable_widgets, y_spin_button);
    focusable_widgets = g_list_append(focusable_widgets, z_spin_button);
    focusable_widgets = g_list_append(focusable_widgets, yaw_spin_button);
    focusable_widgets = g_list_append(focusable_widgets, pitch_spin_button);
    focusable_widgets = g_list_append(focusable_widgets, roll_spin_button);

    gtk_container_set_focus_chain(GTK_CONTAINER(widget),
                                  focusable_widgets);
}


void
on_cancelbutton1_clicked               (GtkButton       *button,
                                        gpointer         user_data)
{
    idle_exit = true;
    gtk_widget_hide(tracking_dialog);
}


void
on_new_trial1_activate                  (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    save_trial_prompt();
    save_tracking_prompt();

    try {
        NewTrialDialog dialog;
        if (dialog.run()) {
            tracker.load(dialog.trial);

            trial_filename = "";
            is_trial_saved = false;
            is_tracking_saved = true;

            manip.set_transform(Mat4d());
            volume_matrix = CoordFrame();

            update();
            fill_notebook();
            redraw_drawingarea(drawingarea1);
            redraw_drawingarea(drawingarea2);
        }
    }
    catch (runtime_error& e) {
        cerr << e.what() << endl;
    }
}


void
on_open_trial1_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    save_trial_prompt();
    save_tracking_prompt();

    string filename = get_filename();
    if (filename.compare("") != 0) {
        load_trial(filename);
    }
}


void
on_save_trial1_activate                 (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    if (trial_filename.compare("") == 0) {
        on_save_as_trial1_activate(menuitem,user_data);
    }
    else {
        try {
            tracker.trial()->save(trial_filename);
            is_trial_saved = true;
        }
        catch (exception& e) {
            cerr << e.what() << endl;
        }
    }
}


void
on_save_as_trial1_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    string filename = get_filename(true);
    if (filename.compare("") != 0) {
        try {
            tracker.trial()->save(filename);
            trial_filename = filename;
            is_tracking_saved = true;
        }
        catch (exception& e) {
            cerr << e.what() << endl;
        }
    }
}


void
on_save_tracking1_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    string filename = get_filename(true);
    if (filename.compare("") != 0) {
        save_tracking_results(filename);
    }
}


void
on_load_tracking1_activate              (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    string filename = get_filename();
    if (filename.compare("") != 0) {
        load_tracking_results(filename);
    }
}


void
on_xromm_markerless_tracking_toolbar_openbutton_clicked
                                        (GtkToolButton   *toolbutton,
                                        gpointer         user_data)
{
    save_trial_prompt();
    save_tracking_prompt();

    string filename = get_filename();
    if (filename.compare("") != 0) {
        load_trial(filename);
    }
}

gboolean
on_xromm_markerless_tracking_window_key_press_event
                                        (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data)
{
    switch (event->keyval) {
        case GDK_w: {
                GtkWidget* toggle_tool_button =
                    lookup_widget(widget, "xromm_markerless_tracking_"
                                          "toolbar_translate_radiobutton");
                gtk_toggle_tool_button_set_active(
                        GTK_TOGGLE_TOOL_BUTTON(toggle_tool_button),TRUE);
            }
            return TRUE;
        case GDK_e: {
                GtkWidget* toggle_tool_button =
                    lookup_widget(widget, "xromm_markerless_tracking_"
                                          "toolbar_rotate_radiobutton");
                gtk_toggle_tool_button_set_active(
                        GTK_TOGGLE_TOOL_BUTTON(toggle_tool_button),TRUE);
            }
            return TRUE;
        case GDK_d: {
                GtkWidget* toggle_tool_button =
                    lookup_widget(widget, "toggletoolbutton1");
                gtk_toggle_tool_button_set_active(
                        GTK_TOGGLE_TOOL_BUTTON(toggle_tool_button),TRUE);
            }
            return TRUE;
        case GDK_h: {
            toggle_drrs();
            redraw_drawingarea(drawingarea1);
            redraw_drawingarea(drawingarea2);
            return TRUE;
        }
        case GDK_t: {
            on_xromm_markerless_tracking_toolbar_track_button_clicked(0,0);
            return TRUE;
		}
		case GDK_r: {
			on_xromm_markerless_toolbar_retrack_button_clicked(0,0);
            return TRUE;
        }
        case GDK_plus:
        case GDK_equal:
            pivot_size *= 1.1f;
            redraw_drawingarea(drawingarea1);
            redraw_drawingarea(drawingarea2);
            return TRUE;
        case GDK_minus:
            pivot_size *= 0.9f;
            redraw_drawingarea(drawingarea1);
            redraw_drawingarea(drawingarea2);
            return TRUE;
        default:
            break;
    }

    return FALSE;
}


gboolean
on_xromm_markerless_tracking_window_key_release_event
                                        (GtkWidget       *widget,
                                        GdkEventKey     *event,
                                        gpointer         user_data)
{

    switch (event->keyval) {
        case GDK_d: {
                GtkWidget* toggle_tool_button =
                    lookup_widget(widget, "toggletoolbutton1");
                gtk_toggle_tool_button_set_active(
                        GTK_TOGGLE_TOOL_BUTTON(toggle_tool_button),FALSE);
            }
            return TRUE;
    }
    return FALSE;
}


void
on_quit1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    save_trial_prompt();
    save_tracking_prompt();
    gtk_main_quit();
}


void
on_tracking1_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{

}


void
on_import_tracking_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    string filename = get_filename(true);
    if (filename.compare("") != 0) {
        load_tracking_results(filename);
    }
}


void
on_export_tracking_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    string filename = get_filename();
    if (filename.compare("") != 0) {
        save_tracking_results(filename);
    }
}

void
on_insert_key1_activate                (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    push_state();
    selected_nodes.clear();

    double xyzypr[6];
    (manip_matrix()*volume_matrix).to_xyzypr(xyzypr);

    tracker.trial()->x_curve.insert(tracker.trial()->frame,xyzypr[0]);
    tracker.trial()->y_curve.insert(tracker.trial()->frame,xyzypr[1]);
    tracker.trial()->z_curve.insert(tracker.trial()->frame,xyzypr[2]);
    tracker.trial()->yaw_curve.insert(tracker.trial()->frame,xyzypr[3]);
    tracker.trial()->pitch_curve.insert(tracker.trial()->frame,xyzypr[4]);
    tracker.trial()->roll_curve.insert(tracker.trial()->frame,xyzypr[5]);

    update_graph_min_max(&position_graph);

    redraw_drawingarea(graph_drawingarea);
}

void
on_break_tangents1_activate            (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    push_state();

    for (unsigned i = 0; i < selected_nodes.size(); i++) {
        KeyCurve& curve = *selected_nodes[i].first.first;
        KeyCurve::iterator it = selected_nodes[i].first.second;
        if (!position_graph.frame_locks.at((int)curve.time(it))) {
            curve.set_bind_tangents(it,false);
        }
    }
}

void
on_lock_frames1_activate               (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    push_state();

    for (unsigned i = 0; i < selected_nodes.size(); i++) {

        int time = (int)selected_nodes[i].first.first->time(
                   selected_nodes[i].first.second);

        // Force the addition of keys for all curves in order to truely freeze
        // the frame

        tracker.trial()->x_curve.insert(time);
        tracker.trial()->y_curve.insert(time);
        tracker.trial()->z_curve.insert(time);
        tracker.trial()->yaw_curve.insert(time);
        tracker.trial()->pitch_curve.insert(time);
        tracker.trial()->roll_curve.insert(time);

        position_graph.frame_locks.at(time) = true;
    }

    selected_nodes.clear();

    frame_changed();
    redraw_drawingarea(graph_drawingarea);
}


void
on_unlock_frames1_activate             (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    for (unsigned i = 0; i < selected_nodes.size(); i++) {
        position_graph.frame_locks.at((int)selected_nodes[i].first.first->time(
                                   selected_nodes[i].first.second)) = false;
    }

    frame_changed();
    redraw_drawingarea(graph_drawingarea);
}


void
on_copy_frames1_activate               (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    if (!selected_nodes.empty()) {
        copied_nodes.clear();
        for (unsigned i = 0; i < selected_nodes.size(); i++) {
            if (selected_nodes[i].second == NODE) {
                copied_nodes.push_back(selected_nodes[i].first);
            }
        }
    }

    redraw_drawingarea(graph_drawingarea);
}


gboolean
on_xromm_markerless_tracking_graph_drawingarea_button_press_event
                                        (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
    // Only respond to left button click
    if (event->button == 1) {

        double x, y;
        mouse_to_graph(event->x,event->y,x,y);
        marquee[0] = x;
        marquee[1] = y;
        marquee[2] = x;
        marquee[3] = y;

        // If control is pressed then we are modifying nodes or tangents
        if (event->state & GDK_CONTROL_MASK) {
            modify_nodes = true;
            push_state();
        }

        // Otherwise we are creating a selection marquee
        else {
            draw_marquee = true;
        }
    }

    redraw_drawingarea(graph_drawingarea);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_graph_drawingarea_button_release_event
                                        (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data)
{
    // If there are selected nodes and

    if (event->button == 1) {

        if (modify_nodes) {
            modify_nodes = false;
        }
        else if (draw_marquee) {

            float min_x = marquee[0] < marquee[2]? marquee[0]: marquee[2];
            float max_x = marquee[0] < marquee[2]? marquee[2]: marquee[0];
            float min_y = marquee[1] < marquee[3]? marquee[1]: marquee[3];
            float max_y = marquee[1] < marquee[3]? marquee[3]: marquee[1];

            double frame_offset = 48.0*(position_graph.max_frame-
                                        position_graph.min_frame)/
                                        graph_view.viewport_width;
            double min_frame = position_graph.min_frame-frame_offset;
            double max_frame = position_graph.max_frame-1.0;
            double value_offset = 24.0*(position_graph.max_value-
                                        position_graph.min_value)/
                                        graph_view.viewport_height;
            double value_offset_top = 8.0*(position_graph.max_value-
                                           position_graph.min_value)/
                                           graph_view.viewport_height;
            double min_value = position_graph.min_value-value_offset;
            double max_value = position_graph.max_value+value_offset_top;

            float a = (max_frame+1-min_frame)/(max_value-min_value)*
                       graph_view.viewport_height/graph_view.viewport_width;
            float tan_scale = 40.0f*(max_frame+1-min_frame)/graph_view.viewport_width;

            vector<pair<pair<KeyCurve*,KeyCurve::iterator>,Selection_type> > new_nodes;

            for (unsigned i = 0; i < selected_nodes.size(); i++) {
                KeyCurve& curve = *selected_nodes[i].first.first;
                KeyCurve::iterator it = selected_nodes[i].first.second;

                float s_in = tan_scale/sqrt(1.0f+a*a*curve.in_tangent(it)*curve.in_tangent(it));
                float s_out = tan_scale/sqrt(1.0f+a*a*curve.out_tangent(it)*curve.out_tangent(it));

                bool in_selected = curve.time(it)-s_in > min_x &&
                                   curve.time(it)-s_in < max_x &&
                                   curve.value(it)-s_in*curve.in_tangent(it) > min_y &&
                                   curve.value(it)-s_in*curve.in_tangent(it) < max_y;
                bool node_selected = curve.time(it) > min_x &&
                                     curve.time(it) < max_x &&
                                     curve.value(it) > min_y &&
                                     curve.value(it) < max_y;
                bool out_selected = curve.time(it)+s_out > min_x &&
                                    curve.time(it)+s_out < max_x &&
                                    curve.value(it)+s_out*curve.out_tangent(it) > min_y &&
                                    curve.value(it)+s_out*curve.out_tangent(it) < max_y;

                if (in_selected && !node_selected && !out_selected) {
                    new_nodes.push_back(make_pair(make_pair(&curve,it),IN_TANGENT));
                }
                else if (!in_selected && !node_selected && out_selected) {
                    new_nodes.push_back(make_pair(make_pair(&curve,it),OUT_TANGENT));
                }
            }

            //double v = 3.0;
            //double x_sense = (max_frame+1-min_frame)/graph_view.viewport_width;
            //double y_sense = (max_value-min_value)/graph_view.viewport_height;

            if (position_graph.show_x) {
                KeyCurve::iterator it = tracker.trial()->x_curve.begin();
                while (it != tracker.trial()->x_curve.end()) {
                    if (tracker.trial()->x_curve.time(it) > min_x &&
                        tracker.trial()->x_curve.time(it) < max_x &&
                        tracker.trial()->x_curve.value(it) > min_y &&
                        tracker.trial()->x_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&tracker.trial()->x_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (position_graph.show_y) {
                KeyCurve::iterator it = tracker.trial()->y_curve.begin();
                while (it != tracker.trial()->y_curve.end()) {
                    if (tracker.trial()->y_curve.time(it) > min_x &&
                        tracker.trial()->y_curve.time(it) < max_x &&
                        tracker.trial()->y_curve.value(it) > min_y &&
                        tracker.trial()->y_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&tracker.trial()->y_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (position_graph.show_z) {
                KeyCurve::iterator it = tracker.trial()->z_curve.begin();
                while (it != tracker.trial()->z_curve.end()) {
                    if (tracker.trial()->z_curve.time(it) > min_x &&
                        tracker.trial()->z_curve.time(it) < max_x &&
                        tracker.trial()->z_curve.value(it) > min_y &&
                        tracker.trial()->z_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&tracker.trial()->z_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (position_graph.show_yaw) {
                KeyCurve::iterator it = tracker.trial()->yaw_curve.begin();
                while (it != tracker.trial()->yaw_curve.end()) {
                    if (tracker.trial()->yaw_curve.time(it) > min_x &&
                        tracker.trial()->yaw_curve.time(it) < max_x &&
                        tracker.trial()->yaw_curve.value(it) > min_y &&
                        tracker.trial()->yaw_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&tracker.trial()->yaw_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (position_graph.show_pitch) {
                KeyCurve::iterator it = tracker.trial()->pitch_curve.begin();
                while (it != tracker.trial()->pitch_curve.end()) {
                    if (tracker.trial()->pitch_curve.time(it) > min_x &&
                        tracker.trial()->pitch_curve.time(it) < max_x &&
                        tracker.trial()->pitch_curve.value(it) > min_y &&
                        tracker.trial()->pitch_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&tracker.trial()->pitch_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (position_graph.show_roll) {
                KeyCurve::iterator it = tracker.trial()->roll_curve.begin();
                while (it != tracker.trial()->roll_curve.end()) {
                    if (tracker.trial()->roll_curve.time(it) > min_x &&
                        tracker.trial()->roll_curve.time(it) < max_x &&
                        tracker.trial()->roll_curve.value(it) > min_y &&
                        tracker.trial()->roll_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&tracker.trial()->roll_curve,it),NODE));
                    }
                    ++it;
                }
            }

            selected_nodes = new_nodes;

            draw_marquee = false;
        }
    }

    update_graph_min_max(&position_graph);
    redraw_drawingarea(graph_drawingarea);

    return TRUE;
}


gboolean
on_xromm_markerless_tracking_graph_drawingarea_motion_notify_event
                                        (GtkWidget       *widget,
                                        GdkEventMotion  *event,
                                        gpointer         user_data)
{
    if (event->state & GDK_BUTTON1_MASK) {
        if (modify_nodes) {
            double x, y;
            mouse_to_graph(event->x,event->y,x,y);

            int dx = (int)(x-marquee[2]); // Clamp to integer values
            double dy = y-marquee[3];

            for (unsigned i = 0; i < selected_nodes.size(); i++) {
                KeyCurve& curve = *selected_nodes[i].first.first;
                KeyCurve::iterator it = selected_nodes[i].first.second;
                Selection_type type = selected_nodes[i].second;

                if (position_graph.frame_locks.at((int)curve.time(it))) {
                    continue;
                }

                if (type == NODE) {
                    //node.set_x(node.get_x()+dx); // Prevent x from begin
                    //modified
                    curve.set_value(it,curve.value(it)+dy);
                }
                else if (type == IN_TANGENT) {
                    double in = curve.in_tangent(it)-dy;
                    curve.set_in_tangent(it,in);

                }
                else { // OUT_TANGENT
                    double out = curve.out_tangent(it)+dy;
                    curve.set_out_tangent(it,out);
                }
            }

            marquee[2] = abs(dx) > 0? x: marquee[2];
            marquee[3] = y;

            update_xyzypr_and_coord_frame();
        }
        else if (draw_marquee) {
            double x, y;
            mouse_to_graph(event->x,event->y,x,y);

            marquee[2] = x;
            marquee[3] = y;
        }
    }

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);

    return TRUE;
}


void
on_cut1_activate                       (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    push_state();

    if (!selected_nodes.empty()) {
        copied_nodes.clear();
        for (unsigned i = 0; i < selected_nodes.size(); i++) {
            if (selected_nodes[i].second == NODE) {
                copied_nodes.push_back(selected_nodes[i].first);
                if (!position_graph.frame_locks.at((int)selected_nodes[i].first.first->time(selected_nodes[i].first.second))) {
                    selected_nodes[i].first.first->erase(selected_nodes[i].first.second);
                }
            }
        }
        selected_nodes.clear();
    }

    update_xyzypr_and_coord_frame();
    update_graph_min_max(&position_graph);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);
}


void
on_paste1_activate                     (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    push_state();

    if (!copied_nodes.empty()) {
        float frame_offset = copied_nodes.front().first->time(copied_nodes.front().second);
        for (unsigned i = 0; i < copied_nodes.size(); i++) {
            float frame = tracker.trial()->frame+copied_nodes[i].first->time(copied_nodes[i].second)-frame_offset;
            if (!position_graph.frame_locks.at((int)frame)) {
                copied_nodes[i].first->insert(frame,copied_nodes[i].first->value(copied_nodes[i].second));
            }
        }
        selected_nodes.clear();
    }

    update_xyzypr_and_coord_frame();
    update_graph_min_max(&position_graph);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);
}


void
on_delete1_activate                    (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    if (!selected_nodes.empty()) {
        push_state();

        for (unsigned i = 0; i < selected_nodes.size(); i++) {
            if (selected_nodes[i].second == NODE) {
                if (!position_graph.frame_locks.at((int)selected_nodes[i].first.first->time(selected_nodes[i].first.second))) {
                    selected_nodes[i].first.first->erase(selected_nodes[i].first.second);
                }
            }
        }
        selected_nodes.clear();

        update_xyzypr_and_coord_frame();
        update_graph_min_max(&position_graph);

        redraw_drawingarea(drawingarea1);
        redraw_drawingarea(drawingarea2);
        redraw_drawingarea(graph_drawingarea);
    }
}


void
on_xromm_markerless_tracking_spline_radiobutton_toggled
                                        (GtkToggleButton *togglebutton,
                                        gpointer         user_data)
{
    if (gtk_toggle_button_get_active(togglebutton)) {
        tracker.trial()->guess = 3;
    }
}


void
on_undo1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    undo_state();
}


void
on_redo1_activate                      (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    redo_state();
}


void
on_smooth_tangents1_activate           (GtkMenuItem     *menuitem,
                                        gpointer         user_data)
{
    push_state();

    for (unsigned i = 0; i < selected_nodes.size(); i++) {
        KeyCurve& curve = *selected_nodes[i].first.first;
        KeyCurve::iterator it = selected_nodes[i].first.second;

        if (!position_graph.frame_locks.at((int)curve.time(it))) {
            curve.set_bind_tangents(it,true);
            curve.set_in_tangent_type(it,KeyCurve::SMOOTH);
            curve.set_out_tangent_type(it,KeyCurve::SMOOTH);
        }
    }

    update_xyzypr_and_coord_frame();
    update_graph_min_max(&position_graph);

    redraw_drawingarea(drawingarea1);
    redraw_drawingarea(drawingarea2);
    redraw_drawingarea(graph_drawingarea);
}
