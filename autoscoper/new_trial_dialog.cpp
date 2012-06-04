#include "new_trial_dialog.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <Camera.hpp>
#include <Video.hpp>

using namespace std;

static const char* NEW_TRIAL_DIALOG_UI = "new_trial_dialog.glade";

NewTrialDialog::NewTrialDialog()
{
    GtkBuilder* builder = gtk_builder_new();

	GError* error = NULL;
	if (!gtk_builder_add_from_file(builder,NEW_TRIAL_DIALOG_UI,&error)) {
		string message = string("Failed to load ")+NEW_TRIAL_DIALOG_UI+": "+error->message;
		g_error_free(error);
		throw runtime_error(message.c_str());
	}

    gtk_builder_connect_signals(builder,this);

    dialog = GTK_WIDGET(gtk_builder_get_object(builder,"dialog1"));
    camera1_mayacam_chooser = GTK_WIDGET(gtk_builder_get_object(builder,"camera1_mayacam_chooser"));
    camera1_video_path_chooser = GTK_WIDGET(gtk_builder_get_object(builder,"camera1_video_path_chooser"));
    camera2_mayacam_chooser = GTK_WIDGET(gtk_builder_get_object(builder,"camera2_mayacam_chooser"));
    camera2_video_path_chooser = GTK_WIDGET(gtk_builder_get_object(builder,"camera2_video_path_chooser"));
    volume_filename_chooser = GTK_WIDGET(gtk_builder_get_object(builder,"volume_filename_chooser"));
    volume_scale_x_entry = GTK_WIDGET(gtk_builder_get_object(builder,"volume_scale_x_entry"));
    volume_scale_y_entry = GTK_WIDGET(gtk_builder_get_object(builder,"volume_scale_y_entry"));
    volume_scale_z_entry = GTK_WIDGET(gtk_builder_get_object(builder,"volume_scale_z_entry"));
    volume_flip_x_toggle = GTK_WIDGET(gtk_builder_get_object(builder,"volume_flip_x_toggle"));
    volume_flip_y_toggle = GTK_WIDGET(gtk_builder_get_object(builder,"volume_flip_y_toggle"));
    volume_flip_z_toggle = GTK_WIDGET(gtk_builder_get_object(builder,"volume_flip_z_toggle"));
    volume_units_combobox = GTK_WIDGET(gtk_builder_get_object(builder,"volume_units_combobox"));

    g_object_unref(G_OBJECT(builder));
}

NewTrialDialog::~NewTrialDialog()
{
    gtk_widget_destroy(dialog);
}

bool
NewTrialDialog::run()
{
    gint result = gtk_dialog_run(GTK_DIALOG(dialog));
    if (result != GTK_RESPONSE_OK) {
        return false;
    }

    gchar* mayacam1_temp = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(camera1_mayacam_chooser));
    gchar* mayacam2_temp = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(camera2_mayacam_chooser));
    gchar* video_path1_temp = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(camera1_video_path_chooser));
    gchar* video_path2_temp = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(camera2_video_path_chooser));
    gchar* volume_filename_temp = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(volume_filename_chooser));

    if (!mayacam1_temp) {
        cerr << "Required field 'Camera 1 MayaCam' is empty" << endl;
        return false;
    }
    if (!mayacam2_temp) {
        cerr << "Required field 'Camera 2 MayaCam' is empty" << endl;
        return false;
    }
    if (!volume_filename_temp) {
        cerr << "Required field 'Volume Filename' is empty" << endl;
        return false;
    }

    string mayacam1(mayacam1_temp);
    string mayacam2(mayacam2_temp);
    string video_path1(video_path1_temp);
    string video_path2(video_path2_temp);
    string volume_filename(volume_filename_temp);

    g_free(mayacam1_temp);
    g_free(mayacam2_temp);
    g_free(video_path1_temp);
    g_free(video_path2_temp);
    g_free(volume_filename_temp);

    string volume_scale_x_str = gtk_entry_get_text(GTK_ENTRY(volume_scale_x_entry));
    string volume_scale_y_str = gtk_entry_get_text(GTK_ENTRY(volume_scale_y_entry));
    string volume_scale_z_str = gtk_entry_get_text(GTK_ENTRY(volume_scale_z_entry));

    stringstream scale_stream;
    scale_stream << volume_scale_x_str << " " << volume_scale_y_str << " " << volume_scale_z_str;

    float volume_scale_x, volume_scale_y, volume_scale_z;
    if (!(scale_stream >> volume_scale_x >> volume_scale_y >> volume_scale_z)) {
        cerr << "Invalid input in 'Volume Scale' fields" << endl;
        return false;
    }

	gint units = gtk_combo_box_get_active(GTK_COMBO_BOX(volume_units_combobox));
	switch (units) {
		case 0: { // micrometers->millimeters
			volume_scale_x /= 1000;
			volume_scale_y /= 1000;
			volume_scale_z /= 1000;
			break;
		}
		default:
		case 1: { // milimeters->millimeters
			break;
		}
		case 2: { // centemeters->millimeters
			volume_scale_x *= 10;
			volume_scale_y *= 10;
			volume_scale_z *= 10;
			break;
		}
	}

    gboolean volume_flip_x = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(volume_flip_x_toggle));
    gboolean volume_flip_y = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(volume_flip_y_toggle));
    gboolean volume_flip_z = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(volume_flip_z_toggle));

    try {

        trial = xromm::Trial();
        trial.cameras.push_back(xromm::Camera(mayacam1));
        trial.cameras.push_back(xromm::Camera(mayacam2));

        trial.videos.push_back(xromm::Video(video_path1));
        trial.videos.push_back(xromm::Video(video_path2));
        trial.num_frames = max(trial.videos.at(0).num_frames(),
                               trial.videos.at(1).num_frames());

        xromm::Volume volume(volume_filename);

        volume.scaleX(volume_scale_x);
        volume.scaleY(volume_scale_y);
        volume.scaleZ(volume_scale_z);

        volume.flipX(volume_flip_x);
        volume.flipY(volume_flip_y);
        volume.flipZ(volume_flip_z);

        trial.volumes.push_back(volume);

        trial.offsets[0] = 0.1;
        trial.offsets[1] = 0.1;
        trial.offsets[2] = 0.1;
        trial.offsets[3] = 0.1;
        trial.offsets[4] = 0.1;
        trial.offsets[5] = 0.1;

        trial.render_width = 512;
        trial.render_height = 512;

        return true;
    }
    catch (exception& e) {
        cerr << e.what() << endl;
        return false;
    }
}
