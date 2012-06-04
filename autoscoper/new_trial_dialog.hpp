#ifndef NEW_TRIAL_DIALOG_HPP
#define NEW_TRIAL_DIALOG_HPP

#include <gtk/gtk.h>

#include <Trial.hpp>

class NewTrialDialog
{
public:    
    
    NewTrialDialog();
    
    ~NewTrialDialog();

    bool run();

    xromm::Trial trial;

private:

    GtkWidget* dialog;
    GtkWidget* camera1_mayacam_chooser;
    GtkWidget* camera1_video_path_chooser;
    GtkWidget* camera2_mayacam_chooser;
    GtkWidget* camera2_video_path_chooser;
    GtkWidget* volume_filename_chooser;
    GtkWidget* volume_scale_x_entry;
    GtkWidget* volume_scale_y_entry;
    GtkWidget* volume_scale_z_entry;
    GtkWidget* volume_flip_x_toggle;
    GtkWidget* volume_flip_y_toggle;
    GtkWidget* volume_flip_z_toggle;
    GtkWidget* volume_units_combobox;
};

#endif // NEW_TRIAL_DIALOG_HPP
