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

// xromm_gtk_tree_view.cpp

#include "xromm_gtk_tree_view.hpp"

#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>

#include <View.hpp>
#include <SobelFilter.hpp>
#include <ContrastFilter.hpp>
#include <RayCaster.hpp>
#include <RadRenderer.hpp>

#include "callbacks.hpp"
#include "interface.hpp"
#include "support.hpp"

using namespace std;
using namespace xromm;
using namespace cuda;

enum
{
    XROMM_CUDA_VIEW,
    XROMM_CUDA_DRR_RENDERER,
    XROMM_CUDA_RAD_RENDERER,
    XROMM_CUDA_FILTER,
};

enum
{
    ENABLED_COLUMN = 0,
    NAME_COLUMN,
    POINTER_COLUMN,
    TYPE_COLUMN,
    VISIBLE_COLUMN,
    VECTOR_COLUMN,
    NUM_OF_COLUMNS
};

GtkTreeStore* tree_store = 0;

GtkTreeStore* xromm_gtk_tree_store_new_from_views(const vector<View*>& views)
{
    // Create a tree store
    tree_store = gtk_tree_store_new(NUM_OF_COLUMNS,
                                      G_TYPE_BOOLEAN,
                                      G_TYPE_STRING,
                                      G_TYPE_POINTER,
                                      G_TYPE_INT,
                                      G_TYPE_BOOLEAN,
                                      G_TYPE_POINTER);

    int view_idx = 1;
    vector<View*>::const_iterator view_iter;
    for (view_iter = views.begin(); view_iter != views.end(); ++view_iter) {

        stringstream view_name_stream;
        view_name_stream << "View" << view_idx++;

        GtkTreeIter view_tree_iter;
        gtk_tree_store_append(tree_store, &view_tree_iter, NULL);
        gtk_tree_store_set(tree_store,
                           &view_tree_iter,
                           ENABLED_COLUMN, TRUE,
                           NAME_COLUMN, view_name_stream.str().c_str(),
                           POINTER_COLUMN, *view_iter,
                           TYPE_COLUMN, XROMM_CUDA_VIEW,
                           VISIBLE_COLUMN, FALSE,
                           -1);

        GtkTreeIter drr_tree_iter;
        gtk_tree_store_append(tree_store, &drr_tree_iter, &view_tree_iter);
        gtk_tree_store_set(tree_store,
                           &drr_tree_iter,
                           ENABLED_COLUMN, TRUE,
                           NAME_COLUMN, (*view_iter)->drrRenderer()->getName().c_str(),
                           POINTER_COLUMN, (*view_iter)->drrRenderer(),
                           TYPE_COLUMN, XROMM_CUDA_DRR_RENDERER,
                           VECTOR_COLUMN, &(*view_iter)->drrFilters(),
                           VISIBLE_COLUMN, TRUE,
                           -1);

        vector<Filter*>::const_iterator filter_iter;
        for (filter_iter = (*view_iter)->drrFilters().begin();
             filter_iter != (*view_iter)->drrFilters().end();
             ++filter_iter) {

            GtkTreeIter filter_tree_iter;
            gtk_tree_store_append(tree_store,
                                  &filter_tree_iter,
                                  &drr_tree_iter);
            gtk_tree_store_set(tree_store,
                               &filter_tree_iter,
                               ENABLED_COLUMN, TRUE,
                               NAME_COLUMN, (*filter_iter)->name().c_str(),
                               POINTER_COLUMN, *filter_iter,
                               TYPE_COLUMN, XROMM_CUDA_FILTER,
                               VISIBLE_COLUMN, TRUE,
                               VECTOR_COLUMN, &(*view_iter)->drrFilters(),
                               -1);
        }

        GtkTreeIter rad_tree_iter;
        gtk_tree_store_append(tree_store, &rad_tree_iter, &view_tree_iter);
        gtk_tree_store_set(tree_store,
                           &rad_tree_iter,
                           ENABLED_COLUMN, TRUE,
                           NAME_COLUMN, (*view_iter)->radRenderer()->getName().c_str(),
                           POINTER_COLUMN, (*view_iter)->radRenderer(),
                           TYPE_COLUMN, XROMM_CUDA_RAD_RENDERER,
                           VISIBLE_COLUMN, TRUE,
                           VECTOR_COLUMN, &(*view_iter)->radFilters(),
                           -1);

        for (filter_iter = (*view_iter)->radFilters().begin();
             filter_iter != (*view_iter)->radFilters().end();
             ++filter_iter) {

            GtkTreeIter filter_tree_iter;
            gtk_tree_store_append(tree_store,
                                  &filter_tree_iter,
                                  &rad_tree_iter);
            gtk_tree_store_set(tree_store,
                               &filter_tree_iter,
                               ENABLED_COLUMN, TRUE,
                               NAME_COLUMN, (*filter_iter)->name().c_str(),
                               POINTER_COLUMN, *filter_iter,
                               TYPE_COLUMN, XROMM_CUDA_FILTER,
                               VISIBLE_COLUMN, TRUE,
                               VECTOR_COLUMN, &(*view_iter)->radFilters(),
                               -1);
        }
    }

    return tree_store;
}

void
modify_filter_list(GtkWidget* menu_item, gpointer data)
{
}

void
properties_activate_drr_renderer(GtkWidget* menu_item, gpointer data)
{
    RayCaster* rayCaster = (RayCaster*)data;
    stringstream title_stream;
    title_stream << rayCaster->getName() << " Properties";

    GtkWidget* properties_dialog =
        create_xromm_drr_renderer_properties_dialog();

    gtk_window_set_title(GTK_WINDOW(properties_dialog),title_stream.str().c_str());

    GtkWidget* sample_scale = lookup_widget(properties_dialog,
                                            "xromm_drr_renderer_properties_"
                                            "dialog_sample_distance_scale");

    GtkWidget* intensity_scale = lookup_widget(properties_dialog,
                                               "xromm_drr_renderer_"
                                               "properties_dialog_intensity_"
                                               "scale");
    GtkWidget* cutoff_scale = lookup_widget(properties_dialog,
                                            "xromm_drr_renderer_properties_"
                                            "dialog_cutoff_scale");

    // Set the intensity and cutoff values.
    gtk_range_set_value(GTK_RANGE(sample_scale),
                        (log(rayCaster->getSampleDistance())+5.0f)/7.0f);
    gtk_range_set_value(GTK_RANGE(intensity_scale),
                        (log(rayCaster->getRayIntensity())+5.0f)/15.0f);
    //gtk_range_set_range(GTK_RANGE(cutoff_scale),
    //rayCaster->getMinCutoff(),
    //rayCaster->getMaxCutoff());
    gtk_range_set_value(GTK_RANGE(cutoff_scale),
          (rayCaster->getCutoff()-rayCaster->getMinCutoff())/
          (rayCaster->getMaxCutoff()-rayCaster->getMinCutoff()));

    // Connect the signals.
    g_signal_connect(sample_scale,
                     "value-changed",
                     G_CALLBACK(on_xromm_drr_renderer_properties_dialog_sample_distance_scale_value_changed),
                     data);

    g_signal_connect(intensity_scale,
                     "value-changed",
                     G_CALLBACK(on_xromm_drr_renderer_properties_dialog_intensity_scale_value_changed),
                     data);
    g_signal_connect(cutoff_scale,
                     "value-changed",
                     G_CALLBACK(on_xromm_drr_renderer_properties_dialog_cutoff_scale_value_changed),
                     data);

    gtk_widget_show(properties_dialog);
}

void
properties_activate_filter(GtkWidget* menu_item, gpointer data)
{
    Filter* filter = (Filter*)data;
    stringstream title_stream;
    title_stream << filter->name() << " Properties";

    switch (filter->type()) {
        case Filter::XROMM_CUDA_SOBEL_FILTER: {
            SobelFilter* sobel_filter = (SobelFilter*)filter;

            GtkWidget* properties_dialog =
                create_xromm_sobel_properties_dialog();

            gtk_window_set_title(GTK_WINDOW(properties_dialog),title_stream.str().c_str());

            GtkWidget* scale_scale = lookup_widget(properties_dialog,
                                                   "xromm_sobel_properties_"
                                                   "dialog_scale_scale");

            GtkWidget* blend_scale = lookup_widget(properties_dialog,
                                                   "xromm_sobel_properties_"
                                                   "dialog_blend_scale");

            // Set the intensity and cutoff values.
            gtk_range_set_value(GTK_RANGE(scale_scale),
                                sobel_filter->getScale());
            gtk_range_set_value(GTK_RANGE(blend_scale),
                                sobel_filter->getBlend());

            // Connect the signals.
            g_signal_connect(scale_scale,
                             "value-changed",
                             G_CALLBACK(on_xromm_sobel_properties_dialog_scale_scale_value_changed),
                             data);

            g_signal_connect(blend_scale,
                             "value-changed",
                             G_CALLBACK(on_xromm_sobel_properties_dialog_blend_scale_value_changed),
                             data);

            // Display the dialog
            gtk_widget_show(properties_dialog);

            break;
        }
        case Filter::XROMM_CUDA_CONTRAST_FILTER: {
            ContrastFilter* contrast_filter = (ContrastFilter*)filter;

            GtkWidget* properties_dialog =
                create_xromm_contrast_properties_dialog();

            gtk_window_set_title(GTK_WINDOW(properties_dialog),title_stream.str().c_str());

            GtkWidget* alpha_scale = lookup_widget(properties_dialog,
                                                   "xromm_contrast_properties_"
                                                   "dialog_alpha_scale");

            GtkWidget* beta_scale = lookup_widget(properties_dialog,
                                                  "xromm_contrast_properties_"
                                                  "dialog_beta_scale");

            // Set the intensity and cutoff values.
            gtk_range_set_value(GTK_RANGE(alpha_scale),
                                contrast_filter->alpha());
            gtk_range_set_value(GTK_RANGE(beta_scale),
                                contrast_filter->beta());

            // Connect the signals.
            g_signal_connect(alpha_scale,
                             "value-changed",
                             G_CALLBACK(on_xromm_contrast_properties_dialog_alpha_scale_value_changed),
                             data);

            g_signal_connect(beta_scale,
                             "value-changed",
                             G_CALLBACK(on_xromm_contrast_properties_dialog_beta_scale_value_changed),
                             data);

            // Display the dialog
            gtk_widget_show(properties_dialog);

            break;
        }
        default: return;
    }
}

gboolean
xromm_gtk_tree_view_on_button_press(GtkWidget* tree_view,
                                    GdkEventButton* event,
                                    gpointer data)
{
    // Get the path to the element in the tree view that was clicked
    GtkTreePath* tree_path;
    gtk_tree_view_get_path_at_pos(GTK_TREE_VIEW(tree_view),
                                  (gint)event->x,
                                  (gint)event->y,
                                  &tree_path,
                                  NULL, NULL, NULL);
    if (!tree_path) {
        return FALSE;
    }

    // Check for single click with right mouse button
    if (event->type == GDK_BUTTON_PRESS && event->button == 3) {

        // Highlight the currently selected item
        GtkTreeSelection* tree_selection =
            gtk_tree_view_get_selection(GTK_TREE_VIEW(tree_view));
        gtk_tree_selection_select_path(tree_selection, tree_path);

        // Get a reference to the iter at this path
        GtkTreeModel* tree_model =
            gtk_tree_view_get_model(GTK_TREE_VIEW(tree_view));
        GtkTreeIter tree_iter;
        gtk_tree_model_get_iter(tree_model, &tree_iter, tree_path);

        // Get the data associated with this iter
        gpointer iter_pointer;
        gint iter_type;
        gpointer iter_vector;
        gtk_tree_model_get(tree_model, &tree_iter,
                           POINTER_COLUMN, &iter_pointer,
                           TYPE_COLUMN, &iter_type,
                           VECTOR_COLUMN, &iter_vector,
                           -1);

        gtk_tree_path_free(tree_path);

        // Create the popup menu.
        GtkWidget* menu = gtk_menu_new();

        switch (iter_type) {
            case XROMM_CUDA_VIEW: {
                GtkWidget* import_menu_item =
                    gtk_menu_item_new_with_label("Import");
                GtkWidget* export_menu_item =
                    gtk_menu_item_new_with_label("Export");

                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      import_menu_item);
                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      export_menu_item);

                g_signal_connect(import_menu_item,
                                 "activate",
                                 G_CALLBACK(on_import_view_activate),
                                 iter_pointer);
                g_signal_connect(export_menu_item,
                                 "activate",
                                 G_CALLBACK(on_export_view_activate),
                                 iter_pointer);
                break;
            }
            case XROMM_CUDA_DRR_RENDERER: {


                GtkWidget* new_filter_menu_item =
                    gtk_menu_item_new_with_label("New Filter");
                GtkWidget* properties_menu_item =
                    gtk_menu_item_new_with_label("Properties");

                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      new_filter_menu_item);
                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      gtk_separator_menu_item_new());
                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      properties_menu_item);

                GtkWidget* new_filter_menu = gtk_menu_new();
                GtkWidget* sobel_menu_item =
                    gtk_menu_item_new_with_label("Sobel");
                GtkWidget* contrast_menu_item =
                    gtk_menu_item_new_with_label("Contrast");

                gtk_menu_shell_append(GTK_MENU_SHELL(new_filter_menu),
                                      sobel_menu_item);
                gtk_menu_shell_append(GTK_MENU_SHELL(new_filter_menu),
                                      contrast_menu_item);

                gtk_menu_item_set_submenu(GTK_MENU_ITEM(new_filter_menu_item),
                                          new_filter_menu);

                g_signal_connect(sobel_menu_item,
                                 "activate",
                                 G_CALLBACK(on_new_filter_activate),
                                 iter_vector);
                g_signal_connect(contrast_menu_item,
                                 "activate",
                                 G_CALLBACK(on_new_filter_activate),
                                 iter_vector);

                g_signal_connect(properties_menu_item,
                                 "activate",
                                 G_CALLBACK(properties_activate_drr_renderer),
                                 iter_pointer);
                break;
            }
            case XROMM_CUDA_RAD_RENDERER: {
                GtkWidget* new_filter_menu_item =
                    gtk_menu_item_new_with_label("New Filter");

                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      new_filter_menu_item);

                GtkWidget* new_filter_menu = gtk_menu_new();
                GtkWidget* sobel_menu_item =
                    gtk_menu_item_new_with_label("Sobel");
                GtkWidget* contrast_menu_item =
                    gtk_menu_item_new_with_label("Contrast");

                gtk_menu_shell_append(GTK_MENU_SHELL(new_filter_menu),
                                      sobel_menu_item);
                gtk_menu_shell_append(GTK_MENU_SHELL(new_filter_menu),
                                      contrast_menu_item);

                gtk_menu_item_set_submenu(GTK_MENU_ITEM(new_filter_menu_item),
                                          new_filter_menu);

                g_signal_connect(sobel_menu_item,
                                 "activate",
                                 G_CALLBACK(on_new_filter_activate),
                                 iter_vector);
                g_signal_connect(contrast_menu_item,
                                 "activate",
                                 G_CALLBACK(on_new_filter_activate),
                                 iter_vector);

                break;
            }
            case XROMM_CUDA_FILTER: {

                //GtkWidget* new_filter_before_menu_item =
                //    gtk_menu_item_new_with_label("Insert Before");
                //GtkWidget* new_filter_after_menu_item =
                //    gtk_menu_item_new_with_label("Insert After");
                GtkWidget* remove_menu_item =
                    gtk_menu_item_new_with_label("Remove");
                GtkWidget* properties_menu_item =
                    gtk_menu_item_new_with_label("Properties");

                //gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                //                      new_filter_before_menu_item);
                //gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                //                      new_filter_after_menu_item);
                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      remove_menu_item);
                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      gtk_separator_menu_item_new());
                gtk_menu_shell_append(GTK_MENU_SHELL(menu),
                                      properties_menu_item);

                Args* args = (Args*)g_malloc(sizeof(Args));
                args->filters = (vector<Filter*>*)iter_vector;
                args->filter = (Filter*)iter_pointer;

                g_signal_connect(remove_menu_item,
                                 "activate",
                                 G_CALLBACK(on_remove_filter_activate),
                                 (gpointer)args);

                g_signal_connect_swapped(remove_menu_item,
                                         "destroy-event",
                                         G_CALLBACK(g_free),
                                         (gpointer)args);

                g_signal_connect(properties_menu_item,
                                 "activate",
                                 G_CALLBACK(properties_activate_filter),
                                 iter_pointer);


                break;
            }
            default:
                return false;
        }

        // Display the menu
        gtk_widget_show_all(menu);
        gtk_menu_popup(GTK_MENU(menu),
                       NULL, NULL, NULL, NULL,
                       event->button,
                       gdk_event_get_time(NULL));

        return TRUE;
    }

    return FALSE;
}

void
xromm_gtk_tree_view_on_toggle(GtkCellRendererToggle* cell,
                              gchar* path,
                              gpointer data)
{
    // Get the tree model and create the path
    GtkTreeModel* tree_model = GTK_TREE_MODEL(data);
    GtkTreePath* tree_path = gtk_tree_path_new_from_string(path);

    // Get a reference to the iter at this path
    GtkTreeIter tree_iter;
    gtk_tree_model_get_iter(tree_model, &tree_iter, tree_path);

    // Get the data associated with this iter
    gboolean toggled;
    gpointer filter;
    gint type;
    gtk_tree_model_get(tree_model, &tree_iter,
                       ENABLED_COLUMN, &toggled,
                       POINTER_COLUMN, &filter,
                       TYPE_COLUMN, &type, -1);

    // Toggle the butten and update the filter
    toggled ^= 1;

    // Update the toggle button to reflect its new value
    gtk_tree_store_set(GTK_TREE_STORE(tree_model), &tree_iter, 0, toggled, -1);

    if (type == XROMM_CUDA_FILTER) {
        on_toggle_filter(filter,toggled);
    }
    else if (type == XROMM_CUDA_RAD_RENDERER ||
             type == XROMM_CUDA_DRR_RENDERER) {

        gtk_tree_path_up(tree_path);

        GtkTreeIter view_iter;
        gtk_tree_model_get_iter(tree_model, &view_iter, tree_path);

        gpointer view;
        gtk_tree_model_get(tree_model,&view_iter,POINTER_COLUMN,&view,-1);

        on_toggle_renderer(view,type == XROMM_CUDA_DRR_RENDERER? 0:1,toggled);
    }

    gtk_tree_path_free(tree_path);
}

void toggle_drrs()
{
    if (tree_store) {
        xromm_gtk_tree_view_on_toggle(0,(gchar*)"0:0",tree_store);
        xromm_gtk_tree_view_on_toggle(0,(gchar*)"1:0",tree_store);
    }
}

void toggle_rads()
{
    if (tree_store) {
        xromm_gtk_tree_view_on_toggle(0,(gchar*)"0:1",tree_store);
        xromm_gtk_tree_view_on_toggle(0,(gchar*)"1:1",tree_store);
    }
}

GtkWidget*
xromm_gtk_tree_view_new_from_views(const vector<View*>& views)
{
    // Create a tree view with two visible columns
    GtkWidget* tree_view = gtk_tree_view_new ();
    gtk_tree_selection_set_mode(
        gtk_tree_view_get_selection(GTK_TREE_VIEW(tree_view)),
        GTK_SELECTION_SINGLE);
    gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(tree_view), FALSE);

    // Alternating row colors...
    gtk_tree_view_set_rules_hint(GTK_TREE_VIEW(tree_view), TRUE);
    gtk_widget_show(tree_view);

    // Create and append a toggle column
    GtkCellRenderer* toggle_renderer = gtk_cell_renderer_toggle_new();
    GtkTreeViewColumn* toggle_column =
        gtk_tree_view_column_new_with_attributes("Enable",
                                                 toggle_renderer,
                                                 "active", ENABLED_COLUMN,
                                                 "visible", VISIBLE_COLUMN,
                                                 NULL);
    gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), toggle_column);

    // Create and append a name column
    GtkCellRenderer* name_renderer = gtk_cell_renderer_text_new();
    GtkTreeViewColumn* name_column =
        gtk_tree_view_column_new_with_attributes("Layer",
                                                 name_renderer,
                                                 "text", NAME_COLUMN,
                                                 NULL);
    gtk_tree_view_column_set_expand(name_column, TRUE);
    gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), name_column);
    gtk_tree_view_set_expander_column(GTK_TREE_VIEW(tree_view), name_column);

    // Create and add a model to the tree view
    GtkTreeStore* tree_store = xromm_gtk_tree_store_new_from_views(views);
    gtk_tree_view_set_model(GTK_TREE_VIEW(tree_view),
                            GTK_TREE_MODEL(tree_store));

    // Expand all elements in the tree view
    gtk_tree_view_expand_all(GTK_TREE_VIEW(tree_view));

    // Connect signals
    g_signal_connect(tree_view,
                     "button-press-event",
                     G_CALLBACK(xromm_gtk_tree_view_on_button_press),
                     tree_store);

    g_signal_connect(toggle_renderer,
                     "toggled",
                     G_CALLBACK(xromm_gtk_tree_view_on_toggle),
                     tree_store);

    return tree_view;
}

