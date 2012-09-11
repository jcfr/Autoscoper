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

// xromm_gtk_draw.cpp

#ifdef _WIN32
#include <windows.h>
#endif

#include "xromm_gtk_draw.hpp"

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GL_GLEXT_PROTOTYPES 1

#ifdef _MACOSX
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

void
draw_gradient(const float* top_color, const float* bot_color)
{
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glBegin(GL_QUADS);
    glColor3fv(bot_color);
    glVertex3i(-1,-1,-1);
    glVertex3i(1,-1,-1);
    glColor3fv(top_color);
    glVertex3i(1,1,-1);
    glVertex3i(-1,1,-1);
    glEnd();

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}

void
draw_xz_grid(int width, int height, float scale)
{
    glPushAttrib(GL_LINE_BIT);

    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f,0.0f,0.0f);
    glVertex3f(-scale*width/2, 0.0f, 0.0f);
    glVertex3f(scale*width/2, 0.0f, 0.0f);
    glColor3f(0.0f,0.0f,1.0f);
    glVertex3f(0.0f, 0.0f, -scale*height/2);
    glVertex3f(0.0f, 0.0f, scale*height/2);
    glEnd();

    glColor3f(0.5f,0.5f,0.5f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int i = 0; i <= width; ++i) {
        glVertex3f(scale*(i-width/2), 0.0f, -scale*height/2);
        glVertex3f(scale*(i-width/2), 0.0f, scale*height/2);
    }
    for (int i = 0; i <= height; ++i) {
        glVertex3f(-scale*width/2, 0.0f, scale*(i-height/2));
        glVertex3f(scale*width/2, 0.0f, scale*(i-height/2));
    }
    glEnd();

    glPopAttrib();
}

void
draw_cylinder(float radius, float height, int slices)
{
    for (int i = 0; i < slices; ++i) {
        float alpha = 2*M_PI*i/slices;
        float beta = 2*M_PI*(i+1)/slices;

        float cos_alpha = cos(alpha);
        float sin_alpha = sin(alpha);

        float cos_beta = cos(beta);
        float sin_beta = sin(beta);

        glBegin(GL_TRIANGLES);
        glNormal3f(0,-1,0);
        glVertex3f(radius*cos_alpha,-height/2,radius*sin_alpha);
        glVertex3f(radius*cos_beta,-height/2,radius*sin_beta);
        glVertex3f(0,-height/2,0);
        glEnd();

        glBegin(GL_QUADS);
        glNormal3f(cos_alpha,0,sin_alpha);
        glVertex3f(radius*cos_alpha,-height/2,radius*sin_alpha);
        glVertex3f(radius*cos_alpha,height/2,radius*sin_alpha);
        glVertex3f(radius*cos_beta,height/2,radius*sin_beta);
        glVertex3f(radius*cos_beta,-height/2,radius*sin_beta);
        glEnd();

        glBegin(GL_TRIANGLES);
        glNormal3f(0,1,0);
        glVertex3f(radius*cos_alpha,height/2,radius*sin_alpha);
        glVertex3f(0,height/2,0);
        glVertex3f(radius*cos_beta,height/2,radius*sin_beta);
        glEnd();
    }
}

void
draw_camera()
{
    float length = 1.0f;
    float width = 0.3f;
    float height = 0.5f;

    glBegin(GL_QUADS);

    glNormal3f(0.0f,1.0f,0.0f);
    glVertex3f(-width,height,-length);
    glVertex3f(-width,height,length);
    glVertex3f(width,height,length);
    glVertex3f(width,height,-length);

    glNormal3f(1.0f,0.0f,0.0f);
    glVertex3f(width,-height,-length);
    glVertex3f(width,height,-length);
    glVertex3f(width,height,length);
    glVertex3f(width,-height,length);

    glNormal3f(0.0f,-1.0f,0.0f);
    glVertex3f(-width,-height,-length);
    glVertex3f(width,-height,-length);
    glVertex3f(width,-height,length);
    glVertex3f(-width,-height,length);

    glNormal3f(-1.0f,0.0f,0.0f);
    glVertex3f(-width,-height,-length);
    glVertex3f(-width,-height,length);
    glVertex3f(-width,height,length);
    glVertex3f(-width,height,-length);

    glNormal3f(0.0f,0.0f,1.0f);
    glVertex3f(-width,-height,length);
    glVertex3f(width,-height,length);
    glVertex3f(width,height,length);
    glVertex3f(-width,height,length);

    glNormal3f(0.0f,0.0f,-1.0f);
    glVertex3f(-width,-height,-length);
    glVertex3f(-width,height,-length);
    glVertex3f(width,height,-length);
    glVertex3f(width,-height,-length);

    glEnd();

    glBegin(GL_TRIANGLES);

    float mag = sqrt(height*height+9*length*length/25);

    glNormal3f(3*length/5/mag,0.0f,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(height,-height,-8*length/5);
    glVertex3f(height,height,-8*length/5);

    glNormal3f(-3*length/5/mag,0.0f,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(-height,height,-8*length/5);
    glVertex3f(-height,-height,-8*length/5);

    glNormal3f(0.0f,3*length/5/mag,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(height,height,-8*length/5);
    glVertex3f(-height,height,-8*length/5);

    glNormal3f(0.0f,-3*length/5/mag,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(-height,-height,-8*length/5);
    glVertex3f(height,-height,-8*length/5);

    glEnd();

    glPushMatrix();
    glTranslatef(0.0f,11*height/5,6*height/5);
    glRotatef(90.0f,0.0f,0.0f,1.0f);
    draw_cylinder(4*height/3,width,10);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0.0f,11*height/5,-6*height/5);
    glRotatef(90.0f,0.0f,0.0f,1.0f);
    draw_cylinder(4*height/3,width,10);
    glPopMatrix();
}

void
draw_textured_quad(const double* pts, unsigned int texid)
{
    glPushAttrib(GL_ENABLE_BIT);

    //glColor3f(1.0f,1.0f,1.0f);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,texid);
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3d(pts[0], pts[1],  pts[2]);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3d(pts[3], pts[4],  pts[5]);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3d(pts[6], pts[7],  pts[8]);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3d(pts[9], pts[10], pts[11]);
    glEnd();

    glPopAttrib();
}

