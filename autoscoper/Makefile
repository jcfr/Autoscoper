TARGET = autoscoper

CXX = g++
CXXFLAGS = -Wall -g -m64 -DGLDEBUG
LDFLAGS =

GTK_PACKAGE := gtkglext-x11-1.0
GTK_PACKAGE_TEST = $(shell pkg-config --libs --cflags --print-errors $(GTK_PACKAGE); echo $$?)

GTHREAD_PACKAGE := gthread-2.0
GTHREAD_PACKAGE_TEST = $(shell pkg-config --libs --cflags --print-errors $(GTHREAD_PACKAGE); echo $$?)

INCLUDES = \
	$(shell pkg-config --cflags $(GTK_PACKAGE) $(GTHREAD_PACKAGE)) \
	-I../libautoscoper \
	-I../math

LIBS = \
	$(shell pkg-config --libs $(GTK_PACKAGE) $(GTHREAD_PACKAGE)) \
	-L../libautoscoper -lautoscoper \
	-ltiff \
	-lGLEW

# Test OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
LIBS += -framework OpenCL -framework OpenGL -framework GLUT -L/usr/X11/lib
else
LIBS += -lOpenCL -lglut
endif

OBJDIR = obj
CXXFILES = $(shell ls *.cpp)
OBJS = $(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o)

.PHONY:	check-gtk clean

all:	$(TARGET)

check-gtk:
ifeq ($(GTK_PACKAGE_TEST), 1)
	$(error pkg-config could not find GTK with gtkglext support)
endif
ifeq ($(GTHREAD_PACKAGE_TEST), 1)
	$(error pkg-config could not find the gthread library from glib)
endif

$(TARGET) : check-gtk $(OBJS)
	@ echo $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o
