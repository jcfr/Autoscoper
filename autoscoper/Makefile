TARGET = autoscoper

CXX = g++
CXXFLAGS = -Wall -g -m64
LDFLAGS =

GTK_PACKAGE := gtkglext-x11-1.0
GTK_PACKAGE_TEST = $(shell pkg-config --print-errors $(GTK_PACKAGE); echo $$?)

GTHREAD_PACKAGE := gthread-2.0
GTHREAD_PACKAGE_TEST = $(shell pkg-config --print-errors $(GTHREAD_PACKAGE); echo $$?)

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
CXXFLAGS += -D_MACOSX
LDFLAGS += -L/System/Library/Frameworks/OpenGL.framework/Libraries -framework GLUT
LIBS += -L$(CUDA_DIR)/lib -lcudart -L/opt/X11/lib
else
LIBS += -lglut -L$(CUDA_DIR)/lib64 -lcudart
endif

OBJDIR = obj
CXXFILES = $(shell ls *.cpp)
OBJS = $(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o)

.PHONE:	check-gtk check-cuda clean

all:	$(TARGET)

check-gtk:
ifeq ($(GTK_PACKAGE_TEST), 1)
	$(error pkg-config could not find GTK with gtkglext support)
endif
ifeq ($(GTHREAD_PACKAGE), 1)
	$(error pkg-config could not find the gthread library from glib)
endif

check-cuda:
ifndef CUDA_DIR
	$(error CUDA_DIR is undefined)
endif
	@ echo "CUDA_DIR = $(CUDA_DIR)"

$(TARGET) : check-gtk check-cuda $(OBJS)
	@ echo $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o
