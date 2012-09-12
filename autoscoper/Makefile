TARGET = autoscoper

CXX = g++
CXXFLAGS = -Wall -g -m64
LDFLAGS =

# Test OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
CXXFLAGS += -D_MACOSX
LDFLAGS += -framework OpenGL -framework GLUT -framework Cocoa
endif

# Test for GTK
GTK_PACKAGE := gtkglext-quartz-1.0
GTK_PACKAGE_TEST = $(shell pkg-config --print-errors $(GTK_PACKAGE); echo $$?)

INCLUDES = \
	$(shell pkg-config --cflags-only-I $(GTK_PACKAGE)) \
	-I../libautoscoper \
	-I../math

LIBDIRS = \
	$(shell pkg-config --libs-only-L $(GTK_PACKAGE)) \
	-L../libautoscoper \
	-L/Developer/NVIDIA/CUDA-5.0/lib

LIBS = \
	$(shell pkg-config --libs-only-l $(GTK_PACKAGE)) \
	-lautoscoper \
	-ltiff \
	-lcudart \
	-lGLEW

OBJDIR = obj
CXXFILES = $(shell ls *.cpp)
OBJS = $(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o)

all:	$(TARGET)

.PHONE:	check-gtk clean

check-gtk:
ifeq ($(GTK_PACKAGE_TEST), 1)
	$(error pkg-config could not find GTK with gtkglext support)
endif

$(TARGET) : check-gtk $(OBJS)
	@ echo $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBDIRS) $(LIBS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o
