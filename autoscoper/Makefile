TARGET = autoscoper

CXX = g++
CXXFLAGS = -Wall -g -m64

GTK_PACKAGE := gtkglext-quartz-1.0
GTK_PACKAGE_TEST = $(shell pkg-config --print-errors $(GTK_PACKAGE); echo $$?)

INCLUDES = \
	$(shell pkg-config --cflags $(GTK_PACKAGE)) \
	-I../libautoscoper \
	-I../math

LIBS = \
	$(shell pkg-config --libs $(GTK_PACKAGE)) \
	-L../libautoscoper -lautoscoper \
	-ltiff \
	-lGLEW

# Test OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
CXXFLAGS += -D_MACOSX
LIBS += -framework GLUT -L$(CUDA_DIR)/lib -lcudart
else
LIBS += -L$(CUDA_DIR)/lib64 -lcudart
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

check-cuda:
ifndef CUDA_DIR
	$(error CUDA_DIR is undefined)
endif
	@ echo "CUDA_DIR = $(CUDA_DIR)"

$(TARGET) : check-gtk $(OBJS)
	@ echo $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o
