TARGET = autoscoper

CXX = g++
CXXFLAGS = -Wall -g -m64 -D_MACOSX
LDFLAGS =

LDFLAGS += -framework OpenGL -framework GLUT -framework Cocoa

CXXFILES = $(shell ls *.cpp)

GTK_PACKAGE=gtkglext-quartz-1.0

INCLUDES = \
	$(shell pkg-config --cflags-only-I $(GTK_PACKAGE)) \
	-I../libautoscoper \
	-I../math

LIBDIRS = \
	$(shell pkg-config --libs-only-L $(GTK_PACKAGE)) \
	-L../libautoscoper \
	-L/Developer/NVIDIA/CUDA-5.0/lib \
	-L/System/Library/Frameworks/OpenGL.framework/Libraries

LIBS = \
	$(shell pkg-config --libs-only-l $(GTK_PACKAGE)) \
	-lautoscoper \
	-ltiff \
	-lcudart \
	-lGLEW

OBJDIR = obj
OBJS = $(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o)

$(TARGET) : $(OBJS)
	@ echo $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBDIRS) $(LIBS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o
