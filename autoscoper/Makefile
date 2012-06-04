TARGET = autoscoper

CXX = g++
CXXFLAGS = -Wall -g -export-dynamic

CXXFILES = $(shell ls *.cpp)

INCLUDES = \
	$(shell pkg-config --cflags-only-I gtkglext-x11-1.0) \
	-I../libautoscoper \
	-I../math

LIBDIRS = \
	$(shell pkg-config --libs-only-L gtkglext-x11-1.0) \
	-L../libautoscoper \
    -L/usr/local/cuda/lib64

LIBS = \
	$(shell pkg-config --libs gtkglext-x11-1.0) \
	-lautoscoper \
	-ltiff \
	-lcudart \
	-lglut \
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
