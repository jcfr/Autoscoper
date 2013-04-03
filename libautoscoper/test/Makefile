TARGETS = filters ncc render

CXX = g++
CXXFLAGS = -Wall -g -O2
INCLUDES = -I../
LIBS = ../libautoscoper.a -ltiff

# Test OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
LIBS += -framework OpenCL -framework OpenGL -framework GLUT -L/opt/X11/lib
else
LIBS += -lOpenCL -lglut
endif

.PHONY:	clean

all:	$(TARGETS)

filters: filters.cpp ../libautoscoper.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) filters.cpp -o $@ $(LIBS)

ncc: ncc.cpp ../libautoscoper.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) ncc.cpp -o $@ $(LIBS)

render: render.cpp ../libautoscoper.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) render.cpp -o $@ $(LIBS)

clean:
	rm -f $(TARGETS) $(OBJDIR)/*.o
