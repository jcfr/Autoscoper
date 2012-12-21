TARGETS = filters ncc

CXX = g++
CXXFLAGS = -Wall -g -O2
INCLUDES = -I../
LIBS = ../libautoscoper.a -ltiff

# Test OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
LIBS += -framework OpenCL -framework OpenGL -framework GLUT
else
LIBS += -lOpenCL -lglut -lGL
endif

.PHONY:	clean

all:	$(TARGETS)

filters: filters.cpp ../libautoscoper.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) filters.cpp -o $@ $(LIBS)

ncc: ncc.cpp ../libautoscoper.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) ncc.cpp -o $@ $(LIBS)

clean:
	rm -f $(TARGETS) $(OBJDIR)/*.o
