TARGETS = filters

CXX = g++
CXXFLAGS = -Wall -g -O2 -fPIC
INCLUDES = -I../
LIBS = ../libautoscoper.a -ltiff

# Test OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
LIBS += -framework OpenCL -framework OpenGL
else
LIBS += -L$(OPENCL_DIR)/lib64 -lOpenCL -lGL
endif

.PHONY:	clean

all:	$(TARGETS)

filters: filters.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) filters.cpp -o $@ $(LIBS)

clean:
	rm -f $(TARGETS) $(OBJDIR)/*.o
