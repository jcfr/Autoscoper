TARGETS = filters render

CXX = g++
CXXFLAGS = -Wall -g -O2 -fPIC
INCLUDES = -I../ -I../cutil -I$(CUDA_DIR)/include
LIBS = ../libautoscoper.a -ltiff

# Test OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
CXXFLAGS += -D_MACOSX
LIBS += -L$(CUDA_DIR)/lib -lcudart -framework OpenCL
else
LIBS += -L$(CUDA_DIR)/lib64 -lcudart -lOpenCL
endif

.PHONY:	check-cuda clean

all:	$(TARGETS)

check-cuda:
ifndef CUDA_DIR
	$(error CUDA_DIR is undefined)
endif
	@ echo "CUDA_DIR = $(CUDA_DIR)"

filters: check-cuda filters.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) filters.cpp -o $@ $(LIBS)

render: check-cuda render.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) render.cpp -o $@ $(LIBS)

clean:
	rm -f $(TARGETS) $(OBJDIR)/*.o
