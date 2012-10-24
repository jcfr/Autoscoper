TARGET = libautoscoper.a

CXX = g++
CXXFLAGS = -Wall -g -O2 -fPIC
NVCC = nvcc
NVCCFLAGS = -m64
AR = ar
ARFLAGS = cru

CXXFILES = $(shell ls *.cpp)
CUFILES = $(shell ls *.cu)
CLFILES = $(shell ls *.cl)

CLDIR = cl
KERNELS = $(CLFILES:%.cl=$(CLDIR)/%.cl.h)

OBJDIR = obj
OBJS = \
	$(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o) \
	$(CUFILES:%.cu=$(OBJDIR)/%.cu.o)

INCLUDES = -I. -I$(CLDIR) -Icutil -I$(CUDA_DIR)/include

.PHONY:	check-cuda clean

all:	$(TARGET)

check-cuda:
ifndef CUDA_DIR
	$(error CUDA_DIR is undefined)
endif
	@ echo "CUDA_DIR = $(CUDA_DIR)"

$(TARGET) : check-cuda $(KERNELS) $(OBJS)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.cu.o: %.cu
	mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(CLDIR)/%.cl.h: %.cl
	mkdir -p $(CLDIR)
	./cl-compile $< $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o $(CLDIR)/*.h

