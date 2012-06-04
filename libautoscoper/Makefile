TARGET = libautoscoper.a

CXX = g++
CXXFLAGS = -Wall -g -O2
NVCC = nvcc
NVCCFLAGS =
AR = ar
ARFLAGS = -cq

CXXFILES = $(shell ls *.cpp)
CUFILES = $(shell ls *.cu)

OBJDIR = obj
OBJS = \
	$(CXXFILES:%.cpp=$(OBJDIR)/%.cpp.o) \
	$(CUFILES:%.cu=$(OBJDIR)/%.cu.o)

INCLUDES = -I. -Icutil -I/usr/local/cuda/include

$(TARGET) : $(OBJS)
	@ echo $(OBJS)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJS)

$(OBJDIR)/%.cpp.o: %.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.cu.o: %.cu
	mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJDIR)/*.o
