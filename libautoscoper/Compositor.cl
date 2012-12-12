__kernel
void composite_kernel(__global const float* src1,
                      __global const float* src2,
                      __global float* dest,
                      size_t width,
                      size_t height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > width-1 || y > height-1) {
        return;
    }

    // src1 maps to orange and src2 to blue
    dest[3*(y*width+x)+0] = src1[y*width+x];
    dest[3*(y*width+x)+1] = src1[y*width+x]/2.0f+src2[y*width+x]/2.0f;
    dest[3*(y*width+x)+2] = src2[y*width+x];
}

// vim: ts=4 syntax=cpp noexpandtab
