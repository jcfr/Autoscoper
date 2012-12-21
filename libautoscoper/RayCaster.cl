// Render the volume using ray marching.
__kernel
void volume_render_kernel(__global float* buffer,
                          unsigned width, unsigned height,
                          float step, float intensity, float cutoff,
                          __constant float* viewport,
                          __constant int* flip,
                          __constant float* imv,
                          __read_only image3d_t image)
{
	const uint x = get_global_id(0);
    const uint y = get_global_id(1);

	const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
	                          CLK_ADDRESS_CLAMP_TO_EDGE |
	                          CLK_FILTER_LINEAR;

    if (x > width-1 || y > height-1) {
        return;
    }

    // Calculate the normalized device coordinates using the viewport
    const float u = viewport[0]+viewport[2]*(x/(float)width);
    const float v = viewport[1]+viewport[3]*(y/(float)height);

    // Determine the look ray in camera space.
    const float3 look = step*normalize((float3)(u, v, -2.0f));

    // Calculate the ray in world space.
	// Origin is the last column of the invModelView matrix
	float3 ray_origin = (float3)(imv[3], imv[7], imv[11]);
	// Origin is the invModelView x look
	float3 ray_direction = (float3)(
						dot(look, (float3)(imv[0], imv[1], imv[2])),
						dot(look, (float3)(imv[4], imv[5], imv[6])),
						dot(look, (float3)(imv[8], imv[9], imv[10])));

    // Find intersection with box.
    const float3 boxMin = (float3)(0.0f, 0.0f, -1.0f);
    const float3 boxMax = (float3)(1.0f, 1.0f, 0.0f);
	float near;
    float far;

    // Compute intersection of ray with all six planes.
    const float3 invDirection = (float3)(1.0f,1.0f,1.0f) / ray_direction;
    const float3 tBot = invDirection*(boxMin-ray_origin);
    const float3 tTop = invDirection*(boxMax-ray_origin);

    // Re-order intersections to find smallest and largest on each axis.
    const float3 tMin = fmin(tTop, tBot);
    const float3 tMax = fmax(tTop, tBot);

    // Find the largest tMin and the smallest tMax.
	near = fmax(fmax(tMin.x, tMin.y), tMin.z);
	far = fmin(fmin(tMax.x, tMax.y), tMax.z);

	if (!(far > near)) {
        buffer[y*width+x] = 0.0f;
        return;
    }
    
    // Clamp to near plane.
	if (near < 0.0f) near = 0.0f;
   
    // Preform the ray marching from back to front.
    float t = far;
    float density = 0.0f;
    while (t > near) {
        const float3 point = ray_origin+t*ray_direction;
        const float s = read_imagef(image, sampler, (float4)(
                (flip[0] == 1 ? 1.0f-point.x : point.x),
                (flip[1] == 1 ? point.y      : 1.0f-point.y),
                (flip[2] == 1 ? point.z+1.0f : -point.z),
				0) ).x;
        density += s > cutoff ? step*s : 0.0f;
        t -= 1.0f;
    }

    buffer[y*width+x] = clamp(density/intensity, 0.0f, 1.0f);
}

// vim: ts=4 syntax=cpp noexpandtab
