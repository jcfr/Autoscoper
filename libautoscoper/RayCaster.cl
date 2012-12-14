struct Ray
{
    float3 origin;
    float3 direction;
};

// Render the volume using ray marching.
__kernel
void volume_render_kernel(__global float* buffer, size_t width, size_t height,
                          float step, float intensity, float cutoff,
                          float4 viewport, int3 flip,
                          __global __read_only float* imv,
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
    const float u = viewport.x+viewport.z*(x/(float)width);
    const float v = viewport.y+viewport.w*(y/(float)height);

    // Determine the look ray in camera space.
    const float3 look = step*normalize((float4)(u, v, -2.0f));

    // Calculate the ray in world space.
    Ray ray;
	// Origin is the last column of the invModelView matrix
	ray.origin = (float3)(imv[3], imv[7], imv[11]);
	// Origin is the invModelView x look
	ray.direction = (float3)(
						dot(look, (float3)(imv[0], imv[1], imv[2])),
						dot(look, (float3)(imv[4], imv[5], imv[6])),
						dot(look, (float3)(imv[8], imv[9], imv[10])));

    // Find intersection with box.
    const float3 boxMin = (float3)(0.0f, 0.0f, -1.0f);
    const float3 boxMax = (float3)(1.0f, 1.0f, 0.0f);
	float near;
    float far;

    // Compute intersection of ray with all six planes.
    const float3 invDirection = float3(1.0f,1.0f,1.0f) / ray.direction;
    const float3 tBot = invDirection*(boxMin-ray.origin);
    const float3 tTop = invDirection*(boxMax-ray.origin);

    // Re-order intersections to find smallest and largest on each axis.
    const float3 tMin = fminf(tTop, tBot);
    const float3 tMax = fmaxf(tTop, tBot);

    // Find the largest tMin and the smallest tMax.
	near = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
	far = fminf(fminf(tMax.x, tMax.y), tMax.z);

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
        const float3 point = ray.origin+t*ray.direction;
        const float s = read_imagef(image, sample, (float3)(
                (flip.x == 1 ? 1.0f-point.x : point.x),
                (flip.y == 1 ? point.y      : 1.0f-point.y),
                (flip.z == 1 ? point.z+1.0f : -point.z))).x;
        density += s > cutoff ? step*sample : 0.0f;
        t -= 1.0f;
    }

    buffer[y*width+x] = clamp(density/intensity, 0.0f, 1.0f);
}

// vim: ts=4 syntax=cpp noexpandtab
