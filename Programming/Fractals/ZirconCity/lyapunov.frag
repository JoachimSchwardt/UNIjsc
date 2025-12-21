#version 330 core

out vec4 rgba;
in vec2 uv;

uniform float a_min;
uniform float a_max;
uniform float b_min;
uniform float b_max;
uniform int n_iterations;

void main()
{
    float a = mix(a_min, a_max, uv.y);
    float b = mix(b_min, b_max, uv.x);

    float x = 0.5;
    float lyap = 0.0;

    for (int i = 0; i < n_iterations; ++i) {
        float r = (mod(i / 6, 2) == 1) ? a : b;
        x = r * x * (1.0 - x);
        float deriv = abs(r * (1.0 - 2.0 * x));
        lyap += log(max(deriv, 1e-8));
    }

    lyap /= float(n_iterations);

    // Simple colormap
    float c = clamp(lyap, -1.0, 0.6);
    if (c < 0.0) {
        rgba = vec4(0.9*(1.0+c), 0.9*(1.0+c), 0.8 + 0.1*(1.0+c), 1.0);
    } else {
        c /= 0.6;
        rgba = vec4(0.8 + 0.1*(1.0-c), 0.8 + 0.1*(1.0-c), 0.9*(1.0-c), 1.0);
    }
}

