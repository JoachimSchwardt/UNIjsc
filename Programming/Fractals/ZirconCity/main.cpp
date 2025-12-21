#include <iostream>
#include <fstream>
#include <sstream>
#include "glad.h"
#include "glfw3.h"

static const float quad[] = {
    -1, -1,
     1, -1,
    -1,  1,
     1,  1
};

GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
}

std::string load(const char* path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main()
{ //g++ main.cpp glad.c -lglfw -lGL -o lyapunovgl -O2
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(1920, 1080, "Lyapunov", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    auto vs_src = load("fullscreen.vert");
    auto fs_src = load("lyapunov.frag");

    GLuint prog = glCreateProgram();
    glAttachShader(prog, compile_shader(GL_VERTEX_SHADER, vs_src.c_str()));
    glAttachShader(prog, compile_shader(GL_FRAGMENT_SHADER, fs_src.c_str()));
    glLinkProgram(prog);

    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "a_min"), 3.4f);
    glUniform1f(glGetUniformLocation(prog, "a_max"), 4.0f);
    glUniform1f(glGetUniformLocation(prog, "b_min"), 2.5f);
    glUniform1f(glGetUniformLocation(prog, "b_max"), 3.4f);
    glUniform1i(glGetUniformLocation(prog, "n_iterations"), 2048);

    while (!glfwWindowShouldClose(win)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
}

