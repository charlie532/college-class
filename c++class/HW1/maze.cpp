#include "maze.h"

Maze::Maze() {
    width = 0;
    height = 0;
};
void Maze::initMapRow(const string line) {
    vector<int> mapRowi;
    vector<int> dirRowi;
    for (int j = 0; j < width; ++j) {
        if (line[j] == '#') mapRowi.push_back(OBSTACLE);
        else if (line[j] == '.') mapRowi.push_back(UNREACHED_PATH);
        else if (line[j] == 'O') mapRowi.push_back(BEGIN);
        dirRowi.push_back(UNREACHED_DIR);
    }
    map.push_back(mapRowi);
    direction.push_back(dirRowi);
};
void Maze::setSize(int width, int height) {
    this->height = height;
    this->width = width;
};
void Maze::setMap(int y, int x, int value) {
    map[y][x] = value;
};
void Maze::setDirection(int y, int x, int value) {
    direction[y][x] = value;
};
const int Maze::getWidth() const {
    return width;
};
const int Maze::getHeight() const {
    return height;
};
const int Maze::getMap(int y, int x) const {
    return map[y][x];
};
const int Maze::getDirection(int y, int x) const {
    return direction[y][x];
};