#ifndef GROUP_MAZE_INCLUDED
#define GROUP_MAZE_INCLUDED

#include <string>
#include <vector>
using namespace std;
static constexpr int BEGIN = 0;
static constexpr int UNREACHED_PATH = -1;
static constexpr int OBSTACLE = -2;
static constexpr int UNREACHED_DIR = -3;

class Maze{
public:
    Maze();
    void initMapRow(const string line);
    void setSize(const int width, const int height);
    void setMap(const int y, const int x, const int value);
    void setDirection(const int y, const int x, const int value);
    const int getWidth() const;
    const int getHeight() const;
    const int getMap(const int y, const int x) const;
    const int getDirection(const int y, const int x) const;
private:
    vector<vector<int>> map;
    vector<vector<int>> direction;
    int width;
    int height;
};

#endif