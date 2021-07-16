#include <iostream>
#include <string>
#include <vector>
#define START 0
#define PATH -1
#define OBSTACLE -2
#define UNKNOWN_DIRECTION -3
using namespace std;

class Robot {
public:
    Robot();
    void predict();
    void setCoordinate();
    void turnRight();
    int getRobotDir();
    int getX();
    void setX(int x);
    int getY();
    void setY(int y);
    int getNextX();
    void setNextX(int nextX);
    int getNextY();
    void setNextY(int nextY);
    long long getTotal_steps();
    void setTotal_steps(long long total_steps);
private:
    int x;
    int y;
    int nextX; 
    int nextY; 
    int robotDir;
    long long total_steps;
};
class Maze {
public:
    Maze();
    void initMap(string line, int i);
    void setWidth(int width);
    int getWidth();
    void setHeight(int height);
    int getHeight();
    void setMap(int y, int x, int value);
    int getMap(int y, int x);
    void setDir(int y, int x, int dir);
    int getDir(int y, int x);
private:
    vector<vector<int>> map;
    vector<vector<int>> dir;
    int width, height;
};

Maze::Maze(){
};
void Maze::initMap(string line, int i){
    vector<int> mapRowi;
    vector<int> dirRowi;
    for(int j = 0; j < width; j++){
        if(line[j] == '#') mapRowi.push_back(OBSTACLE);
        else if(line[j] == '.') mapRowi.push_back(PATH);
        else if(line[j] == 'O') mapRowi.push_back(START);
        dirRowi.push_back(UNKNOWN_DIRECTION);
    }
    map.push_back(mapRowi);
    dir.push_back(dirRowi);
};
void Maze::setHeight(int height){
    this->height = height;
};
int Maze::getWidth(){
    return width;
};
void Maze::setWidth(int width){
    this->width = width;
};
int Maze::getHeight(){
    return height;
};
void Maze::setMap(int y, int x, int value){
    map[y][x] = value;
};
int Maze::getMap(int y, int x){
    return map[y][x];
};
void Maze::setDir(int y, int x, int dir){
    map[y][x] = dir;
};
int Maze::getDir(int y, int x){
    return dir[y][x];
};
Robot::Robot(){
    robotDir = 0;
}
ostream & operator << (ostream &os, Robot robot){
    os << robot.getX() << " " << robot.getY();
    return os;
}
void Robot::predict(){
    switch(robotDir){
        case 0:
            setNextY(y - 1);
            setNextX(x);
        case 1:
            setNextY(y);
            setNextX(x + 1);
        case 2:
            setNextY(y + 1);
            setNextX(x);
        case 3:
            setNextY(y);
            setNextX(x - 1);
        
    }
}
void Robot::setCoordinate(){
    setX(nextX);
    setY(nextY);
}
void Robot::turnRight(){
    robotDir = (robotDir + 1) % 4;
    predict();
}
int Robot::getRobotDir(){
    return robotDir;
}
int Robot::getX(){
    return x;
}
void Robot::setX(int x){
    this->x = x;
}
int Robot::getY(){
    return y;
}
void Robot::setY(int y){
    this->y = y;
}
int Robot::getNextX(){
    return nextX;
}
void Robot::setNextX(int nextX){
    this->nextX = nextX;
}
int Robot::getNextY(){
    return nextY;
}
void Robot::setNextY(int nextY){
    this->nextY = nextY;
}
long long Robot::getTotal_steps(){
    return total_steps;
}
void Robot::setTotal_steps(long long total_steps){
    this->total_steps = total_steps;
}
void input(Maze maze, Robot robot){
    int width;
    int height;
    long long total_steps;

    cin >> width >> height; cin.ignore();
    cin >> total_steps; cin.ignore();
    maze.setWidth(width);
    maze.setHeight(height);
    robot.setTotal_steps(total_steps);
    for (int i = 0; i < maze.getHeight(); i++) {
        string line;
        getline(cin, line);
        maze.initMap(line, i);
    }
}
void initialize(Maze maze, Robot robot){
    bool found_start = false;

    for (int i = 0; i < maze.getHeight(); i++) {
        for (int j = 0; j < maze.getWidth(); j++) {
            if (maze.getMap(i, j) == 0) {
                robot.setY(i);
                robot.setX(j);
                maze.setMap(i, j, 0);
                found_start = true;
                break;
            }
        }
        if (found_start == true) break;
    }
}
void robotRunning(Maze maze, Robot robot){
    int loop_count;
    bool found_loop = false;

    for (long long i = 0; i < robot.getTotal_steps(); i++) {
        maze.setDir(robot.getY(), robot.getX(), robot.getRobotDir());
        robot.predict();
        while (maze.getMap(robot.getNextY(), robot.getNextX()) == OBSTACLE) {
            robot.turnRight();
        }
        if ((maze.getMap(robot.getNextY(), robot.getNextX()) == PATH || maze.getMap(robot.getNextY(), robot.getNextX()) == START) && !found_loop) {
            maze.setMap(robot.getNextY(), robot.getNextX(), i + 1);
        } else {
            if (maze.getDir(robot.getNextY(), robot.getNextX()) == robot.getRobotDir() && !found_loop) {
                loop_count = maze.getMap(robot.getY(), robot.getX()) - maze.getMap(robot.getNextY(), robot.getNextX()) + 1;
                if (loop_count != 0) {
                    robot.setTotal_steps((robot.getTotal_steps() - i - loop_count) % loop_count);
                    i = 0;
                    found_loop = true;
                }
            }
        }
        robot.setCoordinate();
    }
}
int main(){
    Robot robot;
    Maze maze;

    input(maze, robot);
    initialize(maze, robot);
    robotRunning(maze, robot);
    cout << robot << endl;
    return 0;
}