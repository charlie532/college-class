#include "maze.cpp"
#include "robot.cpp"
#include "controller.cpp"

int main() {
    Robot robot;
    Maze maze;
    Controller controller(&maze, &robot);

    controller.input();
    controller.runRobot();
    cout << robot << endl;
    return 0;
}