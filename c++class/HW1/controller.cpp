#include "robot.h"
#include "maze.h"
#include "controller.h"
#include <iostream>

Controller::Controller(Maze *maze, Robot *robot) {
    this->maze = maze;
    this->robot = robot;
}
const int Controller::getNextCondition() const {
    return maze->getMap(robot->getNextY(), robot->getNextX());
}
const int Controller::getCurrentCondition() const {
    return maze->getMap(robot->getY(), robot->getX());
}
const int Controller::getNextDir() const {
    return maze->getDirection(robot->getNextY(), robot->getNextX());
}
void Controller::setCurrentDir() {
    maze->setDirection(robot->getY(), robot->getX(), robot->getRobotDir());
}
void Controller::setNextStep(const int value) {
    maze->setMap(robot->getNextY(), robot->getNextX(), value + 1);
}
void Controller::input() {
    int width = 0;
    int height = 0;
    long long totalSteps = 0;

    cin >> width >> height; cin.ignore();
    cin >> totalSteps; cin.ignore();
    maze->setSize(width, height);
    robot->setTotalSteps(totalSteps);
    for (int i = 0; i < maze->getHeight(); ++i) {
        string line;
        getline(cin, line);
        maze->initMapRow(line);
        for (int j = 0; j < maze->getWidth(); ++j) {
            if (maze->getMap(i, j) == BEGIN){
                robot->setPosition(i, j);
                maze->setDirection(i, j, 0);
            }
        }
    }
}
void Controller::runRobot(){
    long long remainingSteps = robot->getTotalSteps();
    bool haveLoop = false;

    for (long long i = 0; i < remainingSteps; ++i) {
        setCurrentDir();
        robot->predictNextPos();
        while (getNextCondition() == OBSTACLE) {
            robot->turnRight();
        }
        if ((getNextCondition() == UNREACHED_PATH || getNextCondition() == BEGIN) && !haveLoop) {
            setNextStep(i);
        } else {
            if (getNextDir() == robot->getRobotDir() && !haveLoop) {
                int loopStepCount = getCurrentCondition() - getNextCondition() + 1;
                if (loopStepCount != 0) {
                    remainingSteps = (remainingSteps - i - loopStepCount) % loopStepCount;
                    robot->setTotalSteps((robot->getTotalSteps() - i - loopStepCount) % loopStepCount);
                    i = 0;
                    haveLoop = true;
                }
            }
        }
        robot->setPosition(robot->getNextX(), robot->getNextY());
    }
}