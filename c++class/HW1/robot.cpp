#include "robot.h"
#include <iostream>

Robot::Robot(){
    x = 0;
    y = 0;
    nextX = 0; 
    nextY = 0; 
    robotDir = NORTH;
    totalSteps = 0;
}
ostream & operator << (ostream &os, Robot robot) {
    os << robot.getX() << " " << robot.getY();
    return os;
}
void Robot::predictNextPos() {
    switch(robotDir){
        case NORTH:
            setNextPosition(x, y - 1);
            break;
        case EAST:
            setNextPosition(x + 1, y);         
            break;
        case SOUTH:
            setNextPosition(x, y + 1);
            break;
        case WEST:
            setNextPosition(x - 1, y);
            break;
    }
}
void Robot::turnRight() {
    robotDir = (robotDir + 1) % 4;
    predictNextPos();
}
void Robot::setPosition(const int x, const int y) {
    this->x = x;
    this->y = y;
}
void Robot::setNextPosition(const int nextX, const int nextY) {
    this->nextX = nextX;
    this->nextY = nextY;
}
void Robot::setTotalSteps(const long long totalSteps) {
    this->totalSteps = totalSteps;
}
const int Robot::getRobotDir() const {
    return robotDir;
}
const int Robot::getX() const {
    return x;
}
const int Robot::getY() const {
    return y;
}
const int Robot::getNextX() const {
    return nextX;
}
const int Robot::getNextY() const {
    return nextY;
}
const long long Robot::getTotalSteps() const {
    return totalSteps;
}