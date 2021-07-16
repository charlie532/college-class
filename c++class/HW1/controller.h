#ifndef GROUP_CONTROLLER_INCLUDED
#define GROUP_CONTROLLER_INCLUDED

class Robot;
class Maze;
class Controller {
public:
    Controller(Maze *maze, Robot *robot);
    const int getNextCondition() const;
    const int getCurrentCondition() const;
    const int getNextDir() const;
    void setCurrentDir();
    void setNextStep(const int value);
    void input();
    void runRobot();
private:
    Robot *robot;
    Maze *maze;
};

#endif